import os
import time
from collections import OrderedDict
import copy
import torch
from torch import nn

from dataset import build_video_dataset
from models.unet import UntrimmedNet
from ops.utils import AverageMeter

###### Hyper-parameter For This File
num_class = 101
db_size = 1000
ckpt_rgb_path = ''
ckpt_flow_path = ''  # Should be look like: result/aaaa/aaa_epoch1_model.pth.tar
model_flow = UntrimmedNet(num_class, 'flow')
save_path = ''
batch_size = 64
alpha = 0.5
######

softmax_dim0 = nn.Softmax(0)
softmax_dim1 = nn.Softmax(1)
softmax_dim2 = nn.Softmax(2)


def get_score(logit_rgb, weight_rgb, logit_flow, weight_flow, label, alpha=0.5, softmax_type='in'):
    assert isinstance(label, int)
    if softmax_type == 'in':
        weight_rgb = torch.squeeze(softmax_dim0(weight_rgb), 1)
        logit_rgb = softmax_dim1(logit_rgb)[:, label]
        output_rgb = weight_rgb * logit_rgb  # ouput shape = [length]
    elif softmax_type == 'out':
        raise NotImplementedError

    if softmax_type == 'in':
        weight_flow = torch.squeeze(softmax_dim0(weight_flow), 1)
        logit_flow = softmax_dim1(logit_flow)[:, label]
        output_flow = weight_flow * logit_flow  # ouput shape = [length]
    elif softmax_type == 'out':
        raise NotImplementedError
    return output_rgb * (1 - alpha) + output_flow * alpha, logit_rgb, weight_rgb, logit_flow, weight_flow

def build_distance_matrix(dc, beta):
    s_matrix = torch.zeros((len(dc), len(dc)))
    for i in range(0, len(dc)-1):
        for j in range(i+1, len(dc)):
            s_matrix[i, j] = - beta * torch.dist(dc[i]['feat_cat'], dc[j]['feat_cat'])
    return torch.exp(s_matrix)





def build_model(ckpt, modality):
    model = UntrimmedNet(num_class, modality, require_feature=True)
    if os.path.isfile(ckpt):
        print(("=> loading checkpoint '{}'".format(ckpt)))
        checkpoint = torch.load(ckpt)
        new = OrderedDict()
        for key in checkpoint['state_dict'].keys():
            if key[7:] in model.state_dict():
                new[key[7:]] = checkpoint['state_dict'][key]
        model.load_state_dict(new)
    else:
        print(("=> no checkpoint found at '{}'".format(ckpt)))
    model = torch.nn.DataParallel(model, device_ids=[i for i in range(torch.cuda.device_count())]).cuda()
    model.eval()
    return model


def build_database(model_rgb, model_flow, dataset, threshold):
    assert isinstance(threshold, float)

    model_rgb.eval()
    model_flow.eval()
    end = time.time()

    data_time = AverageMeter()

    database = [[] for _ in range(num_class)]

    for ind, (gen, label, _, _, length) in dataset:
        # gen(), label, rgb_record.path, video_index, self.segment_num
        logit_rgb = []
        logit_flow = []
        weight_rgb = []
        weight_flow = []
        feat_rgb = []
        feat_flow = []
        for seg in gen:
            data_time.update(time.time() - end)
            r, f = seg
            with torch.no_grad():
                r = r.cuda()
                f = f.cuda()
                lr, wr, fr = model_rgb(r)
                lf, wf, ff = model_flow(f)
            logit_rgb.append(lr)
            logit_flow.append(lf)
            weight_rgb.append(wr)
            weight_flow.append(wf)
            feat_rgb.append(fr)
            feat_flow.append(f)
        logit_rgb = torch.cat(logit_rgb)
        logit_flow = torch.cat(logit_flow)
        weight_rgb = torch.cat(weight_rgb)
        weight_flow = torch.cat(weight_flow)
        feat_rgb = torch.cat(feat_rgb)
        feat_flow = torch.cat(feat_flow)
        if isinstance(label, int):
            score, score_rgb, weight_rgb, score_flow, weight_flow \
                = get_score(logit_rgb, weight_rgb, logit_flow, weight_flow, label, alpha)  # [length]
            for sind, s in enumerate(score):
                if s > threshold:
                    database[label].append({'feat_rgb': feat_rgb[sind],
                                            'feat_flow': feat_flow[sind],
                                            'feat_cat': torch.cat([feat_rgb[sind], feat_flow[sind]], 1),
                                            'score': s,
                                            'raw_score_rgb': score_rgb[sind],
                                            'raw_score_flow': score_flow[sind],
                                            'attention_rgb': weight_rgb[sind],
                                            'attention_flow': weight_flow[sind]})

        elif isinstance(label, list):
            for lab in label:
                score, score_rgb, weight_rgb, score_flow, weight_flow \
                    = get_score(logit_rgb, weight_rgb, logit_flow, weight_flow, lab, alpha)
                if s > threshold:
                    database[lab].append({'feat_rgb': feat_rgb[sind],
                                          'feat_flow': feat_flow[sind],
                                          'feat_cat': torch.cat([feat_rgb[sind], feat_flow[sind]], 1),
                                          'score': s,
                                          'raw_score_rgb': score_rgb[sind],
                                          'raw_score_flow': score_flow[sind],
                                          'attention_rgb': weight_rgb[sind],
                                          'attention_flow': weight_flow[sind]})
    print("Database has generated. It has {} classes, and num of element in each class is: {}".format(len(database),
                                                                                                      [len(dc) for dc in
                                                                                                       database]))
    return database


def refine_database(db, size):
    assert isinstance(size, int)
    new_db = []
    for dc in db:   # dc mean ``d_{class}, the sub-database for a specified class.''
        dc = sorted(dc, key=lambda dic: -dic['score'])
        if len(dc)<size:
            dc = [copy.copy(dc[:size-len(dc)])] + dc
        else:
            del dc[size:]
        new_db.append(dc)
    return new_db



def get_q(db, max_iter=100):
    for dc in db:
        s_matrix = build_distance_matrix(dc)

        while True:
            z = torch.matmul(s_matrix, q)
            z = 1/z
            n = 1/len(dc) * torch.matmul(s, z)
            q = torch.dot(n, q)




def save(db, q, path):
    suffix = 'database'
    path = os.path.join(path, suffix)
    torch.save(db, path)
    torch.save(q, path)
    print("Database and Q matrix has been saved at: ", path)


if __name__ == '__main__':
    # build the models
    model_rgb = build_model(ckpt_rgb_path, 'rgb')
    model_flow = build_model(ckpt_flow_path, 'flow')

    # build dataset\
    ds = build_video_dataset('thumos_validation', False, single=True, single_batch_size=batch_size)['dataset']

    db = build_database(model_rgb, model_flow, ds)

    db = refine_database(db, 1000)

    q = get_q(db)

    save(db, q, save_path)

    print("Database and Q Matrix are generated, save at {}".format(save_path))
