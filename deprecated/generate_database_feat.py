import copy
import os
import time
from collections import OrderedDict

import torch
from torch import nn

from dataset import FeatureDataset
from models.unet import UntrimmedNet, UntrimmedNetBack
from ops.fuser import Fuser
from ops.utils import AverageMeter

###### Hyper-parameter For This File
num_class = 101
db_size = 1000
ckpt_rgb_path = 'result/0831_2154_unet_rgb/0831_2154_unet_rgb_epoch499_model.pth.tar'
ckpt_flow_path = 'result/0831_2157_unet_flow/0831_2157_unet_flow_epoch499_model.pth.tar'  # Should be look like: result/aaaa/aaa_epoch1_model.pth.tar
model_flow = UntrimmedNetBack(num_class)
model_rgb = UntrimmedNet(num_class)
save_path = 'result/test/ '
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
    for i in range(0, len(dc) - 1):
        for j in range(i + 1, len(dc)):
            s_matrix[i, j] = - beta * torch.dist(dc[i]['feat_cat'], dc[j]['feat_cat'])
    return torch.exp(s_matrix)


def build_model(ckpt, model, modality):
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

    for i, (input, label, _, _, _) in enumerate(dataset):
        with torch.no_grad():
            input_var = torch.from_numpy(input).cuda()
            target_var = torch.Tensor([label]).type(torch.cuda.LongTensor)
            assert isinstance(alpha, float) and (alpha < 1) and (alpha > 0), alpha
            logit_rgb, weight_rgb = model_rgb(input_var[:, :400])
            logit_flow, weight_flow = model_flow(input_var[:, 400:])

        if isinstance(label, int):
            score, score_rgb, weight_rgb, score_flow, weight_flow \
                = get_score(logit_rgb, weight_rgb, logit_flow, weight_flow, label, alpha)  # [length]
            for sind, s in enumerate(score):
                if s > threshold:
                    database[label].append({'feat_rgb': input[sind, :400],
                                            'feat_flow': input[sind, 400:],
                                            'feat_cat': input[sind, :],
                                            'score': s,
                                            'raw_score_rgb': score_rgb[sind],
                                            'raw_score_flow': score_flow[sind],
                                            'attention_rgb': weight_rgb[sind],
                                            'attention_flow': weight_flow[sind]})
        elif isinstance(label, list):
            for lab in label:
                assert isinstance(lab, int)
                score, score_rgb, weight_rgb, score_flow, weight_flow \
                    = get_score(logit_rgb, weight_rgb, logit_flow, weight_flow, lab, alpha)
                if s > threshold:
                    database[lab].append({'feat_rgb': input[sind, :400],
                                          'feat_flow': input[sind, 400:],
                                          'feat_cat': input[sind, :],
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
    for dc in db:  # dc mean ``d_{class}, the sub-database for a specified class.''
        dc = sorted(dc, key=lambda dic: -dic['score'])
        if len(dc) < size:
            dc = [copy.copy(dc[:size - len(dc)])] + dc
        else:
            del dc[size:]
        new_db.append(dc)
    return new_db


def get_q(db, max_iter=100, threshold=1e-3):
    for dc in db:
        s_matrix = build_distance_matrix(dc)

        while torch.max(n):
            z = torch.matmul(s_matrix, q)
            z = 1 / z
            n = (1 / len(dc)) * torch.matmul(s, z)
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

    ds = FeatureDataset('features/kinetics/thumos_validation/data.csv', Fuser(fuse_type='none'))

    db = build_database(model_rgb, model_flow, ds)

    db = refine_database(db, 1000)

    q = get_q(db)

    save(db, q, save_path)

    print("Database and Q Matrix are generated, save at {}".format(save_path))
