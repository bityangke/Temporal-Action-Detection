import os
import time

import numpy as np
import pandas as pd
import torch

from ops.annotation import prepare_one_video_annotation, build_annotation_dict
from dataset import build_video_dataset
from models.feature_extractor import I3DFeatureExtractor
from models.module import load_fe_from_i3d, Classifier
from collections import OrderedDict

if __name__ == '__main__':

    ######################### HYPER PARAMETER ##############################

    # i3d_model_checkpoint = "models/0809_ucf_act.pth.tar"
    # i3d_model_checkpoint = "result/0811_epoch8_act_model_ckpt.pth.tar"    # Act clf
    index_file = 'ucf_index.txt'
    dataset = 'ucf'  # thumos_test is in fact the testing set.
    num_class = 101

    # num_class = 2

    ######################### HYPER PARAMETER ##############################
    from ops.utils import AverageMeter
    ds = build_video_dataset(dataset, train=False)
    # ds, annotation_root = ds["dataset"], ds["annotation_root"]
    ds = ds["dataset"]

    prefix = "features/kinetics/{}/".format(dataset)
    # prefix = "features/{}_act/".format(dataset)
    # if not os.path.exists(prefix):
    #     os.mkdir(prefix)
    st = time.time()
    # feature_length = 800
    path_list = []
    file_list = []
    label_list = []
    vid = []
    annotation_list = []
    # annotation_dict = build_annotation_dict(annotation_root=annotation_root, index_file=index_file)
    ant = []

    feature = []
    last_ind = 0
    last_path = ""
    last_label = 0
    verbose = False

    testtime = AverageMeter()

    #### build network
    # fe = I3DFeatureExtractor()
    # fe = load_fe_from_i3d(fe, i3d_model_checkpoint)
    # fe = torch.nn.DataParallel(fe, device_ids=[i for i in range(torch.cuda.device_count())]).cuda()
    # fe.eval()

    ####
    # clf = Classifier(feature_length, num_class, isbn=True)
    # fe = I3DFeatureExtractor()
    # I3DClassifier = torch.nn.Sequential(
    #     fe,
    #     clf
    # )
    # ckpt = torch.load(i3d_model_checkpoint)
    # new_key = OrderedDict()
    # for key in ckpt['state_dict'].keys():
    #     new_key[key[7:]] = ckpt['state_dict'][key]
    # I3DClassifier.load_state_dict(new_key)
    # I3DClassifier = torch.nn.DataParallel(I3DClassifier, device_ids=[i for i in range(torch.cuda.device_count())]).cuda()

    ###################### CHOOSE YOUR MODEL!!!!!!!!! ##################

    # model = I3DClassifier
    from models.unet import UntrimmedNet
    model = UntrimmedNet(num_class)
    # unet_ckpt = "result/0826_1613_unet/0826_1613_unet_epoch2_model.pth.tar"
    # unet_ckpt = "result/0829_0018_unet_ucf/0829_0018_unet_ucf_epoch9_model.pth.tar"
    # unet_ckpt = torch.load(unet_ckpt)
    # new_ckpt = OrderedDict()
    # for key in unet_ckpt['state_dict'].keys():
    #     new_ckpt[key[7:]] = unet_ckpt['state_dict'][key]
    # model.load_state_dict(new_ckpt)
    model = torch.nn.DataParallel(model, device_ids=[i for i in range(torch.cuda.device_count())]).cuda()
    model.eval()
    ###################### CHOOSE YOUR MODEL!!!!!!!!! ##################



    ###################### START INDEX ######################
    # START_INDEX = 110
    # START_BATCH = 0
    # for ind, i in enumerate(ds.lookup):
    #     if i[0]==START_INDEX:
    #         START_BATCH = ind
    #         break
    loader = torch.utils.data.DataLoader(
            ds, batch_size=32, shuffle=False,
            num_workers=12, pin_memory=True)
    #########################################################




    #### get start
    for ind, i in enumerate(loader):
        # if ind<START_BATCH:
        #     print("Skip index: ", ind)
        #     continue
        # i is a video.
        with torch.no_grad():
            r, f, path, label, video_index, seg_index, verify = i
            r = r.cuda()
            f = f.cuda()
            video_index = video_index.detach().cpu().numpy()
            feat, _ = model(r, f)
            feat = feat.detach().cpu().numpy()
            if ind==0:
                print("First feat", feat)
            label = label.detach().cpu().numpy()
        unique = np.unique(video_index)
        if verbose:
            print("=====Enter new data Ind {}=====\nlast ind: {}\nunique: {}\nvideo index tensor: {}".format(ind,
                                                                                                             last_ind,
                                                                                                             unique,
                                                                                                             video_index))
        # to deal with the situation that the class of last batch: [0, 0, 0, 0] and the new batch: [1, 1, 1]
        # In normal situation we expect last batch: [0, 0, 0, 1] and the new batch: [1, 1, 2, 2]
        if last_ind != unique[0]:
            if verbose:
                print('Find a ISOLATE!', last_ind, unique, video_index)
            file_name = prefix + "{:05d}".format(last_ind) + ".npy"
            feature = np.concatenate(feature, 0)
            np.save(file_name, feature)
            path_list.append(last_path)
            label_list.append(last_label)
            file_list.append(file_name)
            # ant = prepare_one_video_annotation(annotation_dict, last_path)
            annotation_list.append(ant)
            vid.append(last_ind)
            duration = time.time() - st
            st = time.time()
            testtime.update(duration)
            print(
                """Index: {}\nVideo: {}\nSave@: {}\nLabel: {}\nFeature shape: {}\nAnnotations: {}\nTime {:.3f}({:.3f})=====""".format(
                    last_ind, last_path,
                    file_name,
                    last_label,
                    feature.shape, ant, duration, testtime.avg))
            feature = []

        # 为了防止batchsize过大，包含了多个视频。
        for i in range(len(unique) - 1):
            if verbose:
                print("Enter EDGE!", unique, i)
            feature.append(feat[video_index == unique[i]])
            index_in_batch = np.argwhere(video_index == unique[i])[0][0]
            file_name = prefix + "{:05d}".format(unique[i]) + ".npy"

            feature = np.concatenate(feature, 0)
            assert verify[index_in_batch] == feature.shape[0], (
                verify[index_in_batch], feature.shape[0], unique, index_in_batch, video_index)
            np.save(file_name, feature)

            path_list.append(path[index_in_batch])
            # ant = prepare_one_video_annotation(annotation_dict, path[index_in_batch])
            annotation_list.append(ant)
            label_list.append(label[index_in_batch])
            file_list.append(file_name)
            vid.append(unique[i])
            duration = time.time() - st
            st = time.time()
            testtime.update(duration)
            print(
                """Index: {}\nVideo: {}\nSave@: {}\nLabel: {}\nFeature shape: {}\nAnnotations: {}\nTime {:.3f}({:.3f})=====""".format(
                    last_ind, last_path,
                    file_name,
                    last_label,
                    feature.shape, ant, duration, testtime.avg))
            feature = []
        feature.append(feat[video_index == unique[-1]])
        assert video_index[-1] == unique[-1]
        last_ind = video_index[-1]
        last_path = path[-1]
        last_label = label[-1]

        if ind % 500 == 0:
            df = pd.DataFrame(
                {'video_path': path_list, "feature_path": file_list, 'label': label_list, 'video_id': vid,
                 'annotation': annotation_list})
            df.to_csv(prefix + "data.csv", index=False)
            print("Checkpoint saved! Time elapse: {}, {} videos finished!".format(time.time() - st, last_ind))
    df = pd.DataFrame({'video_path': path_list, "feature_path": file_list, 'label': label_list, 'video_id': vid,
                       'annotation': annotation_list})
    df.to_csv(prefix + "data.csv", index=False)

    et = time.time()
    print('Time elapse: ', et - st)
