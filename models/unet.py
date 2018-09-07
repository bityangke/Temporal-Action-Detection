from models.feature_extractor import I3DFeatureExtractor
from models.i3d import I3D
import torch
from torch import nn

def set_module_grad(module):
    for p in module.parameters():
        p.requires_grad = False

class UntrimmedNet(torch.nn.Module):
    def __init__(self, num_class, modality, reduce=False, require_feature=False):
        super(UntrimmedNet, self).__init__()
        feature_length = 400
        assert modality in ['rgb', 'flow']
        self.modality = modality
        self.require_feat = require_feature
        self.fe = I3D(num_classes=400, modality=modality)
        self.dropout = nn.Dropout()
        self.relu = nn.ReLU()
        self.clf = nn.Linear(feature_length, num_class)
        self.attention = torch.nn.Linear(feature_length, 1)
        if reduce:
            ml = [self.fe.conv3d_1a_7x7,
             self.fe.conv3d_2b_1x1,
             self.fe.conv3d_2c_3x3,
             self.fe.mixed_3b,
             self.fe.mixed_3c]
            for m in ml:
                set_module_grad(m)

    def forward(self, data):
        feat = self.fe(data)
        feat = self.relu(feat)
        feat = self.dropout(feat)
        logit = self.clf(feat)
        weight = self.attention(feat)
        if self.require_feat:
            return logit, weight, feat
        return logit, weight


# class UntrimmedNet(torch.nn.Module):
#     def __init__(self, num_class, checkpoint=None):
#         super(UntrimmedNet, self).__init__()
#         feature_length = 800
#         self.fe = I3DFeatureExtractor()
#         self.clf = nn.Linear(feature_length, num_class)
#         self.attention = torch.nn.Linear(feature_length, 1)
#         if checkpoint:
#             from collections import OrderedDict
#             print(("=> loading checkpoint '{}'".format(checkpoint)))
#             checkpoint = torch.load(checkpoint)
#             fe_new_ckpt = OrderedDict()
#             clf_new_ckpt = OrderedDict()
#             for key in checkpoint['state_dict'].keys():
#                 if key[7]=='0':
#                     fe_new_ckpt[key[9:]] = checkpoint['state_dict'][key]
#                 elif key[7]=='1':
#                     clf_new_ckpt[key[9:]] = checkpoint['state_dict'][key]
#             self.fe.load_state_dict(fe_new_ckpt)
#             clf_new_new_ckpt = OrderedDict()
#             for key in clf_new_ckpt.keys():
#                 if key[11]=='2':
#                     nkey = key.replace('2','3')
#                     clf_new_new_ckpt[nkey] = clf_new_ckpt[key]
#                 elif key[11]=='4':
#                     nkey = key.replace('4', '6')
#                     clf_new_new_ckpt[nkey] = clf_new_ckpt[key]
#                 else:
#                     clf_new_new_ckpt[key] = clf_new_ckpt[key]
#             self.clf.load_state_dict(clf_new_new_ckpt)
#
#     def forward(self, r, f):
#         # size = r.shape[0]*r.shape[1]
#         # feat = self.fe([r.reshape(size, *(r.shape[2:])), f.reshape(size, *(f.shape[2:]))])
#         feat = self.fe([r, f])
#         # weight = self.attention(feat)ss
#         # logit = self.clf(feat)
#         # return logit, weight
#         return feat, None

class UntrimmedNetBack(torch.nn.Module):    # 注意，feature尺寸是400！只有一个流。
    def __init__(self, num_class, checkpoint=None):
        super(UntrimmedNetBack, self).__init__()
        feature_length = 400
        self.fe = I3DFeatureExtractor()
        self.dropout = nn.Dropout(0.7)
        self.clf = nn.Linear(feature_length, num_class)
        self.attention = torch.nn.Linear(feature_length, 1)

    def forward(self, feat):
        feat = self.dropout(feat)
        weight = self.attention(feat)
        logit = self.clf(feat)
        return logit, weight