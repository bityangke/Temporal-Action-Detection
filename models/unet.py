from models.feature_extractor import I3DFeatureExtractor
import torch
from torch import nn

class UntrimmedNet(torch.nn.Module):
    def __init__(self, num_class, checkpoint=None):
        super(UntrimmedNet, self).__init__()
        feature_length = 800
        self.fe = nn.Sequential(
            I3DFeatureExtractor(),
            nn.Dropout()
        )
        self.clf = nn.Linear(feature_length, num_class)
        self.attention = torch.nn.Linear(feature_length, 1)
        if checkpoint:
            from collections import OrderedDict
            print(("=> loading checkpoint '{}'".format(checkpoint)))
            checkpoint = torch.load(checkpoint)
            fe_new_ckpt = OrderedDict()
            clf_new_ckpt = OrderedDict()
            for key in checkpoint['state_dict'].keys():
                if key[7]=='0':
                    fe_new_ckpt[key[9:]] = checkpoint['state_dict'][key]
                elif key[7]=='1':
                    clf_new_ckpt[key[9:]] = checkpoint['state_dict'][key]
            self.fe.load_state_dict(fe_new_ckpt)
            clf_new_new_ckpt = OrderedDict()
            for key in clf_new_ckpt.keys():
                if key[11]=='2':
                    nkey = key.replace('2','3')
                    clf_new_new_ckpt[nkey] = clf_new_ckpt[key]
                elif key[11]=='4':
                    nkey = key.replace('4', '6')
                    clf_new_new_ckpt[nkey] = clf_new_ckpt[key]
                else:
                    clf_new_new_ckpt[key] = clf_new_ckpt[key]
            self.clf.load_state_dict(clf_new_new_ckpt)

    def forward(self, r, f):
        size = r.shape[0]*r.shape[1]
        feat = self.fe([r.reshape(size, *(r.shape[2:])), f.reshape(size, *(f.shape[2:]))])
        weight = self.attention(feat)
        logit = self.clf(feat)
        return logit, weight