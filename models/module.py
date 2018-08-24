import os
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn

from ops.distributions import Categorical, DiagGaussian
from ops.utils import init, init_normc_

action_dict = {
    0: "Do nothing",
    1: "start - 1",
    2: "start + 1",
    3: "end - 1",
    4: "end + 1"
}


class Policy(nn.Module):
    def __init__(self, list_input, action_space, output_size):
        super(Policy, self).__init__()

        self.base = MLPBase(list_input, outputsize=output_size)
        self.base = torch.nn.DataParallel(self.base)

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(output_size, num_outputs)
        else:
            raise NotImplementedError

    def forward(self, inputs, states, masks):
        raise NotImplementedError

    def act(self, inputs, states, masks, deterministic=False):
        value, actor_features, states = self.base(inputs, states, masks)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
            # print("[POLY]\tDeterministic action {}".format(action))
        else:
            action = dist.sample()
            # print("[POLY]\tStocastic action {}".format(action))
        action_log_probs = dist.log_probs(action)
        return value, action, action_log_probs, states

    def get_value(self, inputs, states, masks):
        value, _, _ = self.base(inputs, states, masks)
        return value

    def evaluate_actions(self, inputs, states, masks, action):
        """
        :param inputs:
        :param states:
        :param masks:
        :param action:
        :return:
        """
        value, actor_features, states = self.base(inputs, states, masks)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, states

class CNNBase(nn.Module):
    def __init__(self, context_length, feature_length, class_num):
        super(CNNBase, self).__init__()
        self.class_num = class_num
        self.context_length = context_length

        channel = 64

        def buildCNN(i):
            padding = (0, 0, i//2, i//2) if i%2==1 else (0, 0, int(i/2)-1, int(i/2))
            pad = torch.nn.ZeroPad2d(padding)
            cnn = nn.Conv2d(1, channel, (i, feature_length))
            return nn.Sequential(pad, cnn)

        self.cnns = nn.ModuleList([buildCNN(i) for i in range(1, self.context_length+1)])
        self.maxpool = nn.MaxPool1d
        # self.maxpool = nn.AvgPool1d
        self.fc = nn.Linear(channel*context_length, int(channel*context_length/2))
        self.fc2 = nn.Linear(int(channel*context_length/2), class_num)
        self.softmax = nn.Softmax(1)

    def forward(self, x):
        cat = []
        for cnn in self.cnns:
            cat.append(cnn(x))
        cat = torch.cat(cat, 1)
        cat = torch.squeeze(cat, 3)
        cat = self.maxpool(cat.shape[2])(cat)
        cat = torch.squeeze(cat, 2)
        cat = self.fc(cat)
        logit = self.fc2(cat)
        prob = self.softmax(logit)
        return prob

class MLPBase(nn.Module):
    def __init__(self, list_inputs, outputsize):
        super(MLPBase, self).__init__()
        assert isinstance(list_inputs, list)
        assert len(list_inputs)==3
        self.list_inputs = list_inputs
        self.output_size = outputsize   # outputsize 是隐含层输出数目！
        init_ = lambda m: init(m,
              init_normc_,
              lambda x: nn.init.constant_(x, 0),
                               gain=np.sqrt(2))

        self.start = nn.Sequential(
            init_(nn.Linear(list_inputs[0], 256)),
            nn.ReLU())

        self.proposal = nn.Sequential(
            init_(nn.Linear(list_inputs[1], 512)),
            nn.ReLU())

        self.end = nn.Sequential(
            init_(nn.Linear(list_inputs[2], 256)),
            nn.ReLU())

        self.actor = nn.Sequential(
            init_(nn.Linear(1024 , 256)),
            nn.ReLU()
        )
        self.critic = nn.Sequential(
            init_(nn.Linear(1024, 256)),
            nn.ReLU(),
            init_(nn.Linear(256, 1))
        )
        self.train()

    def forward(self, inputs, states, masks):
        start = inputs[:, :self.list_inputs[0]]
        proposal = inputs[:, self.list_inputs[0]:(self.list_inputs[0] + self.list_inputs[1])]
        end = inputs[:, -self.list_inputs[2]:]
        inputs = torch.cat([self.start(start), self.proposal(proposal), self.end(end)], 1)
        hidden_actor = self.actor(inputs)
        value = self.critic(inputs)
        return value, hidden_actor, states


class Classifier(nn.Module):
    def __init__(self, feature_length, class_num, isbn=True):
        super(Classifier, self).__init__()
        if isbn:
            self.frontlayer = nn.Sequential(
                nn.Linear(feature_length, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(),
                nn.Linear(1024, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Linear(512, class_num)
            )
        else:
            self.frontlayer = nn.Sequential(
                nn.Linear(feature_length, 512),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(512, class_num)
            )

    def forward(self, x):
        logit = self.frontlayer(x)
        return logit

def buildClassifier(checkpoint=None, feature_length=800, class_num=101, cuda=True):
    clf = Classifier(feature_length, class_num)
    if checkpoint:
        checkpoint = torch.load(checkpoint)
        base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint['state_dict'].items())}
        clf.load_state_dict(base_dict)
    if cuda:
        clf.cuda()
    clf = torch.nn.DataParallel(clf, device_ids=[i for i in range(torch.cuda.device_count())])
    clf.eval()
    return clf

def load_clf_from_i3d(clf, i3d_ckpt):
    return _load_module(clf, i3d_ckpt, '1')

def load_fe_from_i3d(fe, i3d_ckpt):
    return _load_module(fe, i3d_ckpt, '0')

def load_i3d_checkpoint(i3d_clf, i3d_ckpt):
    checkpoint = torch.load(i3d_ckpt)
    new_ckpt = OrderedDict()
    for key in checkpoint['state_dict'].keys():
        new_ckpt[key[7:]] = checkpoint['state_dict'][key]
    # args.start_epoch = checkpoint['epoch']
    i3d_clf.load_state_dict(new_ckpt)
    return i3d_clf

def _load_module(module, i3d_ckpt, name):
    assert isinstance(name, str)
    if os.path.isfile(i3d_ckpt):
        print(("=> loading CLASSIFIER checkpoint '{}'".format(i3d_ckpt)))
        checkpoint = torch.load(i3d_ckpt)
        # args.start_epoch = checkpoint['epoch']

        new_state_dict = OrderedDict()
        keys = checkpoint['state_dict'].keys()
        for key in keys:
            if key.split('.')[1] == name:
                new_key = key[9:]   # key look like: module.1.frontlayer.0.weight
                new_state_dict[new_key] = checkpoint['state_dict'][key]
        module.load_state_dict(new_state_dict)
    else:
        print(("=> no checkpoint found at '{}'".format(i3d_ckpt)))
        return None
    return module