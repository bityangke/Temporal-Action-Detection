import torch
import torch.nn as nn
import numpy as np
import yaml
import shutil


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# Necessary for my KFAC implementation.
class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias


# def unet_accuracy(output, target, threshold=0.5):
#     """Computes the precision@k for the specified values of k"""
#     # 在原始代码中，假设target是一个数字。
#     # 现在我要改变这个函数，使得他们同样可以适配 N-hot Vector 的target。
#     if isinstance(topk, int):
#         topk = (topk)
#     maxk = max(topk)
#     batch_size = target.size(0)
#
#     _, pred = output.topk(maxk, 1, True, True)
#     pred = pred.t()
#     correct = pred.eq(target.view(1, -1).expand_as(pred))
#
#     res = []
#     for k in topk:
#         correct_k = correct[:k].view(-1).float().sum(0)
#         res.append(correct_k.mul_(100.0 / batch_size))
#     return res


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if isinstance(topk, int):
        topk = (topk)
    maxk = max(topk)
    batch_size = target.size(0)

    if len(target.shape)==2:
        _, target = target.max(dim=1)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def save_checkpoint(state, is_best, filename, path):
    file = path + '/' + filename + '_model.pth.tar'
    torch.save(state, file)
    if is_best:
        best_name = path + '/' + filename + '_best.pth.tar'
        shutil.copyfile(file, best_name)



def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


# https://github.com/openai/baselines/blob/master/baselines/common/tf_util.py#L87
def init_normc_(weight, gain=1):
    weight.normal_(0, 1)
    weight *= gain / torch.sqrt(weight.pow(2).sum(1, keepdim=True))


def get_configs(dataset):
    data = yaml.load(open('data/dataset_cfg.yaml'))
    return data[dataset]

def get_actionness_configs(dataset):
    data = yaml.load(open('data/dataset_actionness_cfg.yaml'))
    return data[dataset]


def get_reference_model_url(dataset, modality, init, arch):
    data = yaml.load(open('data/reference_models.yaml'))
    return data[dataset][init][arch][modality]


def get_grad_hook(name):
    def hook(m, grad_in, grad_out):
        print(len(grad_in), len(grad_out))
        print((name, grad_out[0].data.abs().mean(), grad_in[0].data.abs().mean()))
        print((grad_out[0].size()))
        print((grad_in[0].size()))
        print((grad_in[1].size()))
        print((grad_in[2].size()))

        # print((grad_out[0]))
        # print((grad_in[0]))

    return hook

def get_iou(proposal_A, proposal_B):
    """
    Calculates the intersection over union of two temporal "bounding boxes"
    inputs: Instance object
    output: Scalar value
    """
    sa, ea, sb, eb = proposal_A.start_frame, proposal_A.end_frame, proposal_B.start_frame, proposal_B.end_frame
    union = (max(sa, sb), min(ea, eb))
    inter = (min(sa, sb), max(ea, eb))

    if inter[0] >= inter[1]:
        return 0
    else:
        return float(inter[1] - inter[0]) / float(union[1] - union[0])

def softmax(scores):
    es = np.exp(scores - scores.max(axis=-1)[..., None])
    return es / es.sum(axis=-1)[..., None]


def temporal_iou(span_A, span_B):
    """
    Calculates the intersection over union of two temporal "bounding boxes"

    span_A: (start, end)
    span_B: (start, end)
    """
    union = min(span_A[0], span_B[0]), max(span_A[1], span_B[1])
    inter = max(span_A[0], span_B[0]), min(span_A[1], span_B[1])

    if inter[0] >= inter[1]:
        return 0
    else:
        return float(inter[1] - inter[0]) / float(union[1] - union[0])


def temporal_nms(bboxes, thresh):
    """
    One-dimensional non-maximal suppression
    :param bboxes: [[st, ed, score, ...], ...]
    :param thresh:
    :return:
    """
    t1 = bboxes[:, 0]
    t2 = bboxes[:, 1]
    scores = bboxes[:, 2]

    durations = t2 - t1
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        tt1 = np.maximum(t1[i], t1[order[1:]])
        tt2 = np.minimum(t2[i], t2[order[1:]])
        intersection = tt2 - tt1
        IoU = intersection / (durations[i] + durations[order[1:]] - intersection).astype(float)

        inds = np.where(IoU <= thresh)[0]
        order = order[inds + 1]

    return bboxes[keep, :]
