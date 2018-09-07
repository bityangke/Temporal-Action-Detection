import argparse
import os
import time
from collections import OrderedDict

# import cudnn
from dataset import FeatureDataset, build_video_dataset
from ops.fuser import Fuser
from ops.utils import *
from models.feature_extractor import I3DFeatureExtractor
from models.unet import UntrimmedNetBack

now = time.strftime('%m%d_%H%M', time.localtime(time.time()))
parser = argparse.ArgumentParser(description="Find a good video representation")
parser.add_argument('--exp-name', type=str, default=None)
parser.add_argument('--batch-size', type=int, default=256)
parser.add_argument('--num-worker', type=int, default=8)
parser.add_argument('--print-freq', type=int, default=400)
parser.add_argument('--restore-rgb', type=str, help='The path toward checkpoint file.', default='')
parser.add_argument('--restore-flow', type=str, help='The path toward checkpoint file.', default='')
parser.add_argument('--unet-softmax', type=str, default='in')
parser.add_argument('--alpha', type=float)
args = parser.parse_args()

softmax_dim0 = nn.Softmax(0)
softmax_dim1 = nn.Softmax(1)
softmax_dim2 = nn.Softmax(2)

batch_time = AverageMeter()
data_time = AverageMeter()
losses = AverageMeter()
top1 = AverageMeter()
top5 = AverageMeter()


def validate(val_loader, model, criterion, modality, alpha=None, model2=None):
    eval_batch_time = AverageMeter()
    eval_losses = AverageMeter()
    eval_top1 = AverageMeter()
    eval_top5 = AverageMeter()
    eval_losses2 = AverageMeter()
    eval_top12 = AverageMeter()
    eval_top52 = AverageMeter()
    eval_fuse_losses = AverageMeter()
    eval_fuse_top1 = AverageMeter()
    eval_fuse_top5 = AverageMeter()

    model.eval()

    end = time.time()
    for i, (input, target, _, _, _) in enumerate(val_loader):
        with torch.no_grad():
            input_var = torch.from_numpy(input).cuda()
            target_var = torch.Tensor([target]).type(torch.cuda.LongTensor)
            assert isinstance(alpha, float) and (alpha<1) and (alpha>0), alpha
            logit, weight = model(input_var[:, :400])
            logit2, weight2 = model2(input_var[:, 400:])
            if args.unet_softmax == 'in':
                output2 = torch.sum(softmax_dim0(weight2).repeat(1, 101) * softmax_dim1(logit2), dim=0, keepdim=True)
            elif args.unet_softmax == 'out':
                output2 = torch.sum(softmax_dim0(weight2).repeat(1, 101) * logit2, dim=0, keepdim=True)
            loss2 = criterion(output2, target_var)
            prec12, prec52 = accuracy(output2, target_var, topk=(1, 5))
            eval_losses2.update(loss2)
            eval_top12.update(prec12)
            eval_top52.update(prec52)
            if args.unet_softmax == 'in':
                output = torch.sum(softmax_dim0(weight).repeat(1, 101) * softmax_dim1(logit), dim=0, keepdim=True)
            elif args.unet_softmax == 'out':
                output = torch.sum(softmax_dim0(weight).repeat(1, 101) * logit, dim=0, keepdim=True)
            else:
                raise ValueError

        loss = criterion(output, target_var)
        prec1, prec5 = accuracy(output, target_var, topk=(1, 5))
        eval_losses.update(loss.item(), 1)
        eval_top1.update(prec1.item(), 1)
        eval_top5.update(prec5.item(), 1)

        fuse_output = output * (1-alpha) + output2 * alpha
        fuse_loss = criterion(fuse_output, target_var)
        fuse_prec1, fuse_prec5 = accuracy(fuse_output, target_var, topk=(1, 5))
        eval_fuse_losses.update(fuse_loss)
        eval_fuse_top1.update(fuse_prec1)
        eval_fuse_top5.update(fuse_prec5)

        eval_batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(('Test: [{0}/{1}]\t'
                   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                   'RGBLoss {loss.val:.6f} ({loss.avg:.6f})\t'
                   'RGBPrec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                   'RGBPrec@5 {top5.val:.3f} ({top5.avg:.3f})'
                   'FLOWLoss {loss2.val:.6f} ({loss2.avg:.6f})\t'
                   'FLOWPrec@1 {top12.val:.3f} ({top12.avg:.3f})\t'
                   'FLOWPrec@5 {top52.val:.3f} ({top52.avg:.3f})'
                   'ALLLoss {lossf.val:.6f} ({lossf.avg:.6f})\t'
                   'ALLPrec@1 {top1f.val:.3f} ({top1f.avg:.3f})\t'
                   'ALLPrec@5 {top5f.val:.3f} ({top5f.avg:.3f})'.format(
                i, len(val_loader), batch_time=eval_batch_time, loss=eval_losses,
                top1=eval_top1, top5=eval_top5, loss2=eval_losses2, top12=eval_top12, top52=eval_top52,
            top1f=eval_fuse_top1, top5f=eval_fuse_top5, lossf=eval_fuse_losses)))

    print(('Testing Results: \nRGBPrec@1 {top1.avg:.3f} RGBPrec@5 {top5.avg:.3f} RGBLoss {loss.avg:.5f} \nFLOWPrec@1 {top12.avg:.3f} FLOWPrec@5 {top52.avg:.3f} FLOWLoss {loss2.avg:.5f} \nALLPrec@1 {top1f.avg:.3f} ALLPrec@5 {top5f.avg:.3f} ALLLoss {lossf.avg:.5f}'
       .format(top1=eval_top1, top5=eval_top5, loss=eval_losses, loss2=eval_losses2, top12=eval_top12, top52=eval_top52,
            top1f=eval_fuse_top1, top5f=eval_fuse_top5, lossf=eval_fuse_losses)))
    print("We use alpha:", alpha)
    print(('{alpha}\t{top1.avg:.3f}\t{top5.avg:.3f}\t{loss.avg:.5f}\t{top12.avg:.3f}\t{top52.avg:.3f}\t{loss2.avg:.5f}\t{top1f.avg:.3f}\t{top5f.avg:.3f}\t{lossf.avg:.5f}'
       .format(alpha=alpha, top1=eval_top1, top5=eval_top5, loss=eval_losses, loss2=eval_losses2, top12=eval_top12, top52=eval_top52,
            top1f=eval_fuse_top1, top5f=eval_fuse_top5, lossf=eval_fuse_losses)))
    return eval_fuse_top1.avg, eval_fuse_top5.avg

class UnetLoss(torch.nn.Module):
    def __init__(self):
        super(UnetLoss, self).__init__()

    def forward(self, logit, target):
        if args.unet_softmax == 'in':
            logit = torch.clamp(logit, 1e-10, 1)
            logit = torch.log(logit)
        elif args.unet_softmax == 'out':
            logit = torch.nn.functional.log_softmax(logit, dim=1)
        else:
            raise ValueError
        if len(target.shape) == 1:
            onehotlabel = torch.zeros(logit.shape)
            for i, t in enumerate(target):
                onehotlabel[i, t] = 1
            target = onehotlabel
        return -torch.mean(
            torch.sum(logit * target.type(torch.cuda.FloatTensor), dim=1) / torch.sum(target, dim=1).type(
                torch.cuda.FloatTensor))

def build_model(model, ckpt):
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
    return model

if __name__ == '__main__':
    if args.exp_name == None:
        args.exp_name = "{}_{}_{}".format(now, 'unet', args.modality)
    print('Experiment {} start!'.format(args.exp_name))
    eval_dataset = FeatureDataset('features/kinetics/thumos_test/data.csv', Fuser(fuse_type='none'))
    num_class = 101

    model_rgb = UntrimmedNetBack(num_class)
    model_rgb = build_model(model_rgb, args.restore_rgb)

    model_flow = UntrimmedNetBack(num_class)
    model_flow = build_model(model_flow, args.restore_flow)

    criterion = UnetLoss().cuda()

    best_prec1, best_prec5 = validate(eval_dataset, model_rgb, criterion, args.modality, model2=model_flow, alpha=args.alpha)
    print('Experiment {} finished! Best Accu@1 is {:.6f}, Best Accu@5 is {:.6f}.'.format(args.exp_name, best_prec1,
                                                                                         best_prec5))