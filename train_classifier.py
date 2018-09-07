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
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--modality', type=str)
parser.add_argument('--front-end', action='store_true', default=False)
parser.add_argument('--evaluate', action='store_true', default=False)
parser.add_argument('--clip-num', type=int, default=8)
parser.add_argument('--dataset', type=str, default='thumos_validation')
parser.add_argument('--start-epoch', type=int, default=0)
parser.add_argument('--epochs', type=int, default=-1)
parser.add_argument('--restore', type=str, help='The path toward checkpoint file.', default='')
parser.add_argument('--eval-freq', type=int, default=500)
parser.add_argument('--clip-gradient', '--gd', default=None, type=float,
                    metavar='W', help='gradient norm clipping (default: disabled)')
parser.add_argument('--print-freq', type=int, default=4)
parser.add_argument('--fuse-type', type=str, default='unet')
parser.add_argument('--unet-softmax', type=str, default='in')
args = parser.parse_args()

softmax_dim0 = nn.Softmax(0)
softmax_dim1 = nn.Softmax(1)
softmax_dim2 = nn.Softmax(2)

batch_time = AverageMeter()
data_time = AverageMeter()
losses = AverageMeter()
top1 = AverageMeter()
top5 = AverageMeter()


def train(train_loader, model, criterion, optimizer, epoch, modality, front=None):
    # switch to train mode
    model.train()
    end = time.time()
    for i, data in enumerate(train_loader):
        data_time.update(time.time()-end)
        if args.front_end:
            with torch.no_grad():
                r, f, _, target, _ = data
                r = r.cuda()
                f = f.cuda()
                shape = r.shape[:2]
                size = r.shape[0] * r.shape[1]
                r = r.reshape(size, *(r.shape[2:]))
                f = f.reshape(size, *(f.shape[2:]))
                input_var = front([r,f]).detach()
                target_var = target.cuda(async=True).type(torch.cuda.LongTensor)
        else:
            input, target, _, _, _ = data
            data_time.update(time.time() - end)
            target_var = target.cuda(async=True).type(torch.cuda.LongTensor)
            input_var = input.cuda(async=True)
            shape = input_var.shape
            input_var = input_var.reshape(shape[0] * shape[1], shape[2])
        # print("FUCKYOU!!!!", front.module)
        if modality == 'rgb':
            logit, weight = model(input_var[:, :400])
        elif modality == 'flow':
            logit, weight = model(input_var[:, 400:])
        else:
            raise ValueError
        logit = logit.reshape(shape[0], shape[1], logit.shape[-1])
        weight = weight.reshape(shape[0], shape[1], 1)
        # attention = softmax_dim1(weight)
        if args.unet_softmax == 'in':
            output = torch.sum(softmax_dim1(weight).repeat(1, 1, 101) * softmax_dim2(logit), dim=1)
        elif args.unet_softmax == 'out':
            output = torch.sum(softmax_dim1(weight).repeat(1, 1, 101) * logit, dim=1)
        else:
            raise ValueError
        loss = criterion(output, target_var)


        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target_var, topk=(1, 5))
        losses.update(loss.item(), shape[0])
        top1.update(prec1.item(), shape[0])
        top5.update(prec5.item(), shape[0])

        # compute gradient and do SGD step
        optimizer.zero_grad()

        loss.backward()

        if args.clip_gradient is not None:
            total_norm = torch.nn.utils.clip_grad_norm(model.parameters(), args.clip_gradient)
            # if total_norm > args.clip_gradient:
            # print("clipping gradient: {} with coef {}".format(total_norm, args.clip_gradient / total_norm))

        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % args.print_freq == 0:
            print(('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                   'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                   'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                   'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i + 1, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5, lr=optimizer.param_groups[-1]['lr'])))


def validate(val_loader, model, criterion, modality):
    eval_batch_time = AverageMeter()
    eval_losses = AverageMeter()
    eval_top1 = AverageMeter()
    eval_top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target, _, _, _) in enumerate(val_loader):
        with torch.no_grad():
            input_var = torch.from_numpy(input).cuda()
            target_var = torch.Tensor([target]).type(torch.cuda.LongTensor)

            if modality == 'rgb':
                logit, weight = model(input_var[:, :400])
            elif modality == 'flow':
                logit, weight = model(input_var[:, 400:])
            else:
                raise ValueError

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
        eval_batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(('Test: [{0}/{1}]\t'
                   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                   'Loss {loss.val:.6f} ({loss.avg:.6f})\t'
                   'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                   'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                i, len(val_loader), batch_time=eval_batch_time, loss=eval_losses,
                top1=eval_top1, top5=eval_top5)))
    print(('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
           .format(top1=eval_top1, top5=eval_top5, loss=eval_losses)))
    return eval_top1.avg, eval_top5.avg


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


def main(args):
    fuser = Fuser(fuse_type=args.fuse_type, s=16)
    if args.front_end:
        train_dataset = build_video_dataset("ucf", train=True, unet=True, unet_clip_num=args.clip_num)['dataset']
    else:
        if args.dataset=='ucf':
            train_dataset = FeatureDataset('features/ucf101/data.csv', fuser)
        elif args.dataset=='thumos':
            train_dataset = FeatureDataset('features/kinetics/thumos_validation/data.csv', fuser)
        else:
            raise ValueError
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.num_worker,
                                               pin_memory=True)

    eval_dataset = FeatureDataset('features/kinetics/thumos_test/data.csv', Fuser(fuse_type='none'))
    num_class = 101



    front = None
    if args.front_end:
        front = I3DFeatureExtractor()
        front = torch.nn.DataParallel(front, device_ids=[i for i in range(torch.cuda.device_count())]).cuda()
        front.eval()


    model = UntrimmedNetBack(num_class)
    if args.restore:
        if os.path.isfile(args.restore):
            print(("=> loading checkpoint '{}'".format(args.restore)))
            checkpoint = torch.load(args.restore)
            new = OrderedDict()
            for key in checkpoint['state_dict'].keys():
                if key[7:] in model.state_dict():
                    new[key[7:]] = checkpoint['state_dict'][key]
            model.load_state_dict(new)
            print(("=> loaded checkpoint '{}' (epoch {})"
                   .format(args.evaluate, checkpoint['epoch'])))
        else:
            print(("=> no checkpoint found at '{}'".format(args.restore)))
    model = torch.nn.DataParallel(model, device_ids=[i for i in range(torch.cuda.device_count())]).cuda()

    criterion = UnetLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0005)

    best_prec1 = 0
    best_prec5 = 0
    lastavg = 0

    if args.evaluate:
        best_prec1, best_prec5 = validate(eval_dataset, model, criterion, args.modality)
        print('Experiment {} finished! Best Accu@1 is {:.6f}, Best Accu@5 is {:.6f}.'.format(args.exp_name, best_prec1,
                                                                                             best_prec5))
        return

    for epoch in range(args.start_epoch, args.epochs):
        train(train_loader, model, criterion, optimizer, epoch, args.modality, front)

        if args.dataset=='thumos':
            if (epoch + 1) == 100:
                top1.reset()
                top5.reset()
                lastavg = losses.avg
                losses.reset()
                batch_time.reset()
                data_time.reset()

            if (epoch + 1) == 1000:
                optimizer.param_groups[-1]['lr'] = optimizer.param_groups[-1]['lr'] / 10

            if (epoch + 1) == 2000:
                optimizer.param_groups[-1]['lr'] = optimizer.param_groups[-1]['lr'] / 2

            if (epoch + 1) == 3000:
                optimizer.param_groups[-1]['lr'] = optimizer.param_groups[-1]['lr'] / 2

            if (epoch + 1) == 4000:
                optimizer.param_groups[-1]['lr'] = optimizer.param_groups[-1]['lr'] / 2
        elif args.dataset=='ucf':
            if (epoch + 1) == 10:
                top1.reset()
                top5.reset()
                lastavg = losses.avg
                losses.reset()
                batch_time.reset()
                data_time.reset()
            if (epoch + 1) == 400:
                optimizer.param_groups[-1]['lr'] = optimizer.param_groups[-1]['lr'] / 10
            if (epoch + 1) == 800:
                optimizer.param_groups[-1]['lr'] = optimizer.param_groups[-1]['lr'] / 2
        else:
            raise ValueError


        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            top1.reset()
            top5.reset()
            lastavg=losses.avg
            losses.reset()
            batch_time.reset()
            data_time.reset()

            prefix = 'result/{}'.format(args.exp_name)
            if not os.path.exists(prefix):
                os.mkdir(prefix)
            prec1, prec5 = validate(eval_dataset, model, criterion, args.modality)

            is_best = (prec1 > best_prec1) or (prec5 > best_prec5)

            best_prec1 = max(prec1, best_prec1)
            best_prec5 = max(prec5, best_prec5)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'best_prec5': best_prec5
            }, is_best, args.exp_name + "_epoch{}".format(epoch), prefix)
    print(args)
    print('Experiment {} finished! Best Accu@1 is {:.6f}, Best Accu@5 is {:.6f}. Saved@ {}'.format(args.exp_name,
                                                                                                   best_prec1,
                                                                                                   best_prec5,
                                                                                                   'result/{}/{}_epoch{}'.format(
                                                                                                       args.exp_name,
                                                                                                       args.exp_name,
                                                                                                       epoch)))

if __name__ == '__main__':
    assert args.modality in ['rgb', 'flow']

    if args.exp_name == None:
        args.exp_name = "{}_{}_{}".format(now, 'unet', args.modality)

    print('Experiment {} start!'.format(args.exp_name))
    main(args)