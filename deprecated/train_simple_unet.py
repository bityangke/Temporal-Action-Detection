import argparse
import os
import time

import torch

from dataset import build_video_dataset
from models.feature_extractor import I3DFeatureExtractor
from models.module import Classifier
from ops.utils import AverageMeter, accuracy, save_checkpoint


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()
    for i, (r, f, path, label, video_index, seg_index, verify) in enumerate(train_loader):
        r = r.cuda()
        f = f.cuda()
        output, _ = model(r, f)

        target_var = label.cuda(async=True).type(torch.cuda.LongTensor)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target_var, topk=(1, 5))
        losses.update(loss.item(), r.shape[0])
        top1.update(prec1.item(), r.shape[0])
        top5.update(prec5.item(), r.shape[0])

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        if args.clip_gradient is not None:
            total_norm = torch.nn.utils.clip_grad_norm(model.parameters(), args.clip_gradient)

        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                   'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                   'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                   'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5, lr=optimizer.lr)))


def validate(val_loader, model, criterion, iter, logger=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (r, f, path, label, video_index, seg_index, verify) in enumerate(val_loader):
        if i==400: break
        with torch.no_grad():
            r = r.cuda()
            f = f.cuda()
            target_var = label.cuda(async=True).type(torch.cuda.LongTensor)
            output, _ = model(r, f)
            loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target_var, topk=(1, 5))

        losses.update(loss.item(), r.shape[0])
        top1.update(prec1.item(), r.shape[0])
        top5.update(prec5.item(), r.shape[0])

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(('Test: [{0}/{1}]\t'
                   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                   'Loss {loss.val:.6f} ({loss.avg:.6f})\t'
                   'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                   'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1, top5=top5)))
        # if i == 99: break  # Randomly Test 100 batch data (which is in the training data, so this is not a really evaluation).

    print(('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
           .format(top1=top1, top5=top5, loss=losses)))

    return top1.avg, top5.avg

class UnetLoss(torch.nn.Module):
    def __init__(self):
        super(UnetLoss, self).__init__()

    def forward(self, logit, target):
        logit = torch.nn.functional.log_softmax(logit, dim=1)
        return -torch.mean(torch.sum(logit * target.type(torch.cuda.FloatTensor), dim=1) / torch.sum(target, dim=1).type(torch.cuda.FloatTensor))



def main(args):
    num_class = 101
    feature_length = 800
    from models.unet import UntrimmedNet
    I3DClassifier = UntrimmedNet(num_class)
    I3DClassifier = torch.nn.DataParallel(I3DClassifier, device_ids=[0, 1, 2, 3]).cuda()

    if args.restore:
        if os.path.isfile(args.restore):
            print(("=> loading checkpoint '{}'".format(args.restore)))
            checkpoint = torch.load(args.restore)
            # args.start_epoch = checkpoint['epoch']
            I3DClassifier.load_state_dict(checkpoint['state_dict'])
            print(("=> loaded checkpoint '{}' (epoch {})"
                   .format(args.evaluate, checkpoint['epoch'])))
        else:
            print(("=> no checkpoint found at '{}'".format(args.restore)))

    ds = build_video_dataset("thumos_validation", train=True)['dataset']
    eval_ds = build_video_dataset("thumos_test", train=False)['dataset']

    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_worker, pin_memory=True)

    eval_loader = torch.utils.data.DataLoader(
        eval_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_worker, pin_memory=True)

    ckpt_dir = 'result/' + args.exp_name
    if not os.path.exists(ckpt_dir):
        os.mkdir(ckpt_dir)

    criterion = torch.nn.CrossEntropyLoss().cuda()
    multilabel_criterion = UnetLoss().cuda()
    optimizer = torch.optim.Adam(I3DClassifier.parameters(), lr=args.lr)

    best_prec1 = 0
    best_prec5 = 0

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train(loader, I3DClassifier, multilabel_criterion, optimizer, epoch)

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.model,
            'state_dict': I3DClassifier.state_dict(),
        }, False, args.exp_name + '_epoch{}'.format(epoch), path=ckpt_dir)

        # evaluate on validation set
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:

            prec1, prec5 = validate(eval_loader, I3DClassifier, criterion, (epoch + 1) * len(loader))

            # remember best prec@1 and save checkpoint
            is_best = (prec1 > best_prec1) or (prec5 > best_prec5)

            best_prec1 = max(prec1, best_prec1)
            best_prec5 = max(prec5, best_prec5)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.model,
                'state_dict': I3DClassifier.state_dict(),
                'best_prec1': best_prec1,
                'best_prec5': best_prec5
            }, is_best, args.exp_name + '_epoch{}'.format(epoch), path=ckpt_dir)

    print('Experiment {} finished! Best Accu@1 is {:.6f}, Best Accu@5 is {:.6f}.'.format(args.exp_name, best_prec1,
                                                                                         best_prec5))


if __name__ == '__main__':
    now = time.strftime('%m%d_%H%M', time.localtime(time.time()))
    parser = argparse.ArgumentParser(description="Find a good video representation")
    parser.add_argument('--exp-name', type=str, default=None)
    parser.add_argument('--fuse-type', type=str, help='In average, cnn, max, concat')
    parser.add_argument('--tsn', action='store_true', default=False)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-worker', type=int, default=8)
    parser.add_argument('--model', type=str, help='Should be in [cnn, mlp, lstm]!')
    parser.add_argument('--lr', type=float, default=4e-5)
    parser.add_argument('--evaluate', action='store_true', default=False)
    parser.add_argument('--start-epoch', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=-1)
    parser.add_argument('--dataset', type=str, default='hmdb51')
    parser.add_argument('--restore', type=str, help='The path toward checkpoint file.', default='')
    parser.add_argument('--eval-freq', type=int, default=10)
    parser.add_argument('--clip-gradient', '--gd', default=None, type=float,
                        metavar='W', help='gradient norm clipping (default: disabled)')
    parser.add_argument('--print-freq', type=int, default=1)
    args = parser.parse_args()

    if (args.restore == '') and (args.start_epoch != 0):
        raise ValueError('When you are not restore from checkpoint, you should not set start_epoch different from 0!')
    if args.epochs == -1:
        raise ValueError("You should explicit declare args.epoch!")

    if args.exp_name == None:
        args.exp_name = "{}_{}".format(now, "unet")

    print('Experiment {} start!'.format(args.exp_name))
    main(args)
