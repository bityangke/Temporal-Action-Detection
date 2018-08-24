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
    accu = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()
    for i, (r, f, path, label, video_index, seg_index, verify) in enumerate(train_loader):
        data_time.update(time.time() - end)
        r = r.cuda()
        f = f.cuda()
        output = model([r, f])

        target_var = label.cuda(async=True).type(torch.cuda.LongTensor)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, = accuracy(output, target_var, topk=(1,))
        losses.update(loss.item(), r.shape[0])
        accu.update(prec1.item(), r.shape[0])

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
                   'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=accu, lr=optimizer.param_groups[-1]['lr'])))


def validate(val_loader, model, criterion, iter=None, logger=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    accu = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (r, f, path, label, video_index, seg_index, verify) in enumerate(val_loader):
        with torch.no_grad():
            r = r.cuda()
            f = f.cuda()
            target_var = label.cuda(async=True).type(torch.cuda.LongTensor)
            output = model([r, f])
            loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, = accuracy(output, target_var, topk=(1,))

        losses.update(loss.item(), r.shape[0])
        accu.update(prec1.item(), r.shape[0])
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(('Test: [{0}/{1}]\t'
                   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                   'Loss {loss.val:.6f} ({loss.avg:.6f})\t'
                   'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=accu)))

    print(('Testing Results: Prec@1 {top1.avg:.3f} Loss {loss.avg:.5f}'
           .format(top1=accu, loss=losses)))

    return accu.avg


def main(args):
    num_class = 102
    feature_length = 800
    clf = Classifier(feature_length, num_class, isbn=True)
    fe = I3DFeatureExtractor()
    I3DClassifier = torch.nn.Sequential(
        fe,
        clf
    )

    if args.restore:
        if os.path.isfile(args.restore):
            from models.module import load_i3d_checkpoint
            # from collections import OrderedDict
            # print(("=> loading checkpoint '{}'".format(args.restore)))
            # checkpoint = torch.load(args.restore)
            # new_ckpt = OrderedDict()
            # for key in checkpoint['state_dict'].keys():
            #     new_ckpt[key[7:]] = checkpoint['state_dict'][key]
            # args.start_epoch = checkpoint['epoch']
            # I3DClassifier.load_state_dict(new_ckpt)
            # print(("=> loaded checkpoint '{}' (epoch {})"
            #        .format(args.evaluate, checkpoint['epoch'])))
        else:
            print(("=> no checkpoint found at '{}'".format(args.restore)))

    I3DClassifier = torch.nn.DataParallel(I3DClassifier, device_ids=[i for i in list(range(torch.cuda.device_count()))]).cuda()
    # I3DClassifier = I3DClassifier.cuda()

    ds = build_video_dataset("background", train=True, test_rate=0.05)['dataset']
    eval_ds = build_video_dataset("background", train=False, test_rate=0.05)['dataset']

    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True)

    eval_loader = torch.utils.data.DataLoader(
        eval_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True)

    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(I3DClassifier.parameters(), lr=args.lr)

    best_prec1 = 0

    if args.evaluate:
        prec1 = validate(eval_loader, I3DClassifier, criterion)
        return prec1

    print('Fuck you')

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train(loader, I3DClassifier, criterion, optimizer, epoch)

        # evaluate on validation set
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            prec1 = validate(eval_loader, I3DClassifier, criterion, (epoch + 1) * len(loader))

            # remember best prec@1 and save checkpoint
            is_best = (prec1 > best_prec1)

            best_prec1 = max(prec1, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.model,
                'tsn_like': args.tsn,
                'state_dict': I3DClassifier.state_dict(),
                'best_prec1': best_prec1,
            }, is_best, args.exp_name)

    print('Experiment {} finished! Best Accu@1 is {:.6f}'.format(args.exp_name, best_prec1))


if __name__ == '__main__':
    now = time.strftime('%m%d_%H%M', time.localtime(time.time()))
    parser = argparse.ArgumentParser(description="Find a good video representation")
    parser.add_argument('--exp-name', type=str, default=None)
    parser.add_argument('--fuse-type', type=str, help='In average, cnn, max, concat')
    parser.add_argument('--tsn', action='store_true', default=False)
    parser.add_argument('--model', type=str, help='Should be in [cnn, mlp, lstm]!')
    parser.add_argument('--num-workers', type=int, default=40)
    parser.add_argument('--lr', type=float, default=4e-5)
    parser.add_argument('--evaluate', action='store_true', default=False)
    parser.add_argument('--start-epoch', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=-1)
    parser.add_argument('--dataset', type=str, default='hmdb51')
    parser.add_argument('--restore', type=str, help='The path toward checkpoint file.', default='')
    parser.add_argument('--eval-freq', type=int, default=1)
    parser.add_argument('--clip-gradient', '--gd', default=None, type=float,
                        metavar='W', help='gradient norm clipping (default: disabled)')
    parser.add_argument('--print-freq', type=int, default=1)
    args = parser.parse_args()

    if (args.restore == '') and (args.start_epoch != 0):
        raise ValueError('When you are not restore from checkpoint, you should not set start_epoch different from 0!')
    if args.epochs == -1:
        raise ValueError("You should explicit declare args.epoch!")

    if args.exp_name == None:
        args.exp_name = "{}_{}".format(now, "102act")

    print('Experiment {} start!'.format(args.exp_name))
    main(args)
