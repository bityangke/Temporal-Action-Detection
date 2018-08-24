import argparse
import os
import time

import torch

from dataset import build_video_dataset
from models.feature_extractor import I3DFeatureExtractor
from models.module import Classifier
from ops.utils import AverageMeter, accuracy, save_checkpoint

def uniform_sample(rgb, flow, n):
    total_length = rgb


weight_softmax = torch.nn.Softmax(1)    # Assert weight = [batchsize, numclips]
clf_softmax = torch.nn.Softmax(2)



def train(train_loader, model, criterion, optimizer, num_class, unet_clip_num, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()
    for i, (r, f, path, label, segnum) in enumerate(train_loader):
        assert r.shape[0]%4==0
        subiter = r.shape[0]//4
        # if i==1: break # For test!!!!
        r = r.cuda()
        f = f.cuda()
        logit = []
        weight = []
        for subi in range(subiter):
            tmplogit, tmpweight = model(r[subi*4:(subi+1)*4], f[subi*4:(subi+1)*4])
            logit.append(tmplogit)
            weight.append(tmpweight)
        logit = torch.cat(logit)
        weight = torch.cat(weight)
        logit = logit.reshape(logit.shape[0]//unet_clip_num, unet_clip_num, *logit.shape[1:])
        weight = weight.reshape(weight.shape[0]//unet_clip_num, unet_clip_num, *weight.shape[1:])
        weight = weight_softmax(weight)

        output = torch.sum(weight.repeat(1,1,num_class) * clf_softmax(logit), dim=1)
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
                data_time=data_time, loss=losses, top1=top1, top5=top5, lr=optimizer.param_groups[-1]['lr'])))


def validate(val_loader, model, criterion, num_class, unet_clip_num):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (r, f, path, label,segnum) in enumerate(val_loader):
        with torch.no_grad():
            r = r.cuda()
            f = f.cuda()
            target_var = label.cuda(async=True).type(torch.cuda.LongTensor)
            logit, weight = model(r, f)
            logit = logit.reshape(logit.shape[0]//unet_clip_num, unet_clip_num, *logit.shape[1:])
            weight = weight.reshape(weight.shape[0]//unet_clip_num, unet_clip_num, *weight.shape[1:])
            output = torch.sum(weight_softmax(weight).repeat(1,1,num_class) * clf_softmax(logit), dim=1)
            loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target_var, topk=(1, 5))
        # print(label)
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
        if not path:
            if i == 99: break  # Randomly Test 100 batch data (which is in the training data, so this is not a really evaluation).

    print(('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
           .format(top1=top1, top5=top5, loss=losses)))

    return top1.avg, top5.avg



def UnetValidate(val_loader, model, criterion, num_class, unet_clip_num):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (r, f, path, label,segnum) in enumerate(val_loader):
        with torch.no_grad():
            r = r.cuda()
            f = f.cuda()
            target_var = label.cuda(async=True).type(torch.cuda.LongTensor)
            logit, weight = model(r, f)
            logit = logit.reshape(logit.shape[0]//unet_clip_num, unet_clip_num, *logit.shape[1:])
            weight = weight.reshape(weight.shape[0]//unet_clip_num, unet_clip_num, *weight.shape[1:])
            output = torch.sum(weight_softmax(weight).repeat(1,1,num_class) * clf_softmax(logit), dim=1)
            loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target_var, topk=(1, 5))
        # print(label)
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
        if not path:
            if i == 99: break  # Randomly Test 100 batch data (which is in the training data, so this is not a really evaluation).

    print(('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
           .format(top1=top1, top5=top5, loss=losses)))

    return top1.avg, top5.avg


def main(args):
    num_class = 101
    feature_length = 800
    unet_clip_num = 8
    # clf = Classifier(feature_length, num_class, isbn=False)
    # fe = I3DFeatureExtractor()
    # I3DClassifier = torch.nn.Sequential(
    #     fe,
    #     clf
    # )
    # I3DClassifier = torch.nn.DataParallel(I3DClassifier, device_ids=[0, 1, 2, 3]).cuda()

    from models.unet import UntrimmedNet
    I3DClassifier = UntrimmedNet(num_class, "result/0804_1708_e2e_ucf_model.pth.tar")
    I3DClassifier = torch.nn.DataParallel(I3DClassifier, device_ids=[0, 1, 2, 3]).cuda()

    ds = build_video_dataset("thumos_validation", train=True, unet=True, unet_clip_num=unet_clip_num)['dataset']
    eval_ds = build_video_dataset("thumos_test", train=False, unet=True, unet_clip_num=unet_clip_num)['dataset']

    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_worker, pin_memory=True)

    # eval_loader = torch.utils.data.DataLoader(
    #     eval_ds, batch_size=4, shuffle=True,
    #     num_workers=args.num_worker, collate_fn=lambda x:zip(*x),
    #     pin_memory=True)

    eval_loader = torch.utils.data.DataLoader(
        eval_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_worker, pin_memory=True)

    # criterion = torch.nn.CrossEntropyLoss().cuda()
    loss = torch.nn.NLLLoss().cuda()
    criterion = lambda x, y: loss(x.log(), y)
    optimizer = torch.optim.Adam(I3DClassifier.parameters(), lr=args.lr)

    best_prec1 = 0
    best_prec5 = 0


    save_path = 'result/' + args.exp_name
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train(loader, I3DClassifier, criterion, optimizer, num_class, unet_clip_num, epoch)

        # evaluate on validation set
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            prec1, prec5 = validate(eval_loader, I3DClassifier, criterion, num_class, unet_clip_num)

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
            }, is_best, args.exp_name, path=save_path)


    prec1, prec5 = validate(eval_loader, I3DClassifier, criterion, num_class, save_path)

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
    }, is_best, args.exp_name+"_final")

    print('Experiment {} finished! Best Accu@1 is {:.6f}, Best Accu@5 is {:.6f}.'.format(args.exp_name, best_prec1,
                                                                                         best_prec5))


if __name__ == '__main__':
    now = time.strftime('%m%d_%H%M', time.localtime(time.time()))
    parser = argparse.ArgumentParser(description="Find a good video representation")
    parser.add_argument('--exp-name', type=str, default=None)
    # parser.add_argument('--fuse-type', type=str, help='In average, cnn, max, concat')
    # parser.add_argument('--tsn', action='store_true', default=False)
    parser.add_argument('--num-worker', type=int, default=18,
                        help='how many training CPU processes to use (default: 16)')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--model', type=str, help='Should be in [cnn, mlp, lstm]!')
    parser.add_argument('--lr', type=float, default=8e-5)
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
