import argparse
import os
import time

# import cudnn
from dataset import FeatureDataset
from ops.fuser import Fuser
from ops.utils import *
from models.module import buildClassifier


def padding(data, maxlength):
    tensor = torch.unsqueeze(torch.from_numpy(data), 0)
    pad = torch.nn.ZeroPad2d((0, 0, 0, maxlength - tensor.shape[2]))(tensor)
    return pad


def lstmpadding(data, maxlength):
    tensor = torch.from_numpy(data)
    zeros = torch.zeros([maxlength - tensor.shape[0], tensor.shape[1]])
    return torch.cat([tensor, zeros], 0)


def cnn_fuse_collate_fn(batch):
    # Input
    maxlength = max([s[0].shape[1] for s in batch])
    features = torch.cat([padding(s[0], maxlength) for s in batch])
    labels = torch.Tensor([s[1] for s in batch])
    return [features, labels]


def lstm_fuse_collate_fn(batch):
    # Input
    maxlength = max([s[0].shape[0] for s in batch])
    features = torch.stack([lstmpadding(s[0], maxlength) for s in batch])
    labels = torch.Tensor([s[1] for s in batch])
    # print('Feature shape', features.shape)
    # print('labels shape', labels.shape)
    return [features, labels]


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()
    for i, data in enumerate(train_loader):
        input, target = data['data'], data['label']
        # measure data loading time
        data_time.update(time.time() - end)
        target_var = target.cuda(async=True).type(torch.cuda.LongTensor)
        input_var = input.cuda(async=True)
        # target_var = torch.autograd.Variable(target)

        # compute ouput
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target_var, topk=(1, 5))
        losses.update(loss.item(), input.shape[0])
        top1.update(prec1.item(), input.shape[0])
        top5.update(prec5.item(), input.shape[0])

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

        if i % args.print_freq == 0:
            print(('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                   'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                   'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                   'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5, lr=optimizer.param_groups[-1]['lr'])))


def validate(val_loader, model, criterion, iter, logger=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, data in enumerate(val_loader):
        input, target = data['data'], data['label']
        with torch.no_grad():
            input_var = input.cuda(async=True)
            target_var = target.cuda(async=True).type(torch.cuda.LongTensor)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target_var, topk=(1, 5))

        losses.update(loss.item(), input.shape[0])
        top1.update(prec1.item(), input.shape[0])
        top5.update(prec5.item(), input.shape[0])

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

    print(('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
           .format(top1=top1, top5=top5, loss=losses)))

    return top1.avg, top5.avg


def main(args):
    fuser = Fuser(fuse_type=args.fuse_type)
    eval_dataset = FeatureDataset('features/thumos14/test/data.csv', fuser, is_thumos14_test_folder=True)
    train_dataset = FeatureDataset('features/thumos14/val/data.csv', fuser)

    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False,
                                              num_workers=2,
                                              pin_memory=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=2,
                                               pin_memory=True)
    num_class = 101
    feature_length = 800
    # model = Classifier(feature_length, num_class)
    model = buildClassifier('result/clf_avg_ucf_all8_model.pth.tar')
    print('FUCK YOU ASSHOLE! NOW I SEE {} DEVICES!'.format(torch.cuda.device_count()))
    model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])

    # if restore
    if args.restore:
        if os.path.isfile(args.restore):
            print(("=> loading checkpoint '{}'".format(args.restore)))
            checkpoint = torch.load(args.restore)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print(("=> loaded checkpoint '{}' (epoch {})"
                   .format(args.evaluate, checkpoint['epoch'])))
        else:
            print(("=> no checkpoint found at '{}'".format(args.restore)))

    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_prec1 = 0
    best_prec5 = 0

    if args.evaluate:
        validate(eval_loader, model, criterion, 0)
        return

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            prec1, prec5 = validate(eval_loader, model, criterion, (epoch + 1) * len(train_loader))

            # remember best prec@1 and save checkpoint
            is_best = (prec1 > best_prec1) or (prec5 > best_prec5)

            best_prec1 = max(prec1, best_prec1)
            best_prec5 = max(prec5, best_prec5)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.model,
                'tsn_like': args.tsn,
                'state_dict': model.state_dict(),
                'fuse_type': args.fuse_type,
                'best_prec1': best_prec1,
                'best_prec5': best_prec5
            }, is_best, args.exp_name)

    print('Experiment {} finished! Best Accu@1 is {:.6f}, Best Accu@5 is {:.6f}.'.format(args.exp_name, best_prec1,
                                                                                         best_prec5))


if __name__ == '__main__':
    now = time.strftime('%m%d_%H%M', time.localtime(time.time()))
    parser = argparse.ArgumentParser(description="Find a good video representation")
    parser.add_argument('--exp-name', type=str, default=None)
    parser.add_argument('--fuse-type', type=str, help='In average, cnn, max, concat')
    parser.add_argument('--tsn', action='store_true', default=False)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--model', type=str, help='Should be in [cnn, mlp, lstm]!')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--evaluate', action='store_true', default=False)
    parser.add_argument('--start-epoch', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=-1)
    parser.add_argument('--dataset', type=str, default='hmdb51')
    parser.add_argument('--restore', type=str, help='The path toward checkpoint file.', default='')
    parser.add_argument('--eval-freq', type=int, default=1)
    parser.add_argument('--clip-gradient', '--gd', default=None, type=float,
                        metavar='W', help='gradient norm clipping (default: disabled)')
    parser.add_argument('--print-freq', type=int, default=100)
    args = parser.parse_args()

    if (args.restore == '') and (args.start_epoch != 0):
        raise ValueError('When you are not restore from checkpoint, you should not set start_epoch different from 0!')
    if args.epochs == -1:
        raise ValueError("You should explicit declare args.epoch!")

    if args.tsn and (args.fuse_type in ['cnn']):
        raise ValueError('When agrs.tsn is True, you cannot use cnn fuse_type!')

    if (not args.tsn) and (args.fuse_type in ['concat']):
        raise ValueError('When agrs.tsn is False, you cannot use concat fuse_type!')

    if args.model == 'cnn':
        if args.fuse_type != 'cnn':
            raise ValueError('Fuse_type and model should be cnn simultaneously!')
    elif args.model == 'mlp':
        if args.fuse_type == 'cnn':
            raise ValueError("Fuse_type cannot be cnn when model is MLP!")
    elif args.model == 'lstm':
        if args.fuse_type != 'lstm':
            raise ValueError("LSTM should be with fuse_type lstm.")
        if args.tsn:
            raise ValueError("LSTM should not be with --tsn")
    else:
        raise ValueError("Unknown args.model!")

    if args.exp_name == None:
        args.exp_name = "{}_{}_{}".format(now, str(args.fuse_type), str(args.model))

    # os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
    print('Experiment {} start!'.format(args.exp_name))
    main(args)
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3, 4, 5, 6, 7"
