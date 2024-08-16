# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 15:22:33 2023

@author: Roy
"""
# -*- coding: utf-8 -*-
"""
Created on Sat May 28 22:02:39 2022

@author: kankana.roy
"""

import sys
import argparse
import time
import math
import numpy as np

import torch
import torch.backends.cudnn as cudnn

from main_ce import set_loader
from util import AverageMeter
from util import adjust_learning_rate, warmup_learning_rate, accuracy
from util import set_optimizer
from networks.network_vit import LinearClassifier
from torchvision import transforms, datasets
from sklearn.metrics import confusion_matrix, classification_report

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=20,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='60,75,90',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='tresnet')
    parser.add_argument('--dataset', type=str, default='path',
                        choices=['cifar10', 'cifar100','path'], help='dataset')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')

    parser.add_argument('--ckpt', type=str, default='/home/kroy/codes/project_hand_over_face/SupContrast-master/workdir/SupCon/path_models/SupCon_path_resnet50_lr_0.05_decay_0.0001_bsz_256_temp_0.07_trial_0/last.pth',
                        help='path to pre-trained model')
    parser.add_argument('--size', type=int, default=384, help='parameter for RandomResizedCrop')


    opt = parser.parse_args()

    # set the path according to the environment
    opt.data_folder = 'data/rgb/train'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_lr_{}_decay_{}_bsz_{}'.\
        format(opt.dataset, opt.model, opt.learning_rate, opt.weight_decay,
               opt.batch_size)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    if opt.dataset == 'cifar10':
        opt.n_cls = 10
    elif opt.dataset == 'cifar100':
        opt.n_cls = 100
    elif opt.dataset == 'path':
        opt.n_cls = 30
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))

    return opt


def set_model(opt):
    criterion = torch.nn.CrossEntropyLoss()

    classifier = LinearClassifier(name=opt.model, num_classes=opt.n_cls)
    #classifier.load_state_dict(torch.load('/content/gdrive/My Drive/epoch_11_classifier_HOF_supcon.pth', map_location='cpu'))

    classifier = classifier.cuda()
    criterion = criterion.cuda()
    cudnn.benchmark = True

    # if torch.cuda.is_available():
    #     if torch.cuda.device_count() > 1:
    #         model.encoder = torch.nn.DataParallel(model.encoder)
    #     else:
    #         new_state_dict = {}
    #         for k, v in state_dict.items():
    #             k = k.replace("module.", "")
    #             new_state_dict[k] = v
    #         state_dict = new_state_dict
    #     model = model.cuda()
    #     classifier = classifier.cuda()
    #     criterion = criterion.cuda()
    #     cudnn.benchmark = True

    #     model.load_state_dict(state_dict)

    return classifier, criterion


def train(train_loader, classifier, criterion, optimizer, epoch, opt):
    """one epoch training"""
    # for p in classifier.encoder.patch_embedding.tresnet.parameters():
    #     p.requires_grad = False
    classifier.train()
    # for p in classifier.encoder.patch_embedding[0].head.parameters():
    #     p.requires_grad = True
    #classifier.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        #with torch.no_grad():
        output = classifier(images)
        loss = criterion(output, labels)

        # update metric
        losses.update(loss.item(), bsz)
        #print(labels.size())
        acc1, acc5 = accuracy(output, labels, topk=(1, 5))
        #print(acc1.size())
        top1.update(acc1[0], bsz)
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    epoch, idx + 1, len(train_loader), batch_time=batch_time,
                    data_time=data_time, 
                    loss=losses, 
                    top1=top1
                    ))
            sys.stdout.flush()
        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.save(classifier.state_dict(), '/content/gdrive/My Drive/epoch_'+str(epoch)+'_classifier_HOF_supcon.pth')
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        

    return losses.avg, top1.avg


def validate(val_loader, classifier, criterion, opt):
    """validation"""
    classifier.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels) in enumerate(val_loader):
            images = images.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]

            # forward
            output = classifier(images)
            loss = criterion(output, labels)

            # update metric
            losses.update(loss.item(), bsz)
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            top1.update(acc1[0], bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time,
                       loss=losses, top1=top1))

    print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
    return losses.avg, top1.avg


def main():
    best_acc = 0
    opt = parse_option()

    # build data loader
    train_loader, val_loader, test_loader = set_loader(opt)

    # build model and criterion
    classifier, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, classifier)

    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss, acc = train(train_loader, classifier, criterion,
                          optimizer, epoch, opt)
        time2 = time.time()
        print('Train epoch {}, total time {:.2f}, accuracy:{:.2f}'.format(
            epoch, time2 - time1, acc))

        # eval for one epoch
        loss, val_acc = validate(val_loader, classifier, criterion, opt)
        if val_acc > best_acc:
            best_acc = val_acc

    print('best accuracy: {:.2f}'.format(best_acc))

    classifier.eval()
    correct = 0
    total = 0
    with torch.no_grad():
               # Set model to evaluate mode
                classifier.eval()
                # Iterate over data.
                all_preds = torch.tensor([])
                all_labels = torch.tensor([])
                for inputs, labels in test_loader:
                    images = inputs.float().cuda()
                    labels = labels.cuda()

                    # forward
                    output = classifier(images)
                    _, predicted = torch.max(output.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    all_preds = torch.cat((all_preds, output.argmax(dim=1).detach().cpu()),dim=0)
                    all_labels = torch.cat((all_labels, labels.detach().cpu()),dim=0)
        
                print('Accuracy of the network on the test images: %f %%' % ( 100 * correct / total))  
                conf_mat = confusion_matrix(all_labels, all_preds)
                np.set_printoptions(threshold=np.inf)
                print(conf_mat)
                print(classification_report(all_labels, all_preds))
#                conf_mat = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
                #conf_mat=confusion_matrix(lbllist.numpy(), predlist.numpy())
#                df_cm = pd.DataFrame(conf_mat, range(num_classes), range(num_classes))
#                fig, ax = plt.subplots(figsize=(10,10))
#                x_axis_labels = [1,2,3,4,5,6,7] # labels for x-axis
#                y_axis_labels = [1,2,3,4,5,6,7] # labels for y-axis
#                sns_plot = sns.heatmap(df_cm, annot=True, fmt='.2', cmap='Blues',ax=ax,xticklabels=num_classes, yticklabels=num_classes)
#                sns_plot_=sns_plot.get_figure()
#                sns_plot_.savefig("first_person_hand_color_set1_resnet152_confusion_matrix".png")


if __name__ == '__main__':
    main()
