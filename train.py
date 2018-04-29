# -*- coding: utf-8 -*-
import os
import os.path as osp

from models import MeshNetV0
from datasets import data_pth
from utils import meter
import train_helper

import torch
from torch import nn
from torch import optim
from torch.utils import data


def train(train_loader, model, criterion, optimizer, epoch, cfg):
    """
    train for one epoch on the training set
    """
    batch_time = meter.TimeMeter(True)
    data_time = meter.TimeMeter(True)
    losses = meter.AverageValueMeter()
    prec = meter.ClassErrorMeter(topk=[1], accuracy=True)
    # training mode
    model.train()

    for i, (pcs, labels) in enumerate(train_loader):
        batch_time.reset()
        # bz x 12 x 3 x 224 x 224
        labels = labels.long().view(-1)
        pcs = Variable(pcs)
        labels = Variable(labels)

        if cfg.cuda:
            pcs = pcs.cuda()
            labels = labels.cuda()

        preds = model(pcs)  # bz x C x H x W

        if cfg.have_aux:
            preds, aux = preds
            loss_main = criterion(preds, labels)
            loss_aux = criterion(aux, labels)
            softmax_loss = loss_main + 0.3 * loss_aux
        else:
            softmax_loss = criterion(preds, labels)

        loss = softmax_loss

        prec.add(preds.data, labels.data)
        losses.add(loss.data[0], preds.size(0))  # batchsize

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % cfg.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time:.3f}\t'
                  'Epoch Time {data_time:.3f}\t'
                  'Loss {loss:.4f} \t'
                  'Prec@1 {top1:.3f}\t'.format(
                epoch, i, len(train_loader), batch_time=batch_time.value(),
                data_time=data_time.value(), loss=losses.value()[0], top1=prec.value(1)))

    print('prec at epoch {0}: {1} '.format(epoch, prec.value(1)))


def validate(val_loader, model, epoch, cfg):
    """
    validation for one epoch on the val set
    """
    batch_time = meter.TimeMeter(True)
    data_time = meter.TimeMeter(True)
    prec = meter.ClassErrorMeter(topk=[1], accuracy=True)

    # testing mode
    model.eval()

    for i, (pcs, labels) in enumerate(val_loader):
        batch_time.reset()
        # bz x 12 x 3 x 224 x 224
        labels = labels.long().view(-1)
        pcs = Variable(pcs)
        labels = Variable(labels)

        # shift data to GPU
        if cfg.cuda:
            pcs = pcs.cuda()
            labels = labels.cuda()

        # forward, backward optimize
        preds = model(pcs)

        if cfg.have_aux:
            preds, aux = preds

        prec.add(preds.data, labels.data)

        if i % cfg.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time:.3f}\t'
                  'Epoch Time {data_time:.3f}\t'
                  'Prec@1 {top1:.3f}\t'.format(
                epoch, i, len(val_loader), batch_time=batch_time.value(),
                data_time=data_time.value(), top1=prec.value(1)))

    print('mean class accuracy at epoch {0}: {1} '.format(epoch, prec.value(1)))

    return prec.value(1)


def main():
    cfg = utils.config.config()
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpu
    train_dataset = data_pth.pc_data(cfg, state='train')
    val_dataset = data_pth.pc_data(cfg, state='val')

    print('number of train samples is: ', len(train_dataset))
    print('number of val samples is: ', len(val_dataset))

    best_prec1 = 0
    prec1 = 0
    resume_epoch = 0
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size,
                                               shuffle=True, num_workers=cfg.workers)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.batch_size,
                                              shuffle=True, num_workers=cfg.workers)

    # create model
    model = point_with_attention.point_net_ours(cfg, num_classes=40)

    if cfg.resume_train:
        print('loading pretrained model from {0}'.format(cfg.ckpt_model))
        checkpoint = torch.load(cfg.ckpt_model)
        # state_dict = model_helper.get_state_dict(model.state_dict(), checkpoint['model_param_best'])
        # model.load_state_dict(state_dict)
        model.load_state_dict(checkpoint['model_param_best'])

    # optimizer
    optimizer = optim.SGD(model.parameters(), cfg.lr,
                          momentum=cfg.momentum,
                          weight_decay=cfg.weight_decay)

    if cfg.resume_train and osp.exists(cfg.ckpt_optim):
        print('loading optim model from {0}'.format(cfg.ckpt_optim))
        optim_state = torch.load(cfg.ckpt_optim)

        resume_epoch = optim_state['epoch']
        best_prec1 = optim_state['best_prec1']
        optimizer.load_state_dict(optim_state['optim_state_best'])

    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, 10, 0.1)

    criterion = nn.CrossEntropyLoss()

    if cfg.cuda:
        print('shift model and criterion to GPU .. ')
        model = model.cuda()
        criterion = criterion.cuda()

    for epoch in range(resume_epoch, cfg.max_epoch):

        lr_scheduler.step(epoch=epoch)
        train(train_loader, model, criterion, optimizer, epoch, cfg)
        prec1 = validate(val_loader, model, epoch, cfg)

        # save checkpoints
        if best_prec1 < prec1:
            best_prec1 = prec1
            train_helper.save_ckpt(cfg, model, epoch, best_prec1, optimizer)
        print('best accuracy: ', best_prec1)


if __name__ == '__main__':
    main()

