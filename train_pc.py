# -*- coding: utf-8 -*-
import os
import os.path as osp

from models import MeshNetV1 as meshnet
from models import PointNetCls as pcnet
from datasets import data_pth
from utils import config
from utils import meter
import utils
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

    for i, (meshes, adjs, labels) in enumerate(train_loader):
        batch_time.reset()
        # bz x n x 3
        meshes = meshes.transpose(1, 2)
        labels = labels.long().view(-1)

        if cfg.cuda:
            meshes = meshes.cuda()
            adjs = adjs.cuda()
            labels = labels.cuda()

        preds, _ = model(meshes)  # bz x C x H x W

        loss = criterion(preds, labels)

        # print('pred: %d, gt: %d'%(torch.argmax(preds.cpu().data), labels.item()))
        prec.add(preds.cpu().data.numpy(), labels.item())
        losses.add(loss.item(), preds.size(0))  # batchsize

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

    for i, (meshes, adjs, labels) in enumerate(val_loader):
        batch_time.reset()
        # bz x n x 3
        meshes = meshes.transpose(1, 2)
        labels = labels.long().view(-1)

        # shift data to GPU
        if cfg.cuda:
            meshes = meshes.cuda()
            adjs = adjs.cuda()
            labels = labels.cuda()

        # forward, backward optimize
        preds, _ = model(meshes)

        prec.add(preds.cpu().data.numpy(), labels.item())

        if i % cfg.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time:.3f}\t'
                  'Epoch Time {data_time:.3f}\t'
                  'Prec@1 {top1:.3f}\t'.format(
                epoch, i, len(val_loader), batch_time=batch_time.value(),
                data_time=data_time.value(), top1=prec.value(1)))

    print('mean class accuracy at epoch {0}: {1} '.format(epoch, prec.value(1)))

    return prec.value(1)


def train_pc(train_loader, model, criterion, optimizer, epoch, cfg):
    """
    train for one epoch on the training set
    """
    batch_time = meter.TimeMeter(True)
    data_time = meter.TimeMeter(True)
    losses = meter.AverageValueMeter()
    prec = meter.ClassErrorMeter(topk=[1], accuracy=True)
    # training mode
    model.train()

    for i, (meshes, labels) in enumerate(train_loader):
        batch_time.reset()
        # bz x n x 3
        labels = labels.long().view(-1)
        meshes = meshes.squeeze(3)

        if cfg.cuda:
            meshes = meshes.cuda()
            labels = labels.cuda()

        preds, _ = model(meshes)  # bz x C x H x W

        loss = criterion(preds, labels)

        # print('pred: %d, gt: %d'%(torch.argmax(preds.cpu().data), labels.item()))
        prec.add(preds.cpu().data.numpy(), labels.item())
        losses.add(loss.item(), preds.size(0))  # batchsize

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


def validate_pc(val_loader, model, epoch, cfg):
    """
    validation for one epoch on the val set
    """
    batch_time = meter.TimeMeter(True)
    data_time = meter.TimeMeter(True)
    prec = meter.ClassErrorMeter(topk=[1], accuracy=True)

    # testing mode
    model.eval()

    for i, (meshes, labels) in enumerate(val_loader):
        batch_time.reset()
        # bz x n x 3
        labels = labels.long().view(-1)
        meshes = meshes.squeeze(3)

        # shift data to GPU
        if cfg.cuda:
            meshes = meshes.cuda()
            labels = labels.cuda()

        # forward, backward optimize
        preds, _ = model(meshes)

        prec.add(preds.cpu().data.numpy(), labels.item())

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
    # cfg = config(osp.join(osp.split(__file__)[0], 'config/config.cfg'))
    cfg = config('config/config.cfg')
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpu
    train_dataset = data_pth.MeshData(cfg, state='train')
    val_dataset = data_pth.MeshData(cfg, state='val')
    # train_dataset = data_pth.PCData(cfg, state='train')
    # val_dataset = data_pth.PCData(cfg, state='val')

    print('number of train samples is: ', len(train_dataset))
    print('number of val samples is: ', len(val_dataset))

    best_prec1 = 0
    prec1 = 0
    resume_epoch = 0
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.workers)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.workers)

    # create model
    model = pcnet(3, 40)
    # print(model)

    if cfg.resume_train:
        print('loading pretrained model from {0}'.format(cfg.ckpt_model))
        checkpoint = torch.load(cfg.ckpt_model)
        # state_dict = model_helper.get_state_dict(model.state_dict(), checkpoint['model_param_best'])
        # model.load_state_dict(state_dict)
        model.load_state_dict(checkpoint['model_param_best'])

    # optimizer
    # optimizer = optim.SGD(model.parameters(), cfg.lr, weight_decay=cfg.weight_decay)
    optimizer = optim.Adam(model.parameters(), cfg.lr, weight_decay=cfg.weight_decay)

    if cfg.resume_train and osp.exists(cfg.ckpt_optim):
        print('loading optim model from {0}'.format(cfg.ckpt_optim))
        optim_state = torch.load(cfg.ckpt_optim)

        resume_epoch = optim_state['epoch']
        best_prec1 = optim_state['best_prec1']
        optimizer.load_state_dict(optim_state['optim_state_best'])

    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, 20, 0.1)

    criterion = nn.CrossEntropyLoss()

    if cfg.cuda:
        print('shift model and criterion to GPU .. ')
        model = model.cuda()
        criterion = criterion.cuda()

    for epoch in range(resume_epoch, cfg.max_epoch):

        lr_scheduler.step(epoch=epoch)
        # train_pc(train_loader, model, criterion, optimizer, epoch, cfg)
        # prec1 = validate_pc(val_loader, model, epoch, cfg)
        train(train_loader, model, criterion, optimizer, epoch, cfg)
        prec1 = validate(val_loader, model, epoch, cfg)

        # save checkpoints
        if best_prec1 < prec1:
            best_prec1 = prec1
            train_helper.save_ckpt(cfg, model, epoch, best_prec1, optimizer)
        print('best accuracy: ', best_prec1)


if __name__ == '__main__':
    main()

