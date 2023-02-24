"""
file - main.py
Main script to train the aesthetic model on the AVA dataset.

Copyright (C) Yunxiao Shi 2017 - 2021
NIMA is released under the MIT license. See LICENSE for the fill license text.
"""

import argparse
import data_loader
import random
import os

import numpy as np
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt

import torch
import torch.autograd as autograd
import torch.optim as optim

import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torchvision.models as models

from torch.utils.tensorboard import SummaryWriter

from dataset.dataset import ourDataset, ourDataset1

from model.model import *

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

def main(config):

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter()

    # train_transform = transforms.Compose([
    #     # transforms.Scale(256),
    #     # transforms.RandomCrop(224),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #         std=[0.229, 0.224, 0.225])])
    #
    # val_transform = transforms.Compose([
    #     # transforms.Scale(256),
    #     # transforms.RandomCrop(224),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #         std=[0.229, 0.224, 0.225])])

    base_model = models.vgg16(pretrained=True)
    model = NIMA(base_model).cuda()

    if config.warm_start:
        model.load_state_dict(torch.load(os.path.join(config.ckpt_path, 'epoch-%d.pth' % config.warm_start_epoch)))
        print('Successfully loaded model epoch-%d.pth' % config.warm_start_epoch)

    # if config.multi_gpu:
    #     model.features = torch.nn.DataParallel(model.features, device_ids=config.gpu_ids)
    #     model = model.to(device)
    # else:
    #     model = model.to(device)

    conv_base_lr = config.conv_base_lr
    dense_lr = config.dense_lr
    optimizer = optim.SGD([
        {'params': model.features.parameters(), 'lr': conv_base_lr},
        {'params': model.classifier.parameters(), 'lr': dense_lr}],
        momentum=0.9
        )

    param_num = 0
    for param in model.parameters():
        if param.requires_grad:
            param_num += param.numel()
    print('Trainable params: %.2f million' % (param_num / 1e6))

    # if config.train:
    # data_path = '/mnt/disk10T/fqy/dataset/FQY-dataset/'
    # data_path = '/mnt/disk10T/fqy/hyperIQA-master/tid2013/'
    data_path = '/mnt/disk10T/fqy/dataset/kadid10k/'
    # id_test = np.arange(17472)
    # train_id = id_test.tolist()
    # id_test = np.arange(2496)
    # test_id = id_test.tolist()
    # trainset = ourDataset(root=data_path, transform=train_transform)
    # valset = ourDataset1(root=data_path, transform=val_transform)

    # train_loader = torch.utils.data.DataLoader(trainset, batch_size=config.train_batch_size,
    #     shuffle=True, num_workers=config.num_workers)
    # val_loader = torch.utils.data.DataLoader(valset, batch_size=config.val_batch_size,
    #     shuffle=False, num_workers=config.num_workers)
    sel_num = list(range(0, 81))
    random.shuffle(sel_num)
    # train_id = [23, 7, 24, 18, 11, 1, 0, 17, 13, 19, 21, 15, 22, 14, 12, 2, 8, 10, 3, 4]
    # test_id = [20, 6, 16, 9, 5]
    train_id = [80, 14, 41, 26, 55, 38, 72, 23, 74, 34, 47, 3, 46, 32, 36, 13, 50, 69, 2, 37, 9, 18, 59, 12, 56, 10, 61, 15, 71,
     20, 24, 0, 49, 77, 58, 76, 21, 78, 54, 65, 52, 25, 42, 40, 17, 68, 63, 62, 6, 43, 31, 19, 35, 11, 75, 45, 60, 28,
     70, 66, 16, 33, 67, 39, 73]
    test_id = [22, 5, 1, 8, 53, 4, 48, 44, 30, 27, 51, 29, 64, 7, 57, 79]

    # train_id = sel_num[0:int(round(0.8 * len(sel_num)))]
    # test_id = sel_num[int(round(0.8 * len(sel_num))):len(sel_num)]
    train_loader = data_loader.DataLoader('kadid_10k', data_path, train_id, 224, 1,
                                          batch_size=config.train_batch_size, istrain=True)
    test_loader = data_loader.DataLoader('kadid_10k', data_path, test_id, 224, 1, istrain=False)
    train_data = train_loader.get_data()
    test_data = test_loader.get_data()
    # for early stopping
    count = 0
    init_val_loss = float('inf')
    train_losses = []
    val_losses = []
    for epoch in range(config.warm_start_epoch, config.epochs):
        batch_losses = []
        # for data in train_loader:
        for img, label in train_data:
            # images = data['image'].to(device)
            # labels = data['annotations'].to(device).float()
            images = torch.tensor(img.cuda())
            labels = torch.tensor(label.cuda())
            # labels = labels.view(-1, 10, 1)

            outputs = model(images)
            outputs = outputs.view(-1, 10, 1)
            labels = labels.view(-1, 10, 1)
            # outputs = outputs.view(10, 1)
            # pred = 0.0
            # preds = []
            # for i, elem in enumerate(outputs, 1):
            #     pred = 0.0
            #     for j, k in enumerate(elem):
            #         pred += j * k
            #     preds.append(pred)
            # preds = torch.tensor(preds)
            # preds = preds.requires_grad_()

            optimizer.zero_grad()

            loss = emd_loss(labels, outputs)
            # l1_loss = torch.nn.L1Loss().cuda()
            # loss = l1_loss(preds.squeeze(), label.float().detach())
            batch_losses.append(loss.item())

            loss.backward()

            optimizer.step()

            # print('Epoch: %d/%d | Step: %d/%d | Training EMD loss: %.4f' % (epoch + 1, config.epochs, i + 1, len(trainset) // config.train_batch_size + 1, loss.data[0]))
            # writer.add_scalar('batch train loss', loss.data[0], i + epoch * (len(trainset) // config.train_batch_size + 1))

        avg_loss = sum(batch_losses) / (len(train_data) // config.train_batch_size + 1)
        train_losses.append(avg_loss)
        print('Epoch %d mean training EMD loss: %.4f' % (epoch + 1, avg_loss))

        # exponetial learning rate decay
        # if config.decay:
        if (epoch + 1) % 10 == 0:
            conv_base_lr = conv_base_lr * config.lr_decay_rate ** ((epoch + 1) / config.lr_decay_freq)
            dense_lr = dense_lr * config.lr_decay_rate ** ((epoch + 1) / config.lr_decay_freq)
            optimizer = optim.SGD([
                {'params': model.features.parameters(), 'lr': conv_base_lr},
                {'params': model.classifier.parameters(), 'lr': dense_lr}],
                momentum=0.9
            )
        print('Saving model...')
        torch.save(model.state_dict(), os.path.join(config.ckpt_path, 'epoch-%d.pth' % (epoch + 1)))
        print('Done.\n')
        # do validation after each epoch
        # batch_val_losses = []
        # # for data in val_loader:
        # #     images = data['image'].to(device)
        # #     labels = data['annotations'].to(device).float()
        # for img, label in test_data:
        #     images = torch.tensor(img.cuda())
        #     labels = torch.tensor(label.cuda())
        #     with torch.no_grad():
        #         outputs = model(images)
        #     # outputs = outputs.view(-1, 10, 1)
        #     outputs = outputs.view(10, 1)
        #     # val_loss = emd_loss(labels, outputs)
        #     pred = 0.0
        #     for i, elem in enumerate(outputs, 1):
        #         pred += i * elem
        #     # loss = emd_loss(labels, outputs)
        #     l1_loss = torch.nn.L1Loss().cuda()
        #     val_loss = l1_loss(pred.squeeze(), label.float().detach())
        #     batch_val_losses.append(val_loss.item())
        # avg_val_loss = sum(batch_val_losses) / (len(test_data) // config.val_batch_size + 1)
        # val_losses.append(avg_val_loss)
        # print('Epoch %d completed. Mean EMD loss on val set: %.4f.' % (epoch + 1, avg_val_loss))
        # # writer.add_scalars('epoch losses', {'epoch train loss': avg_loss, 'epoch val loss': avg_val_loss}, epoch + 1)
        #
        # # Use early stopping to monitor training
        # if avg_val_loss < init_val_loss:
        #     init_val_loss = avg_val_loss
        #     # save model weights if val loss decreases
        #     print('Saving model...')
        #     if not os.path.exists(config.ckpt_path):
        #         os.makedirs(config.ckpt_path)
        #     torch.save(model.state_dict(), os.path.join(config.ckpt_path, 'epoch-%d.pth' % (epoch + 1)))
        #     print('Done.\n')
        #     # reset count
        #     count = 0
        # elif avg_val_loss >= init_val_loss:
        #     count += 1
        #     if count == config.early_stopping_patience:
        #         print('Val EMD loss has not decreased in %d epochs. Training terminated.' % config.early_stopping_patience)
        #         break

    print('Training completed.')

    '''
    # use tensorboard to log statistics instead
    if config.save_fig:
        # plot train and val loss
        epochs = range(1, epoch + 2)
        plt.plot(epochs, train_losses, 'b-', label='train loss')
        plt.plot(epochs, val_losses, 'g-', label='val loss')
        plt.title('EMD loss')
        plt.legend()
        plt.savefig('./loss.png')
    '''

    # if config.test:
    #     model.eval()
    #     # compute mean score
    #     test_transform = val_transform
    #     testset = AVADataset(csv_file=config.test_csv_file, root_dir=config.img_path, transform=val_transform)
    #     test_loader = torch.utils.data.DataLoader(testset, batch_size=config.test_batch_size, shuffle=False, num_workers=config.num_workers)
    #
    #     mean_preds = []
    #     std_preds = []
    #     for data in test_loader:
    #         image = data['image'].to(device)
    #         output = model(image)
    #         output = output.view(10, 1)
    #         predicted_mean, predicted_std = 0.0, 0.0
    #         for i, elem in enumerate(output, 1):
    #             predicted_mean += i * elem
    #         for j, elem in enumerate(output, 1):
    #             predicted_std += elem * (j - predicted_mean) ** 2
    #         predicted_std = predicted_std ** 0.5
    #         mean_preds.append(predicted_mean)
    #         std_preds.append(predicted_std)
        # Do what you want with predicted and std...


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # input parameters
    # parser.add_argument('--img_path', type=str, default='./data/images')
    # parser.add_argument('--train_csv_file', type=str, default='./data/train_labels.csv')
    # parser.add_argument('--val_csv_file', type=str, default='./data/val_labels.csv')
    # parser.add_argument('--test_csv_file', type=str, default='./data/test_labels.csv')

    # training parameters
    # parser.add_argument('--train',action='store_true')
    # parser.add_argument('--test', action='store_true')
    # parser.add_argument('--decay', action='store_true')
    parser.add_argument('--conv_base_lr', type=float, default=5e-3)
    parser.add_argument('--dense_lr', type=float, default=5e-4)
    parser.add_argument('--lr_decay_rate', type=float, default=0.95)
    parser.add_argument('--lr_decay_freq', type=int, default=10)
    parser.add_argument('--train_batch_size', type=int, default=64)
    parser.add_argument('--val_batch_size', type=int, default=96)
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=1000)

    # misc
    parser.add_argument('--ckpt_path', type=str, default='/mnt/disk10T/fqy/Neural-IMage-Assessment-master/our_models_kadid10k')
    parser.add_argument('--multi_gpu', action='store_true')
    parser.add_argument('--gpu_ids', type=list, default=None)
    parser.add_argument('--warm_start', action='store_true')
    parser.add_argument('--warm_start_epoch', type=int, default=0)
    parser.add_argument('--early_stopping_patience', type=int, default=10)
    parser.add_argument('--save_fig', action='store_true')

    config = parser.parse_args()

    main(config)

