
from init import Options
from networks import build_net, update_learning_rate, build_UNETR, My_build_UNETR, TransformerClassifier
from util.dataset2 import Dataset
# from networks import build_net
import logging
import os
import sys
import tempfile
from glob import glob

# import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable

import monai
from monai.data import create_test_image_3d, list_data_collate, decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import (EnsureType, Compose, LoadImaged, AddChanneld, Transpose,Activations,AsDiscrete, RandGaussianSmoothd, CropForegroundd, SpatialPadd,
                              ScaleIntensityd, ToTensord, RandSpatialCropd, Rand3DElasticd, RandAffined, RandZoomd,
    Spacingd, Orientationd, Resized, ThresholdIntensityd, RandShiftIntensityd, BorderPadd, RandGaussianNoised, RandAdjustContrastd,NormalizeIntensityd,RandFlipd)

from monai.visualize import plot_2d_or_3d_image

def main():
    opt = Options().parse()
    # monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    # check gpus
    if opt.gpu_ids != '-1':
        num_gpus = len(opt.gpu_ids.split(','))
    else:
        num_gpus = 0
    print('number of GPU:', num_gpus)

    
    root = '/home/murad/summer semester/Melbourne/Dataset'

    # create a training data loader
    # check_train = monai.data.Dataset(data=train_dicts, transform=train_transforms)
    trainset = Dataset(root=root, load_set='train', transform=None)
    train_loader = DataLoader(trainset, batch_size=opt.batch_size, shuffle=True, collate_fn=list_data_collate, num_workers=opt.workers, pin_memory=False)

    # create a training_dice data loader
    # check_val = monai.data.Dataset(data=train_dice_dicts, transform=val_transforms)
    # train_dice_loader = DataLoader(check_val, batch_size=1, num_workers=opt.workers, collate_fn=list_data_collate, pin_memory=False)

    # create a validation data loader
    # check_val = monai.data.Dataset(data=val_dicts, transform=val_transforms)
    valset = Dataset(root=root, load_set='val', transform=None)
    val_loader = DataLoader(valset, batch_size=1, num_workers=opt.workers, collate_fn=list_data_collate, pin_memory=False)

    # create a validation data loader
    # check_val = monai.data.Dataset(data=test_dicts, transform=val_transforms)
    # test_loader = DataLoader(check_val, batch_size=1, num_workers=opt.workers, collate_fn=list_data_collate, pin_memory=False)


       # build the network
    if opt.network is 'nnunet':
        net = build_net()  # nn build_net
    elif opt.network is 'unetr':
        net = build_UNETR() # UneTR
    elif opt.network is 'transformer':
        net = TransformerClassifier(num_channels=5, signal_length=4960)

    print('we are here1')
    # print(net)
    net.cuda()
    print('we are here')
    # if num_gpus > 1:
        # net = torch.nn.DataParallel(net)

    if opt.preload is not None:
        net.load_state_dict(torch.load(opt.preload))

    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    post_trans = Compose([EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold_values=True)])

    #loss_function = monai.losses.DiceCELoss(sigmoid=True)
    # loss_function = torch.nn.CrossEntropyLoss()
    loss_function = torch.nn.BCELoss()
    # loss_function = torch.nn.L1Loss()
    # loss_function = torch.nn.MSELoss()
    torch.backends.cudnn.benchmark = opt.benchmark
    lambda1 = 4
    lambda2 = 4500


    if opt.network is 'nnunet':

        optim = torch.optim.SGD(net.parameters(), lr=opt.lr, momentum=0.99, weight_decay=3e-5, nesterov=True,)
        net_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lambda epoch: (1 - epoch / opt.epochs) ** 0.9)

    elif opt.network is 'unetr':

        optim = torch.optim.AdamW(net.parameters(), lr=5e-5, weight_decay=1e-5)
        net_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lambda epoch: (1 - epoch / opt.epochs) ** 0.9)
        
    elif opt.network is 'transformer':

        optim = torch.optim.AdamW(net.parameters(), lr=5e-4, weight_decay=1e-5)
        net_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lambda epoch: (1 - epoch / opt.epochs) ** 0.9)

    if opt.test is True:
    
        print('Begin testing the network...')
        
        net.load_state_dict(torch.load(opt.output_file+'3'+'.pkl'))
        test_inputs=[]
        test_label=[]
        test_pred=[]
        
        running_loss = 0.0
        for v, val_data in enumerate(val_loader):
            # get the inputs
            inputs, label = val_data
            dataset_len = 17516
            
            inputs = np.einsum('ijk->ikj', inputs)
            label = np.einsum('ijk->ikj', label)
            
            inputs = torch.from_numpy(inputs)
            label = torch.from_numpy(label)
            
            # print("input shape is->>>>>>>>>>", inputs.shape)
            
            inputs = Variable(inputs)
            label = Variable(label)
            # dataset_len = Variable(dataset_len)
            
            inputs = inputs.float().cuda()
            label = label.float().cuda()
            # dataset_len = dataset_len.float().cuda()
    
            pred = net(inputs)
            
            # threshold = 0.5
            # pred = (pred >= threshold).float()
            # pred = pred.int()
            
            test_pred.append(pred.cpu().detach().numpy())
            test_inputs.append(inputs.cpu().detach().numpy())
            test_label.append(label.cpu().detach().numpy())

            # loss = (lambda2)* loss_function(fecg_pred, fecg_label) + lambda1 * loss_function(mecg_pred, mecg_label)
            loss = loss_function(pred, label)
            running_loss += loss.data
            
            
        print('val error: %.5f' % (running_loss / (v+1)))
        
        test_pred = np.array(test_pred,dtype=object)
        test_inputs = np.array(test_inputs,dtype=object)
        test_label = np.array(test_label,dtype=object)
        
        np.save('test_inputs.npy',test_inputs)
        np.save('test_pred.npy',test_pred)
        np.save('test_label.npy',test_label)
        
    else:
    
        # start a typical PyTorch training
        val_interval = 1
        best_metric = -1
        best_metric_epoch = -1
        epoch_loss_values = list()
        losses = []
        # writer = SummaryWriter()
        for epoch in range(opt.epochs):
            print("-" * 10)
            print(f"epoch {epoch + 1}/{opt.epochs}")
            net.train()
            epoch_loss = 0
            step = 0
            # for batch_data in train_loader:
            for i, tr_data in enumerate(train_loader):
                step += 1
                # inputs, labels = batch_data["image"].cuda(), batch_data["label"].cuda()
                inputs, label = tr_data
                dataset_len = 54360
                
                inputs = np.einsum('ijk->ikj', inputs)
                label = np.einsum('ijk->ikj', label)
                
                inputs = torch.from_numpy(inputs)
                label = torch.from_numpy(label)
                
                # print("input shape is->>>>>>>>>>", inputs.shape)
                
                inputs = Variable(inputs)
                label = Variable(label)
                # dataset_len = Variable(dataset_len)
                
                inputs = inputs.float().cuda()
                label = label.float().cuda()
                # dataset_len = dataset_len.float().cuda()
                
                # print("dataset_len is ->>>>>>>>>>>",dataset_len)
                
                optim.zero_grad()
                
                pred = net(inputs)
                # label = label[:,:,0]
                # print('the label is------------------->',label.shape)
                loss = loss_function(pred, label)
                
                loss.backward()
                optim.step()
                epoch_loss += loss.item()
                epoch_len = dataset_len // train_loader.batch_size
                print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
                # if step==150:
                    # torch.save(net.state_dict(), opt.output_file+str(epoch+1)+'.pkl')
                    # np.save(opt.output_file+str(epoch+1)+'-losses.npy', np.array(losses))
            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)
            print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
            if opt.network is 'nnunet':
                update_learning_rate(net_scheduler, optim)

            if (epoch + 1) % val_interval == 0:
                net.eval()
                with torch.no_grad():
                    val_loss = 0.0
                    for v, val_data in enumerate(val_loader):
                        # get the inputs
                        inputs, label = val_data
                        dataset_len = 17516
                        
                        inputs = np.einsum('ijk->ikj', inputs)
                        label = np.einsum('ijk->ikj', label)
                        
                        inputs = torch.from_numpy(inputs)
                        label = torch.from_numpy(label)
                        
                        # print("input shape is->>>>>>>>>>", inputs.shape)
                        
                        inputs = Variable(inputs)
                        label = Variable(label)
                        # dataset_len = Variable(dataset_len)
                        
                        inputs = inputs.float().cuda()
                        label = label.float().cuda()
                        # dataset_len = dataset_len.float().cuda()
                
                        pred = net(inputs)
                        
                        loss = loss_function(pred, label)
                        val_loss += loss.data
                    print('val error: %.5f' % (val_loss / (v+1)))
                losses.append((epoch_loss / (i+1)))
            
            if (epoch+1) % 1 == 0:
                torch.save(net.state_dict(), opt.output_file+str(epoch+1)+'.pkl')
                np.save(opt.output_file+str(epoch+1)+'-losses.npy', np.array(losses))

            # Decay Learning Rate
            net_scheduler.step()
        
        print('Finished Training')


if __name__ == "__main__":
    main()
