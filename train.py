import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from torchvision import datasets, transforms

import copy
import random
import argparse
import pickle
from myMLP import *
from plot import *


def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--optim", default="BGD", type=str, help="SGD(Stochastic GD)/BGD(Mini Batch GD)")
    parser.add_argument("--param_init", default="norm", type=str, help="norm/kaiming")
    parser.add_argument("--lr_scheduler", default="const", type=str, help="const/multistep/exp")
    parser.add_argument("--more_layers", action='store_true', default=False)
    parser.add_argument("--less_layers", action='store_true', default=False)
    parser.add_argument("--reg", default="None", type=str, help="None/l1/l2")
    return parser


def train(epochs, batch_size, optim, param_init, lr_scheduler, reg, more_layers, less_layers):
    train_dataset = datasets.MNIST(root='../data/', train=True, download=True, transform=transforms.ToTensor())
    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    dev_dataset = datasets.MNIST(root='../data/', train=False, download=True, transform=transforms.ToTensor())
    dev_loader = Data.DataLoader(dataset=dev_dataset, batch_size=batch_size, shuffle=False)

    net = Net(batch_size=batch_size, input_dim=784, optim=optim, param_init=param_init, lr_scheduler=lr_scheduler, reg=reg)
    if not less_layers and not more_layers:
        net.addLinear(784, 512, "ReLU")
        net.addLinear(512, 128, "ReLU")
        net.addLinear(128, 10, "None")
    elif less_layers and not more_layers:
        net.addLinear(784, 128, "ReLU")
        net.addLinear(128, 10, "None")
    elif not less_layers and more_layers:
        net.addLinear(784, 512, "ReLU")
        net.addLinear(512, 128, "ReLU")
        net.addLinear(128, 64, "ReLU")
        net.addLinear(64, 10, "None")
    else:
        print("Error: Label less_layers and more_layers cannot be tagged True at the same time!")
        return

    net.addSoftmax()

    CEloss = nn.CrossEntropyLoss()
    train_epoch_loss = []
    train_loss = []
    train_acc = []
    dev_epoch_loss = []
    dev_loss = []
    dev_acc = []

    for epoch in range(epochs):
        train_y_pred = []
        train_y_true = []
        train_batch_loss = []
        dev_batch_loss = []
        
        print("---------Training--------")
        for step, (train_x, train_y) in enumerate(train_loader):
            train_x = np.squeeze(train_x.numpy()).reshape(-1, 784)
            batch_size = train_x.shape[0]
            y_pred = net.forward(train_x)
            loss = CEloss(torch.tensor(y_pred), train_y).numpy()
            if net.reg == 'l2':
                for layer in net.layers:
                    loss += 0.1 * np.sum(np.square(layer.w)) / (2 * batch_size)
            elif net.reg == 'l1':
                for layer in net.layers:
                    loss += 0.01 * np.sum(np.abs(layer.w)) / batch_size
            train_y = train_y.numpy()
            y_true = np.eye(10)[train_y]
            net.backward(y_true, epoch)
            
            pred_idx = np.argmax(y_pred, axis=1)
            train_y_pred.extend(pred_idx.tolist())
            train_y_true.extend(train_y.tolist())
            train_batch_loss.append(loss)

            if step % 10 == 0:
                print("Epoch %d Batch %d: Loss %.4f" % (epoch, step, loss))
        
        train_epoch_loss.append(np.mean(np.array(train_batch_loss)))
        train_loss.extend(train_batch_loss)
        train_y_pred = np.array(train_y_pred)
        train_y_true = np.array(train_y_true)
        acc = 100.0 * np.mean(np.array(train_y_pred == train_y_true))
        train_acc.append(acc)
        print("Epoch %d Training Loss: %.4f Acc %.4f " % (epoch, np.mean(np.array(train_batch_loss)), acc))

        dev_y_pred = []
        dev_y_true = []
        for step, (dev_x, dev_y) in enumerate(dev_loader):
            dev_x = np.squeeze(dev_x.numpy()).reshape(-1, 784)
            y_pred = net.forward(dev_x)
            loss = CEloss(torch.tensor(y_pred), dev_y)
            dev_y = dev_y.numpy()

            pred_idx = np.argmax(y_pred, axis=1)
            dev_y_pred.extend(pred_idx.tolist())
            dev_y_true.extend(dev_y.tolist())
            dev_batch_loss.append(loss.numpy())

        dev_epoch_loss.append(np.mean(np.array(dev_batch_loss)))
        dev_loss.extend(dev_batch_loss)
        dev_y_pred = np.array(dev_y_pred)
        dev_y_true = np.array(dev_y_true)
        acc = 100.0 * np.mean(np.array(dev_y_pred == dev_y_true))
        dev_acc.append(acc)
        print("Epoch %d Validating Loss: %.4f Acc %.4f" % (epoch, np.mean(np.array(dev_batch_loss)), acc))

    save_res = {
        'train_epoch_loss': train_epoch_loss,
        'dev_epoch_loss': dev_epoch_loss,
        'train_loss': train_loss,
        'dev_loss': dev_loss,
        'train_acc': train_acc,
        'dev_acc': dev_acc 
    }
    
    fn = args.optim + '_' + args.param_init + "_" + args.lr_scheduler + "_" +args.reg
    if less_layers:
        fn += '_lesslayers'
    elif more_layers:
        fn += '_morelayers'
    res_fn = fn + '.pkl'
    fig_fn = fn + '.png'
    f = open(res_fn, 'wb')
    pickle.dump(save_res, f)
    f.close()
    plotCurve(res_fn, fig_fn)


if __name__ == '__main__':
    parser = get_argparse()
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        optim=args.optim,
        param_init=args.param_init,
        lr_scheduler=args.lr_scheduler,
        reg=args.reg,
        more_layers=args.more_layers,
        less_layers=args.less_layers
    )


