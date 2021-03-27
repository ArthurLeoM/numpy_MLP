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

def train(epochs, batch_size, optim, param_init, lr_scheduler):
    train_dataset = datasets.MNIST(root='./data/', train=True, download=True, transform=transforms.ToTensor())
    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    dev_dataset = datasets.MNIST(root='./data/', train=False, download=True, transform=transforms.ToTensor())
    dev_loader = Data.DataLoader(dataset=dev_dataset, batch_size=batch_size, shuffle=False)

    net = Net(batch_size=batch_size, input_dim=784, optim=optim, param_init=param_init, lr_scheduler=lr_scheduler)
    net.addLinear(784, 512, "ReLU")
    net.addLinear(512, 128, "ReLU")
    net.addLinear(128, 10, "None")
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
            y_pred = net.forward(train_x)
            loss = CEloss(torch.tensor(y_pred), train_y)
            train_y = train_y.numpy()
            y_true = np.eye(10)[train_y]
            net.backward(y_true, epoch+1)
            
            pred_idx = np.argmax(y_pred, axis=1)
            train_y_pred.extend(pred_idx.tolist())
            train_y_true.extend(train_y.tolist())
            train_batch_loss.append(loss.numpy())

            if step % 10 == 0:
                print("Epoch %d Batch %d: Loss %.4f" % (epoch, step, loss.numpy()))
        
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
    
    res_fn = './res/BGD_norm_const_baseline.pkl'
    fig_fn = './fig/BGD_norm_const_baseline.png'
    f = open(res_fn, 'wb')
    pickle.dump(save_res, f)
    f.close()
    plotCurve(res_fn, fig_fn)


if __name__ == '__main__':
    train(
        epochs=50,
        batch_size=256,
        optim="BGD",
        param_init="norm",
        lr_scheduler="const"
    )


