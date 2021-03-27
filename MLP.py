import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from torchvision import datasets, transforms

import matplotlib.pyplot as plt

import copy
import random
import numpy as np
import argparse
import pickle

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class MLP_model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP_model, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()

        self.MLP = nn.Sequential(
            nn.Linear(self.input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        output = self.softmax(self.MLP(x))
        return output


def train(epochs=50, batch_size=256, lr=1e-2):
    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)  # numpy
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)  # cpu
    torch.cuda.manual_seed(RANDOM_SEED)  # gpu
    torch.backends.cudnn.deterministic = True  # cudnn

    train_dataset = datasets.MNIST(root='./data/', train=True, download=True, transform=transforms.ToTensor())
    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    dev_dataset = datasets.MNIST(root='./data/', train=False, download=True, transform=transforms.ToTensor())
    dev_loader = Data.DataLoader(dataset=dev_dataset, batch_size=batch_size, shuffle=False)

    model = MLP_model(input_dim=784, output_dim=10)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    CE_loss = nn.CrossEntropyLoss()

    epoch_loss = []
    epoch_dev_loss = []
    for epoch in range(epochs):
        batch_loss = []
        dev_loss = []
        total_y_pred = []
        total_y_truth = []
        model.train()

        for step, (train_x, train_y) in enumerate(train_loader):
            optimizer.zero_grad()
            train_x = torch.tensor(train_x).view(-1, 784).to(device)
            total_y_truth.extend(train_y.cpu().detach().numpy().tolist())
            train_y = torch.tensor(train_y).to(device)

            output = model(train_x).to(device)
            loss = CE_loss(output, train_y)
            _, pred_idx = torch.max(output, dim=1)
            total_y_pred.extend(pred_idx.cpu().detach().numpy().tolist())

            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 20)
            optimizer.step()
            batch_loss.append(loss.cpu().detach().numpy())
            if step % 10 == 0:
                print('Epoch %d Batch %d: Train Loss = %.4f' % (epoch, step, loss.cpu().detach().numpy()))
        
        epoch_loss.append(np.mean(np.array(batch_loss)))
        total_y_pred = np.array(total_y_pred)
        total_y_truth = np.array(total_y_truth)
        print(total_y_pred, total_y_truth)
        train_acc = sum([total_y_pred[i] == total_y_truth[i] for i in range(len(total_y_truth))]) / len(total_y_truth)
        print('Epoch %d Loss: %.4f, Acc: %.4f' % (epoch, epoch_loss[-1], train_acc))

        print('--------Validating--------')
        with torch.no_grad():
            model.eval()
            dev_y_pred = []
            dev_y_truth = []
            for step, (dev_x, dev_y) in enumerate(dev_loader):
                dev_x = torch.tensor(dev_x).view(-1, 784).to(device)
                dev_y_truth.extend(dev_y.cpu().detach().numpy().tolist())
                train_y = torch.tensor(train_y).to(device)

                output = model(dev_x).to(device)
                _, pred_idx = torch.max(output, dim=1)
                dev_y_pred.extend(pred_idx.cpu().detach().numpy().tolist())
                loss = CE_loss(output, dev_y)
                dev_loss.append(loss.cpu().detach().numpy())
                if step % 10 == 0:
                    print('Epoch %d Batch %d: Dev Loss = %.4f' % (epoch, step, np.mean(np.array(dev_loss))))
        
        epoch_dev_loss.append(np.mean(np.array(dev_loss)))
        dev_y_pred = np.array(dev_y_pred)
        dev_y_truth = np.array(dev_y_truth)
        print(dev_y_pred, dev_y_truth)
        dev_acc = sum([dev_y_pred[i] == dev_y_truth[i] for i in range(len(dev_y_truth))]) / len(dev_y_truth)
        
        print('Epoch %d Dev Loss: %.4f, Dev Acc: %.4f' % (epoch, epoch_dev_loss[-1], dev_acc))
        model.train()

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.plot(range(len(epoch_dev_loss)), epoch_dev_loss, c='red', label='dev')
    ax1.plot(range(len(epoch_loss)), epoch_loss, c='blue', label='train')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss')
    plt.legend(loc='best')
    plt.savefig('./loss.png')
    plt.show()

if __name__ == '__main__':
    train(epochs=50, batch_size=256, lr=0.05)
