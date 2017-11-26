import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from skimage import transform as tf
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np
import cv2
import argparse

from LFWDataset import LFWDataset
from SiameseNet import SiameseNet
from ContrastiveLoss import ContrastiveLoss
from augmentation import augmentation

class Config():
    train_batch_size = 64
    train_number_epochs = 100
    margin = 10

def show_plot(iteration,loss, filename='loss.png', save=False):
    plt.figure(figsize=(10,4))
    plt.plot(iteration,loss)
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
#     plt.show()
    if save:
        plt.savefig(filename)


def train(aug, savemodel=False, model="model"):

    net = SiameseNet(p1b=True).cuda()

    if aug:
        trainset = LFWDataset(train=True,
                          transform=transforms.Compose([augmentation, transforms.Scale((128,128)),
                                                          transforms.ToTensor()
                                                          ]))
    else:
        trainset = LFWDataset(train=True,
                          transform=transforms.Compose([transforms.Scale((128,128)),
                                                          transforms.ToTensor()
                                                          ]))
    trainloader = DataLoader(trainset, batch_size=Config.train_batch_size, shuffle=True, num_workers=0)

    criterion = ContrastiveLoss(margin=Config.margin)
    learning_rate = 1e-6
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    counter = []
    loss_history = []
    iteration_number= 0

    for epoch in range(Config.train_number_epochs):
        for i, data in enumerate(trainloader,0):
            img0, img1 , label = data
            img0, img1 , label = Variable(img0).cuda(), Variable(img1).cuda() , Variable(label).cuda()
            output1, output2 = net(img0,img1)
            label = label.unsqueeze(1).float()

            optimizer.zero_grad()
            loss = criterion(output1, output2, label)
            loss.backward()
            optimizer.step()
            if i %10 == 0 :
                print("Epoch number {}\n Current loss {}\n".format(epoch,loss.data[0]))
                iteration_number +=10
                counter.append(iteration_number)
                loss_history.append(loss.data[0])
    #to see loss
    show_plot(counter,loss_history, save=True)

    if savemodel:
        torch.save(net.state_dict(), model)
        print("Model saved as: " + model)

def test(loadmodel=False, model="model"):

    net = SiameseNet(p1b=True).cuda()

    trainset = LFWDataset(train=True,
                      transform=transforms.Compose([transforms.Scale((128,128)),
                                                      transforms.ToTensor()
                                                      ]))
    trainloader = DataLoader(trainset, batch_size=Config.train_batch_size, shuffle=True, num_workers=2)
    testset = LFWDataset(test=True,
                     transform=transforms.Compose([transforms.Scale((128, 128)),
                                                      transforms.ToTensor()
                                                      ]))
    testloader = DataLoader(testset, batch_size=Config.train_batch_size, shuffle=True, num_workers=2)

    if loadmodel:
        net.load_state_dict(torch.load(model))
        print("Loaded model: " + model)

    #Accuracy on Train Set
    trainright=trainwrong=0.
    for i, data in enumerate(trainloader,0):
        img0, img1, label = data
        img0, img1, label = Variable(img0).cuda(), Variable(img1).cuda(), Variable(label).cuda()

        output1, output2 = net(img0,img1)
        dist = F.pairwise_distance(output1, output2)
        for x,y in zip(dist, label):
            if (x.data[0]>=Config.margin and y.data[0]==1) or (x.data[0]<Config.margin and y.data[0]==0):
                trainright+=1
            else:
                trainwrong+=1

    trainacc = trainright/(trainright+trainwrong)
    print("Accuracy on train set: {:.2f}".format(trainacc))

    #Accuracy on Test Set
    testright=testwrong=0.
    for i, data in enumerate(testloader,0):
        img0, img1, label = data
        img0, img1, label = Variable(img0).cuda(), Variable(img1).cuda(), Variable(label).cuda()

        output1, output2 = net(img0,img1)
        dist = F.pairwise_distance(output1, output2)
        for x,y in zip(dist, label):
            if (x.data[0]>=11 and y.data[0]==1) or (x.data[0]<11 and y.data[0]==0):
                testright+=1
            else:
                testwrong+=1

    testacc = testright/(testright+testwrong)
    print("Accuracy on test set: {:.2f}".format(testacc))


def p1b():
    parser = argparse.ArgumentParser(description='Process loading or saving.')
    parser.add_argument('--aug', dest='aug', action='store_true')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--load', dest='load', action='store',
                        type=str, help='File from which to load model')
    group.add_argument('--save', dest='save', action='store',
                        type=str, help='File to save model to')

    args = parser.parse_args()

    if args.save:
        print("Training...")
        train(args.aug, savemodel=True, model=args.save)
    if args.load:
        print("Testing...")
        test(loadmodel=True, model=args.load)


if __name__ == "__main__":
    p1b()
