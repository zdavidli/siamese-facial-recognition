import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F


import matplotlib.pyplot as plt
import numpy as np

import torch.nn as nn
import torch.optim as optim
import cv2

from LFWDataset import LFWDataset
from SiameseNet import SiameseNet


def show(img, filename=None, save=False):
    npimg = img.numpy()
    plt.axis("off")
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
    plt.show()
    if save and filename is not None:
        plt.savefig(filename)

def show_plot(iteration,loss):
    plt.plot(iteration,loss)
    plt.show()
    

def p1a():
    trainset = LFWDataset(train=True,
                      transform=transforms.Compose([transforms.Scale((128,128)),
                                                                      transforms.ToTensor()
                                                                      ]))
    trainloader = DataLoader(trainset, batch_size=1, shuffle=True, num_workers=2)

    testset = LFWDataset(test=True,
                         transform=transforms.Compose([transforms.Scale((128, 128)),
                                                                          transforms.ToTensor()
                                                                          ]))
    testloader = DataLoader(testset, batch_size=1, shuffle=True, num_workers=2)
    
    net = SiameseNet().cuda()
    criterion = nn.CrossEntropyLoss()
    learning_rate = 1e-6
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    
    counter = []
    loss_history = [] 
    iteration_number= 0
    epochs = 2
    
    for epoch in range(epochs):
        for i, data in enumerate(trainloader,0):
            img0, img1 , label = data 

    #         print(img0)
            img0 = Variable(img0).cuda()
            img1 = Variable(img1).cuda()
            label = Variable(label).cuda()
    #         img0, img1 , label = Variable(img0).cuda(), Variable(img1).cuda() , Variable(label).cuda()

            output = net(img0,img1)

    #         print(output)

            optimizer.zero_grad()
            loss = criterion(output, label)
    #         print(loss)
            loss.backward()
            optimizer.step()

            if i % 10 == 0 :
                print("Epoch number {}\n Current loss {}\n".format(epoch,loss.data[0]))
                iteration_number += 10
                counter.append(iteration_number)
                loss_history.append(loss.data[0])

    show_plot(counter,loss_history)

    
    
if __name__ == "__main__":
    p1a()