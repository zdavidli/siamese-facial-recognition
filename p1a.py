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
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.savefig('output.png')

def train(net):
    trainset = LFWDataset(train=True,
                      transform=transforms.Compose([transforms.Scale((128,128)),
                                                                      transforms.ToTensor()
                                                                      ]))
    trainloader = DataLoader(trainset, batch_size=8, shuffle=True, num_workers=0)

    criterion = nn.CrossEntropyLoss()
    learning_rate = 1e-6
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    
    counter = []
    loss_history = [] 
    iteration_number= 0
    epochs = 10
    
    for epoch in range(epochs):
        for i, data in enumerate(trainloader,0):
            img0, img1 , label = data 
            img0, img1 , label = Variable(img0).cuda(), Variable(img1).cuda() , Variable(label).cuda()

            output = net(img0,img1)
            output = torch.cat((output, 1-output), 1)

            optimizer.zero_grad()
            loss = criterion.forward(output, label)
            loss.backward()
            optimizer.step()

            if i % 10 == 0 :
                print("Epoch number {}\n {}\n Current loss {}\n".format(epoch,iteration_number,loss.data[0]))
                iteration_number += 10
                counter.append(iteration_number)
                loss_history.append(loss.data[0])

    show_plot(counter,loss_history)

def test(net):
    testset = LFWDataset(test=True,
                         transform=transforms.Compose([transforms.Scale((128, 128)),
                                                                          transforms.ToTensor()
                                                                          ]))
    testloader = DataLoader(testset, batch_size=8, shuffle=True, num_workers=0)
    
    
    
def p1a():
    net = SiameseNet().cuda()
    train(net)
    test(net)
   
    

    
    
if __name__ == "__main__":
    p1a()