from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class SiameseNet(nn.Module):
    def __init__(self):
        super(SiameseNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5, stride=(1,1), padding=2)
        self.batch1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d((2,2), stride=(2,2))
        
        self.conv2 = nn.Conv2d(64, 128, 5, stride=(1,1), padding=2)
        self.batch2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d((2,2), stride=(2,2))

        self.conv3 = nn.Conv2d(128, 256, 3, stride=(1,1), padding=1)
        self.batch3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d((2,2), stride=(2,2))

        self.conv4 = nn.Conv2d(256, 512, 3, stride=(1,1), padding=1)
        self.batch4 = nn.BatchNorm2d(512)
        
        self.linear1 = nn.Linear(512 * 16 * 16, 1024)
        self.batch5 = nn.BatchNorm2d(1024)
        
        self.sigmoid = nn.Sigmoid()

    def forward_once(self, x):
        x = self.pool1(self.batch1(F.relu(self.conv1(x))))
        x = self.pool2(self.batch2(F.relu(self.conv2(x))))
        x = self.pool3(self.batch3(F.relu(self.conv3(x))))
        x = self.batch4(F.relu(self.conv4(x)))
        
        #Flatten
        x = x.view(-1, self.num_flat_features(x))
        x = self.batch4(F.relu(self.linear1(x)))
        return x
    
    def forward(self, input1, input2):
        f1 = self.forward_once(input1)
        f2 = self.forward_once(input2)
        f12 = np.concatenate((f1, f2))
        output = self.sigmoid(f12)
        return output
