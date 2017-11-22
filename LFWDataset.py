from torch.utils.data import Dataset, DataLoader
import cv2
import os

class LFWDataset(Dataset):
    """Faces in the wild dataset."""

    def __init__(self, root_dir='lfw/', train=False, test=False, transform=None):
        self.train = train
        self.test = test
        self.root_dir = root_dir
        self.files = []
        self.transform = transform
        
        if (self.train and self.test) or not (self.train or self.test):
            raise ValueError('Exactly one of train and test must be set.')
        
        dataset = set()
        if self.train:
            filename='train.txt'
        else:
            filename='test.txt'

        with open(filename) as f:
            for line in f:
                line = line.split()
                dataset.update({line[0], line[1]})
                    
        for dirpath, subdirs, walkfiles in os.walk(self.root_dir):
            for x in walkfiles:
                if os.path.join(dirpath.replace('lfw/', ''), x) in dataset:
                    self.files.append(os.path.join(dirpath, x))
        #print(self.files)
    
    def __len__(self):
        return len(self.files)
#         return sum([len(files) for r, d, files in os.walk(self.root_dir)])
#         https://stackoverflow.com/questions/16910330/return-number-of-files-in-directory-and-subdirectory

    def __getitem__(self, idx):
        img_name = self.files[idx]
        image = cv2.imread(img_name)
        if self.transform:
            image = self.transform(image)
            
        return image