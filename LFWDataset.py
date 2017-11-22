from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
class LFWDataset(Dataset):
    
    """Faces in the wild dataset."""

    def __init__(self, root_dir='lfw/', train=False, test=False, transform=None):
        self.train = train
        self.test = test
        self.root_dir = root_dir
        self.files = []
        self.labels = []
        self.transform = transform
        
        if (self.train and self.test) or not (self.train or self.test):
            raise ValueError('Exactly one of train and test must be set.')
        
        """
        Getting Train/Test splits
        """
        dataset = set()
        if self.train:
            filename='train.txt'
        else:
            filename='test.txt'

        with open(filename) as f:
            for line in f:
                line = line.split()
                self.files.append(line[:2])
                self.labels.append(int(line[2]))
                
        #print(self.files)
    
    def __len__(self):
        return len(self.files)
#         return sum([len(files) for r, d, files in os.walk(self.root_dir)])
#         https://stackoverflow.com/questions/16910330/return-number-of-files-in-directory-and-subdirectory

    def __getitem__(self, idx):
        im_names = self.files[idx]
        im1 = Image.open(os.path.join(self.root_dir, im_names[0]))
        im2 = Image.open(os.path.join(self.root_dir, im_names[1]))
        label = self.labels[idx]
        
        if self.transform:
            im1 = self.transform(im1)
            im2 = self.transform(im2)
            
        return im1, im2, label