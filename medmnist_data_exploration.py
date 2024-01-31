# Import Relevant Packages and Functions 

from tqdm import tqdm

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import torch
import sklearn
import os 
import random
from PIL import Image

from sklearn.discriminant_analysis import \
(LinearDiscriminantAnalysis as LDA, 
 QuadraticDiscriminantAnalysis as QDA)
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from data import Dataset
import torchvision.transforms as transforms

from medmnist.info import INFO, HOMEPAGE, DEFAULT_ROOT

try:
    from collections.abc import Sequence
except ImportError:
    from collections import Sequence


#Loads each of the MedMNIST datasets and gives each of them a flag.
class MedMNIST(Sequence):

    flag = ...

    def __init__(
        self,
        split,
        transform=None,
        target_transform=None,
        download=False,
        as_rgb=False,
        root=DEFAULT_ROOT,
        size=None,
        mmap_mode=None
    ):
        ''' 
        Args:

            split (str, required): 'train', 'val' or 'test'
            transform (callable, optional): data transformation
            target_transform (callable, optional): target transformation
            size (int, optional): The size of the returned images. If None, use MNIST-like 28. Default: None.
            mmap_mode (str, optional): If not None, read image arrays from the disk directly. This is useful to set `mmap_mode='r'` to save memory usage when the dataset is large (e.g., PathMNIST-224). Default: None.

        '''

        if (size is None) or (size == 28):
            self.size = 28
            self.size_flag = ""
        else:
            assert size in self.available_sizes
            self.size = size
            self.size_flag = f"_{size}"


        self.info = INFO[self.flag]

        if root is not None and os.path.exists(root):
            self.root = root
        else:
            raise RuntimeError("Failed to setup the default `root` directory. " +
                               "Please specify and create the `root` directory manually.")

        if download:
            self.download()

        if not os.path.exists(
                os.path.join(self.root, "{}.npz".format(self.flag))):
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        npz_file = np.load(
            os.path.join(self.root, f"{self.flag}{self.size_flag}.npz"),
            mmap_mode=mmap_mode,
        )

        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.as_rgb = as_rgb

        if self.split in ["train", "val", "test"]:
            self.imgs = npz_file[f"{self.split}_images"]
            self.labels = npz_file[f"{self.split}_labels"]
        else:
            raise ValueError

    def __len__(self):
        return self.imgs.shape[0]

    def __repr__(self):
        '''Adapted from torchvision.ss'''
        _repr_indent = 4
        head = f"Dataset {self.__class__.__name__} ({self.flag})"
        body = [f"Number of datapoints: {self.__len__()}"]
        body.append(f"Root location: {self.root}")
        body.append(f"Split: {self.split}")
        body.append(f"Task: {self.info['task']}")
        body.append(f"Number of channels: {self.info['n_channels']}")
        body.append(f"Meaning of labels: {self.info['label']}")
        body.append(f"Number of samples: {self.info['n_samples']}")
        body.append(f"Description: {self.info['description']}")
        body.append(f"License: {self.info['license']}")

        lines = [head] + [" " * _repr_indent + line for line in body]
        return '\n'.join(lines)

    def download(self):
        try:
            from torchvision.datasets.utils import download_url

            download_url(
                url=self.info[f"url{self.size_flag}"],
                root=self.root,
                filename=f"{self.flag}{self.size_flag}.npz",
                md5=self.info[f"MD5{self.size_flag}"],
            )
        except:
            raise RuntimeError('Something went wrong when downloading! ' +
                               'Go to the homepage to download manually. ' +
                               HOMEPAGE)

    @staticmethod
    def _collate_fn(data):
        xs = []
        ys = []
        for x, y in data:
            xs.append(np.array(x))
            ys.append(y)
        return np.array(xs), np.array(ys)


class MedMNIST2D(MedMNIST):
    available_sizes = [28, 64, 128, 224]

    def __getitem__(self, index):
        '''
        return: (without transform/target_transofrm)
            img: PIL.Image
            target: np.array of `L` (L=1 for single-label)
        '''
        img, target = self.imgs[index], self.labels[index].astype(int)
        img = Image.fromarray(img)

        if self.as_rgb:
            img = img.convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def save(self, folder, postfix="png", write_csv=True):

        from medmnist.utils import save2d

        save2d(
            imgs=self.imgs,
            labels=self.labels,
            img_folder=os.path.join(folder, f"{self.flag}{self.size_flag}"),
            split=self.split,
            postfix=postfix,
            csv_path=os.path.join(folder, f"{self.flag}{self.size_flag}.csv") if write_csv else None
        )

    def montage(self, length=20, replace=False, save_folder=None):
        from medmnist.utils import montage2d

        n_sel = length * length
        sel = np.random.choice(self.__len__(), size=n_sel, replace=replace)

        montage_img = montage2d(imgs=self.imgs,
                                n_channels=self.info['n_channels'],
                                sel=sel)

        if save_folder is not None:
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            montage_img.save(os.path.join(save_folder,
                                          f"{self.flag}{self.size_flag}_{self.split}_montage.jpg"))

        return montage_img


class MedMNIST3D(MedMNIST):
    available_sizes = [28, 64]

    def __getitem__(self, index):
        '''
        return: (without transform/target_transofrm)
            img: an array of 1x28x28x28 or 3x28x28x28 (if `as_RGB=True`), in [0,1]
            target: np.array of `L` (L=1 for single-label)
        '''
        img, target = self.imgs[index], self.labels[index].astype(int)

        img = np.stack([img/255.]*(3 if self.as_rgb else 1), axis=0)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def save(self, folder, postfix="gif", write_csv=True):
        from medmnist.utils import save3d

        assert postfix == "gif"

        save3d(
            imgs=self.imgs,
            labels=self.labels,
            img_folder=os.path.join(folder, f"{self.flag}{self.size_flag}"),
            split=self.split,
            postfix=postfix,
            csv_path=os.path.join(folder, f"{self.flag}{self.size_flag}.csv") if write_csv else None
        )

    def montage(self, length=20, replace=False, save_folder=None):
        assert self.info['n_channels'] == 1

        from medmnist.utils import montage3d, save_frames_as_gif

        n_sel = length * length
        sel = np.random.choice(self.__len__(), size=n_sel, replace=replace)

        montage_frames = montage3d(imgs=self.imgs,
                                   n_channels=self.info['n_channels'],
                                   sel=sel)

        if save_folder is not None:
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)

            save_frames_as_gif(montage_frames,
                               os.path.join(save_folder,
                                            f"{self.flag}{self.size_flag}_{self.split}_montage.gif"))

        return montage_frames


class PathMNIST(MedMNIST2D):
    flag = "pathmnist"


class OCTMNIST(MedMNIST2D):
    flag = "octmnist"


class PneumoniaMNIST(MedMNIST2D):
    flag = "pneumoniamnist"


class ChestMNIST(MedMNIST2D):
    flag = "chestmnist"


class DermaMNIST(MedMNIST2D):
    flag = "dermamnist"


class RetinaMNIST(MedMNIST2D):
    flag = "retinamnist"


class BreastMNIST(MedMNIST2D):
    flag = "breastmnist"


class BloodMNIST(MedMNIST2D):
    flag = "bloodmnist"


class TissueMNIST(MedMNIST2D):
    flag = "tissuemnist"


class OrganAMNIST(MedMNIST2D):
    flag = "organamnist"


class OrganCMNIST(MedMNIST2D):
    flag = "organcmnist"


class OrganSMNIST(MedMNIST2D):
    flag = "organsmnist"


class OrganMNIST3D(MedMNIST3D):
    flag = "organmnist3d"


class NoduleMNIST3D(MedMNIST3D):
    flag = "nodulemnist3d"


class AdrenalMNIST3D(MedMNIST3D):
    flag = "adrenalmnist3d"


class FractureMNIST3D(MedMNIST3D):
    flag = "fracturemnist3d"


class VesselMNIST3D(MedMNIST3D):
    flag = "vesselmnist3d"


class SynapseMNIST3D(MedMNIST3D):
    flag = "synapsemnist3d"


# backward-compatible
OrganMNISTAxial = OrganAMNIST
OrganMNISTCoronal = OrganCMNIST
OrganMNISTSagittal = OrganSMNIST


def get_loader(dataset, batch_size):
    total_size = len(dataset)
    print('Size', total_size)
    index_generator = shuffle_iterator(range(total_size))
    while True:
        data = []
        for _ in range(batch_size):
            idx = next(index_generator)
            data.append(dataset[idx])
        yield dataset._collate_fn(data)


def shuffle_iterator(iterator):
    # iterator should have limited size
    index = list(iterator)
    total_size = len(index)
    i = 0
    random.shuffle(index)
    while True:
        yield index[i]
        i += 1
        if i >= total_size:
            i = 0
            random.shuffle(index)

# Function to name the datasets
def dataset_namer(input_name, suffix):
    global string
    string = f"{input_name}_{suffix}"
    return string

datasets = {}

# Function to generate MedMNIST datasets
def medmnist_generator(data_flag, split):

    info = INFO[data_flag]
    task = info['task']
    n_channels = info['n_channels']
    n_classes = len(info['label'])

    DataClass = getattr(medmnist, info['python_class'])

    # Pre-processing
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
        ])
    
    name = dataset_namer(data_flag, split)
    
    global datasets

    if '3d' not in name: 
        value = DataClass(split=split,transform=data_transform,download=True) 
        entry = {name: value}
        datasets.update(entry)
        globals()[name] = value
    else:
        value = DataClass(split=split, download=True)
        entry = {name: value}
        datasets.update(entry)
        globals()[name] = value

# Specify Data Flags and Data Splits        
data_flag = ('pathmnist','dermamnist','breastmnist','nodulemnist3d')
split = ('train','test','val')

# For Loops to Generate Data
for i in range(len(data_flag)):
    for j in range(len(split)): 
        medmnist_generator(data_flag[i], split[j])

# Show information for PathMNIST
print(pathmnist_train)

# Show informatuon for DermaMNIST
print(dermamnist_train)

# Show information for BreastMNIST
print(breastmnist_train)

# Show information for NoduleMNIST3D
print(nodulemnist3d_test)

# Display 7x7 grid (49 samples) of BreastMNIST images
breastmnist_train.montage(length=7)

# Visualise one layer of a 6x6 Nodule images 
frames = nodulemnist3d_test.montage(length=6)
frames[10] #We do this to visualise one layer of the 3D image

# Pre-processing quantities needed
NUM_EPOCHS = 3
BATCH_SIZE = 128
lr = 0.001

# Function to transform data into dataloader form for deep learning
def data_loader(name, batch_size):
    name = dataset_namer(name, "loader")
    if 'train' in name:
        globals()[name] = data.DataLoader(dataset = name, batch_size = BATCH_SIZE, shuffle = True)
    else: 
        globals()[name] = data.DataLoader(dataset = name, batch_size = BATCH_SIZE, shuffle = False)

# Run function over all our datasets
for key in datasets.keys():
    data_loader(key, BATCH_SIZE)

# Turns the variable name into a string
def variable_name(variable):
    for name, object in chain(globals().items(), locals().items()):
        if object is variable:
            return name
    return None # didn't find anything...

features_and_labels = {}

# Function to extract features and labels from datasets
def features_labels(key, value):   

    X = value.imgs
    X = X.reshape(X.shape[0], -1)
    X = torch.from_numpy(X)
    
    y = value.labels
    y = np.ravel(y)
    y = torch.from_numpy(y)
    
    features = dataset_namer(key, "features")
    labels = dataset_namer(key, "labels")
    
    global features_and_labels
    entry = {X : y}
    features_and_labels.update(entry)
    
    globals()[features] = X
    globals()[labels] = y

# For loop to extract features and labels for all datasets in the dictionary  
for key, value in datasets.items():
    features_labels(key, value)

# Naive Bayes Model Fitting  
naive_bayes = GaussianNB()
naive_bayes.fit(breastmnist_train_features, breastmnist_train_labels)

# Quadratic Disciminant Analysis Model Fitting 
qda = QDA()
qda.fit(breastmnist_train_features, breastmnist_train_labels)

# Linear Discriminant Analysis Model Fitting
lda = LDA()
lda.fit(breastmnist_train_features, breastmnist_train_labels).transform(breastmnist_train_features)

# Naive Bayes Accuracy Score
naive_bayes.score(breastmnist_test_features, breastmnist_test_labels)

# QDA Accuracy Score
qda.score(breastmnist_test_features, breastmnist_test_labels) 

# LDA Accuracy Score
lda.score(breastmnist_test_features, breastmnist_test_labels)