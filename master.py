# Master Script for statml_medmnistplus

## Package Import
### Python Packages
import sys
import warnings
import random

### User Created Modules
from data_loader import data_loader
from data_manip import dataset_namer
from medmnist_setup import medmnist_generator
from analysis import image_analysis
from sklearn.exceptions import FitFailedWarning
from file_path import file_path
from images import *

## Redirect Python to Modules Area
sys.path.append(file_path+'/Modules')

## Ignore Warnings
warnings.filterwarnings("ignore", category=FitFailedWarning)
warnings.filterwarnings("ignore", category=UserWarning)

## Take shell inputs
dataset = str(sys.argv[1])
resolution = int(sys.argv[2])
method = str(sys.argv[3])
if len(sys.argv) > 4:
    stage = str(sys.argv[4])
else:
    stage = ''
if len(sys.argv) > 5:
    colour_change = str(sys.argv[5])
else:
    colour_change = 'no'

## Set random seed
random_seed = 28    
random.seed(random_seed)

## Name the dataset
name = dataset_namer(dataset, resolution)

## Load the data
data = data_loader(dataset, resolution)

## Run the analysis script
image_analysis(name, data, method, stage, random_seed, colour_change)    



