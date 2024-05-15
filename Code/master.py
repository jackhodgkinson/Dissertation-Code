# Master Script for statml_medmnistplus

## Package Import
### Python Packages
import random
from sklearn.exceptions import FitFailedWarning
import sys
import warnings

### User Created Modules
#### Import file_path module in current folder
from file_path import file_path

#### Redirect Python to Modules folder to upload modules
sys.path.append(file_path+'/Modules')

#### Modules
from data_manip import dataset_namer
from medmnist_setup import *
from analysis import image_analysis
from images import *

## Redirect Python to Results folder
sys.path.append(file_path+'/Results')

## Ignore Warnings
warnings.filterwarnings("ignore", category=FitFailedWarning)
warnings.filterwarnings("ignore", category=UserWarning)

## Take shell inputs
dataset = str(sys.argv[1])
resolution = int(sys.argv[2])
method = str(sys.argv[3])
if len(sys.argv) > 4 and str(sys.argv[4]) not in ['no','gray','grey','greyscale','grayscale']:
    stage = str(sys.argv[4])
elif len(sys.argv) > 4 and str(sys.argv[4]) in ['no','gray','grey','greyscale','grayscale']:
    stage = ''
    colour_change = str(sys.argv[4])
else:
    stage = ''
if len(sys.argv) > 5:
    colour_change = str(sys.argv[5])
else:
    try:
        colour_change
    except NameError:
        colour_change = 'no'

## Set random seed
random_seed = 28    
random.seed(random_seed)

## Name the dataset
name = dataset_namer(dataset, resolution)

## Load the data
data = medmnist_loader(dataset, resolution)

## Run the analysis script
image_analysis(name, data, method, stage, random_seed, colour_change)


