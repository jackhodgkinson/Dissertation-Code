# Python code that reads a text file of training from a machine learning model and produces plots

## Import packages
import matplotlib.pyplot as plt
import pandas as pd
from file_path import file_path
import sys
import glob
import os

## Define variables based on terminal input
data = str(sys.argv[1])
resolution = int(sys.argv[2])
method = str(sys.argv[3])
colour = str(sys.argv[4])

## Organisation of methods and colours
if method == "xgb":
    method = "XGBoost"
elif method == "cnn":
    method = method.upper()
else:
    print("Method not supported - terminating.")
    sys.exit()
    
if colour == "rgb":
    col = colour.upper()
elif colour == "gray":
    col = "Grayscale"
else:
    print("Colour not available - terminating.")
    sys.exit()

# Format dataset name for output
name = data.split("mnist")[0]
name = name.title()+"MNIST"

folder_path = file_path+f"Results/{data}_{resolution}/{method}/{col}/"

# Get a list of all files in the specified folder
all_files = glob.glob(os.path.join(folder_path, "*"))

# Filter out TXT files
txt_files = [file for file in all_files if file.endswith(".txt")]

# If a text file exists then a plot is produced.
if len(txt_files) == 1:
    # Get the file name of the first TXT file found
    txt_file = txt_files[0]
    print("Found the text file:", txt_file)

    if method == "XGBoost":
        df = pd.read_csv(txt_file, sep='\s+', skiprows=6, skipfooter=13, header = None, engine='python')  
        min_loss = df[2].min()
        linestyles = ['-', '--', '-.', ':']  # Defining a list of linestyles
        plt.plot(df[0], df[1], label="Training", linestyle=linestyles[0])  # Plotting training data with linestyle 0
        plt.plot(df[0], df[2], label="Validation", linestyle=linestyles[1])  # Plotting validation data with linestyle 1
        plt.axhline(y=min_loss, color='r', linestyle=linestyles[2], label='Minimum Validation Log Loss')
        plt.xlabel('Number of Trees', fontsize=14)  
        plt.ylabel('Log Loss', fontsize=14)  
        plt.title(f'Log Loss of the training and validation datasets \n for the method of {method} on {name}_{resolution}', fontsize=18)
        plt.legend(fontsize=12)
        plt.savefig(folder_path+f"{data}_{resolution}_{method}_{colour}.png")

    elif method == "CNN":
        df = pd.read_csv(txt_file, sep='\s+', skiprows=6, skipfooter=14, header=None, engine='python')
        min_loss = df[4].min()
        linestyles = ['-', '--', '-.', ':']  # Defining a list of linestyles
        plt.plot(df[0], df[2], label="Training", linestyle=linestyles[0])  # Plotting training data with linestyle 0
        plt.plot(df[0], df[4], label="Validation", linestyle=linestyles[1])  # Plotting validation data with linestyle 1
        plt.axhline(y=min_loss, color='r', linestyle=linestyles[2], label='Minimum Validation Log Loss')  # Plotting minimum validation loss line with linestyle '--'
        plt.xlabel('Number of Epochs', fontsize=14)  # Set x-axis label
        plt.ylabel('Log Loss', fontsize=14)  # Set y-axis label
        plt.title(f'Log Loss of the training and validation datasets \n for the method of {method} on {name}_{resolution}', fontsize=18)
        plt.legend(fontsize=12)
        plt.savefig(folder_path+f"{data}_{resolution}_{method}_{colour}.png")

    else:
        print("Method not supported - terminating.")