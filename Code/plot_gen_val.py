# Python code to read in training data from a CSV file and plot the information for LDA and Logistic Regression

## Import Packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
from file_path import file_path
import glob
import os

## Assign variables based on terminal inputs
data = str(sys.argv[1])
resolution = int(sys.argv[2])
method = str(sys.argv[3])
colour = str(sys.argv[4])

# Organise methods and colours
if method == "lda":
    method = method.upper()
elif method == "lr":
    method = "Logistic Regression"
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

# Rectify dataset name for appropriate output presentation
name = data.split("mnist")[0]
# Convert the first part to title case
name = name.title()+"MNIST"

folder_path = file_path+f"Results/{data}_{resolution}/{method}/{col}/"
pattern = f"model_training_{colour}.csv"

# Use glob to find the file paths that contain the pattern
matching_files = [file for file in glob.glob(os.path.join(folder_path, "*.csv")) if pattern in file]

# Check if a file was found
if len(matching_files) == 1:
    # Read the CSV file into a DataFrame
    csv = pd.read_csv(matching_files[0])
    # Sort the DataFrame
    csv = csv.sort_values(by=csv.columns[6], ascending=False)
else:
    print(f"No matching CSV file found for pattern '{pattern}'")
    # You might want to handle this case, for example, by exiting the program or providing default data.


# Sort values
csv = csv.sort_values(by=csv.columns[6], ascending=False)


# Create graph showing percentage of features corresponding to the metrics and methods
scoring = ['Accuracy','F1','Log Loss','Prediction']
fs_pct = range(100,55,-5)

if method == "LDA":
    
        # Create a single figure with four subplots
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'LDA Model Tuning for the {name}_{resolution} validation data', fontsize=20)

        for index, score in enumerate(scoring):
            i = int(index + 8)
            baseline_data = []
            full_mod_data = []
            for j in range(len(csv.iloc[:,0])):
                if 'baseline' in csv.iloc[j,0]:
                    baseline_data += [csv.iloc[j, i]]
                else:
                    full_mod_data += [csv.iloc[j, i]]
                            
            # Determine subplot position
            row = (int(i)-8) // 2
            col = (int(i)-8) % 2
            ax = axs[row, col]

            ax.scatter(fs_pct[1:], baseline_data, zorder=2)
            ax.plot(fs_pct[1:], baseline_data, zorder=1, label="Baseline", linestyle='-')
            ax.scatter(fs_pct[0], full_mod_data, zorder=2, c='black', marker='D', s=50, label="Full Model", linestyle='-')
            ax.invert_xaxis()
            ax.set_xlabel('Percentage of Original Features', fontsize = 14)
            ax.set_ylabel(f'{score} Score', fontsize = 14)
            ax.set_title(f'{score} Score ', fontsize = 18)
            ax.legend(bbox_to_anchor=(0.3, -0.15), loc='upper left', fontsize = 14)
            ax.tick_params(axis='both', which='major', labelsize=12)  
            ax.tick_params(axis='both', which='minor', labelsize=10)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(folder_path+f'{name}_{resolution}_LDA_graph_{colour}.jpeg')  
    
elif method == "Logistic Regression":
    # Create a single figure with four subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 14))
    fig.suptitle(f'Logistic Regression Model Tuning for the {name}_{resolution} validation data', fontsize=20)
    
    # Generate a list of linestyles for each subplot
    linestyles = [['-', '--', '-.', ':']] * 5  # This generates the same set of linestyles for each subplot
    
    for index, score in enumerate(scoring):
        i = int(index + 9)
        LBFGS_data = []
        full_mod_data = []
        SAG_data = []
        L2_SAGA_data = []
        L1_data = []
        LASSO_pct = []
        for j in range(len(csv.iloc[:,0])):
            if 'full' in csv.iloc[j,0]:
                full_mod_data += [csv.iloc[j, i]]
            elif 'LBFGS' in csv.iloc[j,0]:
                LBFGS_data += [csv.iloc[j, i]]
            elif 'SAG' in csv.iloc[j,0] and 'SAGA' not in csv.iloc[j,0]:
                SAG_data += [csv.iloc[j, i]]
            elif 'SAGA' in csv.iloc[j,0] and 'LASSO' not in csv.iloc[j,0]:
                L2_SAGA_data += [csv.iloc[j, i]]
            else:
                L1_data += [csv.iloc[j, i]]
                LASSO_pct += [int(csv.iloc[j, 7])*100]
                        
        # Determine subplot position
        row = (int(i)-9) // 2
        col = (int(i)-9) % 2
        ax = axs[row, col]

        # Plot each line with a different linestyle
        ax.scatter(fs_pct[1:], LBFGS_data, zorder = 2)
        ax.plot(fs_pct[1:], LBFGS_data, label = 'Baseline', zorder = 1, linestyle=linestyles[index][0])
        ax.scatter(fs_pct, SAG_data, zorder = 2)
        ax.plot(fs_pct, SAG_data, label = 'Ridge with SAG solver', zorder = 1, linestyle=linestyles[index][1])
        ax.scatter(fs_pct, L2_SAGA_data, zorder = 2)
        ax.plot(fs_pct, L2_SAGA_data, label = 'Ridge with SAGA solver', zorder = 1, linestyle=linestyles[index][2])
        ax.scatter(LASSO_pct, L1_data, zorder = 2, label = 'LASSO')
        ax.scatter(fs_pct[0], full_mod_data[0], zorder = 2, c='black', marker='D', s=50, label = "Full Model", linestyle=linestyles[index][3])
        ax.invert_xaxis()
        ax.set_xlabel('Percentage of Original Features', fontsize = 14)
        ax.set_ylabel(f'{score} Score', fontsize = 14)
        ax.set_title(f'{score} Score', fontsize = 18)
        ax.tick_params(axis='both', which='major', labelsize=12)  
        ax.tick_params(axis='both', which='minor', labelsize=10)
        ax.legend(bbox_to_anchor=(0.2, -0.20), loc='upper left', fontsize = 14)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(folder_path+f'{name}_{resolution}_LogReg_graph_{colour}.jpeg')