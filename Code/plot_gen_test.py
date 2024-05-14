# Python file to generate plots based on the results of testing 

## Import Packages
import matplotlib.pyplot as plt
import numpy as np
from file_path import file_path

## Input data
scores_dict = {
    'Linear Discriminant Analysis': {
        'BreastMNIST': {
            'Grayscale': {
                784: {'Accuracy': 0.720, 'F1-Score': 0.796, 'Log-Loss': 9.240, 'Prediction': 0.082},
                4096: {'Accuracy': 0.769, 'F1-Score': 0.845, 'Log-Loss': 1.712, 'Prediction': 0.471},
                16384: {'Accuracy': 0.801, 'F1-Score': 0.865, 'Log-Loss': 0.868, 'Prediction': 0.960},
                50176: {'Accuracy': 0.795, 'F1-Score': 0.863, 'Log-Loss': 0.661, 'Prediction': 1.254}
            }
        },
        'DermaMNIST': {
            'RGB': {
                784: {'Accuracy': 0.604, 'F1-Score': 0.600, 'Log-Loss': 2.761, 'Prediction': 0.218},
                4096: {'Accuracy': 0.540, 'F1-Score': 0.553, 'Log-Loss': 10.570, 'Prediction': 0.052},
                16384: {'Accuracy': 0.625, 'F1-Score': 0.604, 'Log-Loss': 5.172, 'Prediction': 0.119},
                50176: {'Accuracy': 0.645, 'F1-Score': 0.614, 'Log-Loss': 3.788, 'Prediction': 0.166}
            },
            'Grayscale': {
                784: {'Accuracy': 0.627, 'F1-Score': 0.580, 'Log-Loss': 1.669, 'Prediction': 0.364},
                4096: {'Accuracy': 0.510, 'F1-Score': 0.527, 'Log-Loss': 6.909, 'Prediction': 0.075},
                16384: {'Accuracy': 0.545, 'F1-Score': 0.541, 'Log-Loss': 8.539, 'Prediction': 0.064},
                50176: {'Accuracy': 0.606, 'F1-Score': 0.576, 'Log-Loss': 5.322, 'Prediction': 0.111}
            }
        },
        'PathMNIST': {
            'RGB': {
                784: {'Accuracy': 0.579, 'F1-Score': 0.563, 'Log-Loss': 1.185, 'Prediction': 0.482},
                4096: {'Accuracy': 0.512, 'F1-Score': 0.501, 'Log-Loss': 1.605, 'Prediction': 0.316},
            },
            'Grayscale': {
                784: {'Accuracy': 0.414, 'F1-Score': 0.371, 'Log-Loss': 1.644, 'Prediction': 0.239},
                4096: {'Accuracy': 0.401, 'F1-Score': 0.374, 'Log-Loss': 1.784, 'Prediction': 0.217},
                16384: {'Accuracy': 0.356, 'F1-Score': 0.336, 'Log-Loss': 2.560, 'Prediction': 0.135},
            }
        },
    },
    'Logistic Regression': {
        'BreastMNIST': {
            'Grayscale': {
                784: {'Accuracy': 0.776, 'F1-Score': 0.851, 'Log-Loss': 0.543, 'Prediction': 1.497},
                4096: {'Accuracy': 0.801, 'F1-Score': 0.869, 'Log-Loss': 0.552, 'Prediction': 1.513},
                16384: {'Accuracy': 0.827, 'F1-Score': 0.886, 'Log-Loss': 0.562, 'Prediction': 1.525},
                50176: {'Accuracy': 0.821, 'F1-Score': 0.881, 'Log-Loss': 0.560, 'Prediction': 1.520}
            }
        },
        'DermaMNIST': {
            'RGB': {
                784: {'Accuracy': 0.690, 'F1-Score': 0.636, 'Log-Loss': 0.880, 'Prediction': 0.754},
                4096: {'Accuracy': 0.690, 'F1-Score': 0.643, 'Log-Loss': 0.885, 'Prediction': 0.754},
                16384: {'Accuracy': 0.694, 'F1-Score': 0.647, 'Log-Loss': 0.885, 'Prediction': 0.758},
                50176: {'Accuracy': 0.694, 'F1-Score': 0.647, 'Log-Loss': 0.884, 'Prediction': 0.758}
            },
            'Grayscale': {
                784: {'Accuracy': 0.669, 'F1-Score': 0.580, 'Log-Loss': 1.041, 'Prediction': 0.600},
                4096: {'Accuracy': 0.670, 'F1-Score': 0.585, 'Log-Loss': 1.037, 'Prediction': 0.605},
                16384: {'Accuracy': 0.674, 'F1-Score': 0.593, 'Log-Loss': 1.037, 'Prediction': 0.611},
                50176: {'Accuracy': 0.676, 'F1-Score': 0.593, 'Log-Loss': 1.034, 'Prediction': 0.614}
            }
        },
        'PathMNIST': {
            'RGB': {
                784: {'Accuracy': 0.422, 'F1-Score': 0.386, 'Log-Loss': 1.701, 'Prediction': 0.237},
                4096: {'Accuracy': 0.408, 'F1-Score': 0.383, 'Log-Loss': 1.806, 'Prediction': 0.219},
            },
            'Grayscale': {
                784: {'Accuracy': 0.159, 'F1-Score': 0.137, 'Log-Loss': 2.200, 'Prediction': 0.067},
                4096: {'Accuracy': 0.135, 'F1-Score': 0.127, 'Log-Loss': 2.310, 'Prediction': 0.057},
                16384: {'Accuracy': 0.134, 'F1-Score': 0.130, 'Log-Loss': 2.471, 'Prediction': 0.053},
            }
        },
    },
    'Convolutional Neural Network': {
        'BreastMNIST': {
            'Grayscale': {
                784: {'Accuracy': 0.878, 'F1-Score': 0.920, 'Log-Loss': 0.657, 'Prediction': 1.369},
                4096: {'Accuracy': 0.872, 'F1-Score': 0.912, 'Log-Loss': 0.638, 'Prediction': 1.398},
                16384: {'Accuracy': 0.821, 'F1-Score': 0.886, 'Log-Loss': 0.964, 'Prediction': 0.885},
                50176: {'Accuracy': 0.833, 'F1-Score': 0.887, 'Log-Loss': 0.569, 'Prediction': 1.512}
            }
        },
        'DermaMNIST': {
            'RGB': {
                784: {'Accuracy': 0.728, 'F1-Score': 0.699, 'Log-Loss': 0.886, 'Prediction': 0.806},
                4096: {'Accuracy': 0.710, 'F1-Score': 0.667, 'Log-Loss': 1.267, 'Prediction': 0.854},
                16384: {'Accuracy': 0.726, 'F1-Score': 0.693, 'Log-Loss': 0.979, 'Prediction': 0.725},
                50176: {'Accuracy': 0.732, 'F1-Score': 0.705, 'Log-Loss': 0.829, 'Prediction': 0.866}
            },
            'Grayscale': {
                784: {'Accuracy': 0.664, 'F1-Score': 0.578, 'Log-Loss': 1.405, 'Prediction': 0.442},
                4096: {'Accuracy': 0.701, 'F1-Score': 0.637, 'Log-Loss': 1.171, 'Prediction': 0.572},
                16384: {'Accuracy': 0.711, 'F1-Score': 0.648, 'Log-Loss': 0.979, 'Prediction': 0.725},
                50176: {'Accuracy': 0.732, 'F1-Score': 0.705, 'Log-Loss': 0.829, 'Prediction': 0.866}
            }
        },
        'PathMNIST': {
            'RGB': {
                784: {'Accuracy': 0.430, 'F1-Score': 0.408, 'Log-Loss': 3.813, 'Prediction': 0.110},
                4096: {'Accuracy': 0.786, 'F1-Score': 0.763, 'Log-Loss': 1.567, 'Prediction': 0.494},
                16384: {'Accuracy': 0.838, 'F1-Score': 0.837, 'Log-Loss': 1.047, 'Prediction': 0.800},
                50176: {'Accuracy': 0.882, 'F1-Score': 0.884, 'Log-Loss': 0.739, 'Prediction': 1.195}
            },
            'Grayscale': {
                784: {'Accuracy': 0.747, 'F1-Score': 0.756, 'Log-Loss': 1.045, 'Prediction': 0.718},
                4096: {'Accuracy': 0.716, 'F1-Score': 0.717, 'Log-Loss': 1.442, 'Prediction': 0.497},
                16384: {'Accuracy': 0.857, 'F1-Score': 0.860, 'Log-Loss': 0.728, 'Prediction': 1.179},
                50176: {'Accuracy': 0.763, 'F1-Score': 0.770, 'Log-Loss': 1.299, 'Prediction': 0.590}
            }
        },
    },
    'XGBoost': {
        'BreastMNIST': {
            'Grayscale': {
                784: {'Accuracy': 0.872, 'F1-Score': 0.918, 'Log-Loss': 0.350, 'Prediction': 2.556},
                4096: {'Accuracy': 0.821, 'F1-Score': 0.885, 'Log-Loss': 0.443, 'Prediction': 1.925},
                16384: {'Accuracy': 0.795, 'F1-Score': 0.867, 'Log-Loss': 0.442, 'Prediction': 1.879},
                50176: {'Accuracy': 0.846, 'F1-Score': 0.902, 'Log-Loss': 0.397, 'Prediction': 2.200}
            }
        },
        'DermaMNIST': {
            'RGB': {
                784: {'Accuracy': 0.723, 'F1-Score': 0.690, 'Log-Loss': 0.752, 'Prediction': 0.943},
                4096: {'Accuracy': 0.727, 'F1-Score': 0.685, 'Log-Loss': 0.773, 'Prediction': 0.913},
                16384: {'Accuracy': 0.717, 'F1-Score': 0.672, 'Log-Loss': 0.769, 'Prediction': 0.903},
                50176: {'Accuracy': 0.723, 'F1-Score': 0.677, 'Log-Loss': 0.768, 'Prediction': 0.912}
            },
            'Grayscale': {
                784: {'Accuracy': 0.698, 'F1-Score': 0.637, 'Log-Loss': 0.887, 'Prediction': 0.752},
                4096: {'Accuracy': 0.694, 'F1-Score': 0.628, 'Log-Loss': 0.888, 'Prediction': 0.744},
                16384: {'Accuracy': 0.701, 'F1-Score': 0.638, 'Log-Loss': 0.892, 'Prediction': 0.751},
                50176: {'Accuracy': 0.700, 'F1-Score': 0.637, 'Log-Loss': 0.887, 'Prediction': 0.753}
            }
        },
        'PathMNIST': {
            'RGB': {
                784: {'Accuracy': 0.820, 'F1-Score': 0.820, 'Log-Loss': 0.513, 'Prediction': 1.598},
                4096: {'Accuracy': 0.815, 'F1-Score': 0.813, 'Log-Loss': 0.522, 'Prediction': 1.560},
                16384: {'Accuracy': 0.791, 'F1-Score': 0.789, 'Log-Loss': 0.563, 'Prediction': 1.404},
                50176: {'Accuracy': 0.779, 'F1-Score': 0.776, 'Log-Loss': 0.594, 'Prediction': 1.309}
            },
            'Grayscale': {
                784: {'Accuracy': 0.652, 'F1-Score': 0.647, 'Log-Loss': 0.932, 'Prediction': 0.696},
                4096: {'Accuracy': 0.684, 'F1-Score': 0.680, 'Log-Loss': 0.838, 'Prediction': 0.814},
                16384: {'Accuracy': 0.686, 'F1-Score': 0.681, 'Log-Loss': 0.841, 'Prediction': 0.813},
                50176: {'Accuracy': 0.690, 'F1-Score': 0.686, 'Log-Loss': 0.841, 'Prediction': 0.818}
            }
        }
    }
}

# Define variables needed for plotting
metrics = ['Accuracy', 'F1-Score', 'Log-Loss', 'Prediction']
colors = ['Grayscale', 'RGB']
markers = {'Grayscale': 'o', 'RGB': 's'}
linestyles = ['-', '--', '-.', ':', (0, (5, 10))]  # Adding a custom dash pattern for variety

# For loop to create each plot. 
for method, datasets in scores_dict.items():
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))  # One figure for all plots
    
    fig.suptitle(f'{method} Performance Metrics', fontsize=20)
    
    for i, metric in enumerate(metrics):
        row = i // 2
        col = i % 2
        ax = axs[row, col]
        ax.set_xlabel('Number of Pixels', fontsize=14)
        ax.set_ylabel(metric, fontsize=14)
        ax.set_title(metric, fontsize=18)
        
        linestyle_iter = iter(linestyles)
        # Plot each dataset for the current metric
        for dataset, colors_data in datasets.items():
            for color, resolutions_data in colors_data.items():
                linestyle = next(linestyle_iter)
                ax.plot(list(resolutions_data.keys()), [score[metric] for score in resolutions_data.values()],
                        label=f'{dataset} ({color})', marker=markers[color], linestyle=linestyle)
        
        ax.legend(fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=12)  
        ax.tick_params(axis='both', which='minor', labelsize=10)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(file_path + f"/Results/Graphs/{method}_performance.png")
    plt.close()

print("Plots saved successfully.")