# Analysis function script

## Package Import
### Python Packages 
import sys

### My Modules
from data_manip import dataset_namer, get_var_name
from data_loader import data_loader
from file_path import file_path
from images import *
from mach_learn import *
from medmnist_setup import medmnist_generator
from stat_learn import *

## Define analysis function
def image_analysis(name, data, method, stage, random_seed, colour_change, img_save = 'No', img_num = ''):

    # Change case of parameters for consistency
    method = method.replace(" ", "").upper()
    colour_change = colour_change.upper()
    stage = stage.upper()
        
    # Split train/test/val sets for the dataset based on function output
    train = data[0]
    test = data[1]
    val = data[2]
    
    # Convert colours if required
    if colour_change in ['GRAY','GREY','GREYSCALE','GRAYSCALE'] and 'breast' not in name: 
        train = image_convert(train, mode=colour_change)
        test = image_convert(test, mode=colour_change)
        val = image_convert(val, mode=colour_change)
    elif 'breast' in name:
        colour_change = 'gray'
    else:
        colour_change = 'rgb'
            
    # Save image samples if required
    if img_save == 'Yes':
        if '224' in name:
            image_generator(name, train, 4, colour_change, number = img_num)
        else:
            image_generator(name, train, 1, colour_change, number = img_num)
    
    #Split each set into features and labels and convert images
    train = features_labels(train, method)
    test = features_labels(test, method)
    val = features_labels(val, method)
    
    # Statistical Methods 
    ## LDA
    if method in ["LDA","LINEARDISCRIMINANTANALYSIS"]:
        
        if stage in ["TRAIN","TRAINING","VALIDATION","VAL","V"]:
            stage = "TRAIN"
            # Define model parameters
            parameters = {'baseline' : {}, 
                          'lsqr' : {'solver' : 'lsqr'},
                          'lsqr_shrinkage' : {'solver' : 'lsqr', 'shrinkage' : 'auto'}}
            lda(name,train,test,val,stage,random_seed,parameters,colour_change)
        
        elif stage in ["TEST","TESTING"]:
            stage = "TEST"
            solver = str(input("Solver: ").lower())
            shrinkage = str(input("Shrinkage: ").lower())
            parameters = {'optimal_model': {'solver' : 'svd' if solver == '' else solver, 
                                            'shrinkage' : None if shrinkage == '' else shrinkage}}
            lda(name,train,test,val,stage,random_seed, parameters, colour_change)
            sys.exit()              
                          
        else:
            print("Error: The stage parameter does not indicate a stage of model building. Terminating.")
            sys.exit()
                           
    # Logistic Regression 
    if method in ["LR","LOGREG","LOGISTICREGRESSION"]:
    
        if stage in ["TRAIN","TRAINING","VALIDATION","VAL","V"]:
            stage = "TRAIN"
            # Define model parameters
            parameters = {'Ridge_LBFGS' : {},
                          'Ridge_NewtonCG' : {'solver' : 'newton-cg'},
                          'Ridge_SAG' : {'solver' : 'sag', 'random_state' : random_seed},
                          'LASSO' : {'penalty' : 'l1', 'solver' : 'saga', 'random_state' : random_seed}}
            logistic_regression(name,train,test,val,stage,random_seed,parameters,colour_change)
            
        elif stage in ["TEST","TESTING"]:
            stage = "TEST"
            penalty = str(input("Penalty: ").lower())
            solver = str(input("Solver: ").lower())
            parameters = {'optimal_model': {'penalty' : 'l2' if penalty == '' else penalty,
                                            'solver' : 'lbfgs' if solver == '' else solver, 
                                            'random_state' : random_seed if solver != 'lbfgs' else None}}
            logistic_regression(name,train,test,val,stage,random_seed, parameters, colour_change)
            sys.exit()              
                          
        else:
            print("Error: The stage parameter does not indicate a stage of model building. Terminating.")
            sys.exit()
    
    # Machine Learning Methods 
    ## CNN
    if method in ["CNN","CONVOLUTIONALNEURALNETWORK","NN","NEURAL","NEURALNETWORK"]:
        
        conv_nn(name,train,test,val, random_seed, colour_change)
    
    ## XGBoost
    if method in ["XGB","XGBOOST","BOOST"]:
    
        if stage in ["TRAIN","TRAINING","VALIDATION","VAL","V"]:
            stage = "TRAIN"
            # Define model parameters
            parameters = {'model1' : {'random_state' : random_seed},
                          'model2' : {'random_state' : random_seed, 'eta': 0.15},
                          'model3' : {'random_state' : random_seed, 'eta':0.05}}
                          
                          
            xg_boost(name,train,test,val,stage,random_seed,parameters,colour_change)
            
        elif stage in ["TEST","TESTING"]:
            stage = "TEST"
            penalty = str(input("Penalty: ").lower())
            solver = str(input("Solver: ").lower())
            parameters = {'optimal_model': {'penalty' : 'l2' if penalty == '' else penalty,
                                            'solver' : 'lbfgs' if solver == '' else solver, 
                                            'random_state' : random_seed if solver != 'lbfgs' else None}}
            xg_boost(name,train,test,val,stage,random_seed, parameters, colour_change)
            sys.exit()              
                          
        else:
            print("Error: The stage parameter does not indicate a stage of model building. Terminating.")
            sys.exit()
        
 
            
            
 

