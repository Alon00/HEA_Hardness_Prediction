# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 18:01:29 2023

@author: alonp
"""
"""##----------Code  Documentation----------##
File to predict new HEA hardness based on the trained models

- Description: Generate the necessary database (CBFV) and predict the hardness on the loaded models
- Objective: Predict hardness of HEA 
- Problem type: ML prediction
- Input: 
    Data source: Excel with the composition values
    Features: CBFV
- Output:
    Database: predicted hardness nd error measurement
"""
#%%----------Import libraries----------##
import pandas as pd    #Pandas - library for data analysis and cleanup before training the model
import matplotlib.pyplot as plt    #matplot is used for data visualization 
from CBFV import composition     #Lets easilly create composition vectors based on several databases
import pymatgen.util.plotting as pymat_plt     #poder generar un periodic table heatmap
import numpy as np     #NumPy - Used mainly for matrix operations
from sklearn.model_selection import train_test_split, cross_val_predict, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic, ExpSineSquared
from sklearn.pipeline import Pipeline
from joblib import dump, load


#%%----------Load data----------##
#Define data path
path = r"C:\Users\alonp\OneDrive\Desktop\TFM - Alonso Cuartero\AI - HEA\Article\Code\Experimental_data\Real_predictions.csv"
path_elements = r"C:\Users\alonp\OneDrive\Desktop\TFM - Alonso Cuartero\AI - HEA\Article\Code\Data\Raw_data\Elements_Periodic_Table.csv"

#Create dataframes
df = pd.read_csv(path, header = 0)
df_elements = pd.read_csv(path_elements)

del path; del path_elements

#%%----------FUNCTIONS----------##
import Functions as f


#%%----------Database Creation----------##
#Create the formula based of the fractional vector representation
df_Formula = f.fractionalFormulaCreation(df[df.columns[:-1]])  #Input as argumen a df with only the fractional vectors
df.insert(0, "Formula", df_Formula)   #Add the newly created vector to the first index of the df
del df_Formula

#Having the appropiate formual format, the fractional representation becomes useless
df = df[["Formula", "Hardness_HV"]]

#Create a Composition Based Feature Vector using Oliynyk
df_CBFV, df_Hardness, formulae, skipped = composition.generate_features(
    df.rename(columns = {"Formula": "formula", "Hardness_HV": "target"}),  #Change column names so it matches teh necessary notation
    elem_prop = 'oliynyk')
del formulae; del skipped   #Remove unwanted byproducts of CBFV generation

#Combine the dataframe with the CBFV
df = pd.concat([df, df_CBFV], axis = 1).reset_index(drop = True)


#Change the index so it matches the formula of the alloy
index_df = df.pop("Formula")
df = df.rename(index = index_df)
del index_df

#%%----------Load models----------##
NN_model = load('NN_trained.joblib') 
SVR_model = load('SVR_trained.joblib') 
RF_model = load('RF_trained.joblib') 
GP_model = load('GP_trained.joblib') 


#%%----------Split data----------##
Y = df.pop("Hardness_HV")
X = df


#%%----------Predict and evaluate prediction----------##
Y_pred = RF_model.predict(X)

r2, mae, rmse = f.evalMetrics(Y, Y_pred)     #Calculation of the evaluation metrics

