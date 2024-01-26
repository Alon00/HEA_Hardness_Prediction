# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 10:47:59 2023

@author: alonp
"""
"""##----------Code  Documentation----------##
File to upload, clean and prepare all of the data for the Ml algorithms

- Description: File that includes the ML models for hardness prediction of HEA
- Objective: Dessign several ML models to find the best performing algorithm
- Problem type: Supervised machine learning - Regression
- Input: 
    Data source: Premade DF from "HEA_Property_DatabaseFormation"
    Features: Formula, FBCV, Fractional
- Output:
    Parameter: 4 Models (NN, SVR, RF, GPR)
    Results: Best performing model is the RF with RMSE=95.99 & R2=0.81
    """
#%%----------Import libraries----------##
import numpy as np     #NumPy - Used mainly for matrix operations
import pandas as pd    #Pandas - library for data analysis and cleanup before training the model
import matplotlib.pyplot as plt    #matplot is used for data visualization 
from sklearn.model_selection import train_test_split, cross_val_predict, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic, ExpSineSquared
from sklearn.pipeline import Pipeline
from joblib import dump, load


#%%----------Functions----------##
import Functions as f


#%%----------Load data----------##
#Define datapath
cbfv_path = r"C:\Local_path\Data\Processed_data\CBFV_DF_Property.csv"
fractional_path = r"C:\Local_path\Data\Processed_data\FRACTIONAL_DF_Property.csv"
path_elements = r"C:\Local_path\Data\Raw_data\Elements_Periodic_Table.csv"
df_elements = pd.read_csv(path_elements)

#Create DataFrame
df = pd.read_csv(cbfv_path)
del cbfv_path, fractional_path


#%%----------Split data----------##
#Extract all of the samples that include Pd so we can later observe how well it performs on never before seen elements
list_Pd_composition = ["Co0.2Cr0.2Fe0.2Ni0.2Pd0.2", 
                       "Co0.19230769Cr0.19230769Fe0.19230769Mn0.03846154Ni0.19230769Pd0.19230769", 
                       "Co0.18518519Cr0.18518519Fe0.18518519Mn0.07407407Ni0.18518519Pd0.18518519", 
                       "Co0.17857143Cr0.17857143Fe0.17857143Mn0.10714286Ni0.17857143Pd0.17857143", 
                       "Co0.17241379Cr0.17241379Fe0.17241379Mn0.13793103Ni0.17241379Pd0.17241379"]

df_Pd_test = df[df["Formula"].isin(list_Pd_composition)].reset_index(drop = True)   #Create a df taht includes the values for Pd
df = df[~df["Formula"].isin(list_Pd_composition)].reset_index(drop = True)  #Remove the values of Pd from df
del list_Pd_composition

#Define a random seed for all data 
rng = np.random.RandomState(1)

#Randomly shuffle all of the data
df = df.sample(frac = 1, random_state = rng, ignore_index = True)


#Change the values from categorical to numerical so the models can process them. For simolicity make it a boolean 0-1 if CAST/ANNEAL-POWDER
# df.loc[df['Processing'].isin(['CAST', 'ANNEAL']), 'Processing'] = 1
# df.loc[df['Processing'].isin(['POWDER']), 'Processing'] = 0


#Change the index so it matches the formula of the alloy
index_df = df.pop("Formula")
df = df.rename(index = index_df)
del index_df

#Generate 2 df with the label "y" and features "X"
y = df.pop("Hardness_HV")
X = df
del df

#Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = rng)
del X, y

#Create CV set
folds = 5


#%%----------Normalize data----------##
#In this step we declare the normalization technique for the pipeline
scaler = StandardScaler()

score = {"MAE": "neg_mean_absolute_error", 
         "RMSE": "neg_root_mean_squared_error",
         "R2": "r2"}


#%%----------Create Train and Test NN Model----------##
print("-----Evaluation of a NN model-----")

#Create a NN pipeline
NN_Pipe = Pipeline(steps = [("Scale", scaler), ("NeuralNetwork", MLPRegressor(random_state = rng))])

hyperparam_grid = {"NeuralNetwork__hidden_layer_sizes": [[10, 10], [100, 100], [10, 20, 10], [100, 200, 100]],
                    "NeuralNetwork__activation": ["relu", "logistic"], 
                    "NeuralNetwork__solver": ["lbfgs", "adam"], 
                    "NeuralNetwork__max_iter": [100, 1000, 10000, 100000]}

optimum_hyperparam = {'NeuralNetwork__activation': ['logistic'], 
                    'NeuralNetwork__hidden_layer_sizes': [100, 100], 
                    'NeuralNetwork__max_iter': [10000], 
                    'NeuralNetwork__solver': ['adam']}

#Train and evaluate the NN model
search = GridSearchCV(estimator = NN_Pipe, 
                      param_grid = optimum_hyperparam, 
                      scoring = score,
                      refit = "RMSE",
                      cv = folds, 
                      return_train_score = True)
search.fit(X_train, y_train)
print("Best parameter (CV score=%0.3f):" % search.best_score_)
print(search.best_params_)

#Transform results into DF
grid_search = pd.DataFrame(search.cv_results_)  #Convert the results into a DataFrame
optimum_model_results = f.OptimumModelResutls(grid_search, len(hyperparam_grid), search)    #Create a DataFrame with the best model only


#Test the model against the test dataset
y_pred = search.predict(X_test)
r2, mae, rmse = f.evalMetrics(y_test, y_pred)     #Calculation of the evaluation metrics


#Visualize test validation
f.showResults(y_test, y_pred)

#Save the trained model 
dump(search, 'NN_trained.joblib') 


#%%----------Create Train and Test SVR Model----------##
print("-----Evaluation of a SVR model-----")

#Create a SVR pipeline
SVR_Pipe = Pipeline(steps = [("Scale", scaler), ("SupportVectorRegression", SVR())])

hyperparam_grid = {"SupportVectorRegression__kernel": ["linear", "poly", "rbf", "sigmoid"], #LINEAL IS THE BEST BY FAR
                    "SupportVectorRegression__epsilon": [0.1, 1, 10, 50, 100, 200], 
                    "SupportVectorRegression__C": [1, 5, 15, 50, 100]}

optimum_hyperparam = {'SupportVectorRegression__C': [15], 
                      'SupportVectorRegression__epsilon': [100], 
                      'SupportVectorRegression__kernel': ['linear']}


#Train and evaluate the SVR model
search = GridSearchCV(estimator = SVR_Pipe, 
                      param_grid = optimum_hyperparam, 
                      scoring = score,
                      refit = "RMSE",
                      cv = folds, 
                      return_train_score = True)
search.fit(X_train, y_train)
print("Best parameter (CV score=%0.3f):" %search.best_score_)
print(search.best_params_)

#Transform results into DF
grid_search = pd.DataFrame(search.cv_results_)  #Convert the results into a DataFrame
optimum_model_results = f.OptimumModelResutls(grid_search, len(hyperparam_grid), search)    #Create a DataFrame with the best model only


#Test the model agains the test dataset
y_pred = search.predict(X_test)
r2, mae, rmse = f.evalMetrics(y_test, y_pred)     #Calculation of the evaluation metrics


#Visualize test validation
f.showResults(y_test, y_pred)

#Save the trained model 
dump(search, 'SVR_trained.joblib') 


#%%----------Create Train and Test Random Forest Model----------##
print("-----Evaluation of a RF model-----")

#Create a RF pipeline
RF_Pipe = Pipeline(steps = [
    ("Scale", scaler), 
    ("RandomForestRegressor", RandomForestRegressor(criterion = "squared_error", random_state = rng))
    ])

hyperparam_grid = {"RandomForestRegressor__n_estimators": [10, 100, 1000], #LINEAL IS THE BEST BY FAR
                    "RandomForestRegressor__max_depth": [2, 5, 10], 
                    "RandomForestRegressor__min_samples_split": [5, 10], 
                    "RandomForestRegressor__min_samples_leaf": [2, 5, 10], 
                    "RandomForestRegressor__max_features": [2, 10, 50]}

optimum_hyperparam = {'RandomForestRegressor__max_depth': [10], 
                      'RandomForestRegressor__max_features': [50], 
                      'RandomForestRegressor__min_samples_leaf': [2], 
                      'RandomForestRegressor__min_samples_split': [5], 
                      'RandomForestRegressor__n_estimators': [1000]}

#Train and evaluate the RF model
search = GridSearchCV(estimator = RF_Pipe, 
                      param_grid = optimum_hyperparam, 
                      scoring = score,
                      refit = "RMSE",
                      cv = folds, 
                      return_train_score = True)
search.fit(X_train, y_train)
print("Best parameter (CV score=%0.3f):" %search.best_score_)
print(search.best_params_)


#Transform results into DF
grid_search = pd.DataFrame(search.cv_results_)  #Convert the results into a DataFrame
optimum_model_results = f.OptimumModelResutls(grid_search, len(hyperparam_grid), search)    #Create a DataFrame with the best model only


#Test the model against the test dataset
y_pred = search.predict(X_test)
r2, mae, rmse = f.evalMetrics(y_test, y_pred)     #Calculation of the evaluation metrics

#Visualize test validation
f.showResults(y_test, y_pred)

#Save the trained model 
dump(search, 'RF_trained.joblib') 

#-----Test the model against the Pd data it has never seen-----
print("Dado que el RF nos proporciona el mejor resultado, se intenta predecir el Pd con este modelo")
y_pred = cross_val_predict(RF_Pipe, df_Pd_test[df_Pd_test.columns[3:]], df_Pd_test[df_Pd_test.columns[1]], cv = folds)
r2, mae, rmse = f.evalMetrics(df_Pd_test[df_Pd_test.columns[1]], y_pred)     #Calculation of the evaluation metrics

#Visualize test validation
f.showResults(df_Pd_test[df_Pd_test.columns[1]], y_pred)


#-----Check feature importance of the RF model-----
important_features = pd.Series(search.best_estimator_._final_estimator.feature_importances_, index = X_test.columns)
important_features = important_features.sort_values(ascending = False)

cum_frec = np.cumsum([0]+important_features[:]).tolist()

plt.bar(np.arange(0,important_features.size), important_features)
plt.xlabel("Feature")
plt.ylabel("Weight")
plt.show()

plt.plot(np.arange(0,important_features.size), cum_frec)
plt.xlabel("Feature")
plt.ylabel("Cumulative weight")
plt.show()


# important_features.to_csv(r'C:\Users\alonp\OneDrive\Desktop\TFM - Alonso Cuartero\AI - HEA\TFM_Python_Code\Results\Validation\ImportantFeatures.csv')
important_features = important_features.iloc[0:80]

# y_test.to_csv(r'C:\Users\alonp\OneDrive\Desktop\TFM - Alonso Cuartero\AI - HEA\TFM_Python_Code\Results\Validation\y_test.csv')
# y_pred.to_csv(r'C:\Users\alonp\OneDrive\Desktop\TFM - Alonso Cuartero\AI - HEA\TFM_Python_Code\Results\Validation\y_pred.csv')
# pd.DataFrame(y_pred).to_csv(r'C:\Users\alonp\OneDrive\Desktop\TFM - Alonso Cuartero\AI - HEA\TFM_Python_Code\Results\Validation\y_pred.csv')


#%%----------Create Train and Test GPR (Gaussian Process Regressor) Model----------##
print("-----Evaluation of a GPR model-----")

#Create a GPR pipeline
GPR_Pipe = Pipeline(steps = [("Scale", scaler), ("GaussianProcessRegressor", GaussianProcessRegressor(random_state = rng))])

hyperparam_grid = {"GaussianProcessRegressor__kernel": [RBF(), RationalQuadratic(), ExpSineSquared()],
                    "GaussianProcessRegressor__n_restarts_optimizer": [1, 2, 5, 10]}

hyperparam_grid_RatQuad = {"GaussianProcessRegressor__kernel": [RationalQuadratic(.5), RationalQuadratic(1), RationalQuadratic(5)],
                    "GaussianProcessRegressor__n_restarts_optimizer": [2, 5, 10]}

optimum_hyperparam = {"GaussianProcessRegressor__kernel": [RationalQuadratic(1)],
                    "GaussianProcessRegressor__n_restarts_optimizer": [2]}

#Train and evaluate the GPR model
search = GridSearchCV(estimator = GPR_Pipe, 
                      param_grid = optimum_hyperparam, 
                      scoring = score,
                      refit = "RMSE",
                      cv = folds, 
                      return_train_score = True, 
                      error_score = "raise")
search.fit(X_train, y_train)
print("Best parameter (CV score=%0.3f):" %search.best_score_)
print(search.best_params_)
# print("The average R2 is " + str(np.average(GPR_r2)))


#Transform results into DF
grid_search = pd.DataFrame(search.cv_results_)  #Convert the results into a DataFrame
optimum_model_results = f.OptimumModelResutls(grid_search, len(hyperparam_grid), search)    #Create a DataFrame with the best model only


#Test the model against the test dataset
y_pred = search.predict(X_test)
r2, mae, rmse = f.evalMetrics(y_test, y_pred)     #Calculation of the evaluation metrics


#Visualize test validation
f.showResults(y_test, y_pred)

#Save the trained model 
dump(search, 'GP_trained.joblib') 


#%%----------Expplore test results----------##
final_performance = pd.DataFrame({"MAE": [71.33, 91.71, 64.91, 108.22],
                                "RMSE": [107.63, 123.51, 95.99, 145.31],
                                "R2": [0.76, 0.68, 0.81, 0.56]}, index = ["NN", "SVR", "RF", "GP"])

plt.scatter(final_performance["R2"][0], final_performance["RMSE"][0], 100)
plt.scatter(final_performance["R2"][1], final_performance["RMSE"][1], 100) 
plt.scatter(final_performance["R2"][2], final_performance["RMSE"][2], 100)
plt.scatter(final_performance["R2"][3], final_performance["RMSE"][3], 100)
plt.xlabel("R2")
plt.ylabel("RMSE [HV]")
plt.legend(final_performance.index)
plt.show()



