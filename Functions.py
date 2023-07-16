# -*- coding: utf-8 -*-
"""
Created on Wed May 24 11:37:53 2023

@author: alonp
"""

#%%----------Import libraries----------##
import numpy as np     #NumPy - Used mainly for matrix operations
import pandas as pd    #Pandas - library for data analysis and cleanup before training the model
import matplotlib.pyplot as plt    #matplot is used for data visualization 
from sklearn import metrics
from sklearn.metrics import PredictionErrorDisplay
import pymatgen.core as pymat_co     #utilidades específicas para material science sobretodo con formulas
import math


#%%----------FUNCTIONS DATABASE FORMATION----------##
def formatFormula(formula_col):
    formula_col = formula_col.apply(      #Alphabetical order and normalized composition for the formula column
        lambda x: pymat_co.Composition(x).fractional_composition.alphabetical_formula)
    formula_col = formula_col.str.replace(" ", "")      #Eliminate the spaces between the composition
    return formula_col

def fractionalFormulaCreation(df):
    df_provisional = pd.DataFrame([])  #Create a provisional DF to work with
    
    #Create columns with the element and its composition as values
    for column in df.columns:  #Apply to all except last column, "HV"
        df_provisional[column] = df[column].apply(  #Row-wise operation
            lambda x: str(column) + (str(x)))     #Para cada valor de la fila (es decir, x) concatenalo con el nombre de la columna
 
    #Create a new vector with the concatenataed values of each column to create the chemical formula   
    Formula = df_provisional.apply(
        lambda x: " ".join(x.astype(str)),
        axis = 1)   #Apply as a col-wise operation
    return Formula

def columnFullness(df):
    return np.divide(df.count(), len(df.index))    #Calculate the ammount of non-NaN values in the columns

def repeatedFormula(column):
    doubles = [False]*len(column)    #initialize a vector to store the repeated formulas
    counter = 0
    #We compare each value in the column with the ones after it unless it returns a "True" repeated
    for i in column[:-1]:   #For ever y value until the second to last item
        for item2 in column[counter+1:]:  #For every value after the ith term
            doubles[counter] = pymat_co.Composition(i).almost_equals(   #compare both values
                pymat_co.Composition(item2), 
                rtol = .1, atol = 0.0)  #Consider a 10% margin
            if doubles[counter] == True:     #If at any point the formula i repeated, stop the loop and move onto the next i
                break
        counter += 1

    return doubles

def compare(cols, rows):
    counter = 0
    matrix = pd.DataFrame([])
    for i in cols:  #Compare item i to every row in col2
        matrix[counter] = rows.apply(      #For each row on column 1 do: 
            lambda x: pymat_co.Composition(i).almost_equals(pymat_co.Composition(x), 
                                                         rtol = .25, atol = 0.0))
        counter += 1
    return matrix

    
def countTrue(df):  #Retruns a Series with the sum of True values on each column
    sum_vect = pd.Series([], dtype = (float))
    counter = 0
    for col in df.columns:  #Iterate over every column
        sum_val = 0
        for item in df[col]:    #Iterate over every row
            sum_val = sum_val + int(item == True)
            #if item == True:      
            sum_vect[counter] = sum_val
            
        counter += 1
    return sum_vect
       
def graph_bars(X, Y, Xlab, Ylab):
    plt.bar(X, Y)
    plt.xlabel(str(Xlab))
    #plt.xticks(np.arange(min(X), max(X), step = (max(X) - min(X))/5))
    plt.ylabel(str(Ylab))
    plt.show()
  
#Creates a matrix of just the elements in the formula    
def justElements(formula_cols):
    formula_cols = formula_cols.map(lambda x: "".join([i for i in x if not i.isdigit()]))   #We remove all of the numbers and since all of them are rational we are left with a "." between each element.
    formula_cols = formula_cols.str.split(".", expand=True)     #Split the strings into columns based on the "." delimitor
    return formula_cols

#Creates a matrix with the fractinal representation, column names are the elements and the items the composition
def createFractional(formulas):
    fractional = pd.DataFrame([])
    for i in formulas: 
        formula_dict = pymat_co.Composition(i).as_dict() #Obtain a dict with the format {Element: composition, Element: composition}
        fractional_vect = pd.DataFrame(pd.Series(formula_dict)) #We first make it a Series so we can change a dict to pandas format, then we change it into a df so we can transpose it
        fractional = pd.concat([fractional, fractional_vect.T]) 
    fractional = fractional.fillna(0) #replace all of the Nan values for 0
    fractional = fractional.reset_index(drop = True)
    return fractional

def countElements(df, elements_col):
    vector = pd.Series([])
    counter = 0
    for element in elements_col: 
        matrix = df.isin([element])
        val = pd.Series(countTrue(matrix).sum())
        #if true add the value of the row  from the column RMSE
        vector = pd.concat([vector, val])
        counter += 1
    #df_return = pd.concat([elements_col, vector], axis = 1, ignore_index = True)
    return vector


def countElements2(df, elements_col, dif_sqrd):
    vector = pd.Series([])
    vector_rmse = pd.Series([])
    counter = 0
    for element in elements_col: 
        matrix = df.isin([element])
        val, rmse = countTrue2(matrix, dif_sqrd)
        #if true add the value of the row  from the column RMSE
        # vector = pd.concat([vector, pd.Series[val]])
        vector_rmse = pd.concat([vector_rmse, pd.Series(rmse)])
        counter += 1
    #df_return = pd.concat([elements_col, vector], axis = 1, ignore_index = True)
    return vector, vector_rmse


def countTrue2(df, dif):  #Retruns a Series with the sum of True values on each row
    sum_vect = pd.Series([], dtype = (float))
    sum_dif_vect = pd.Series([], dtype = (float))

    counter = 0
    for index in df.index:  #Iterate over every row
        sum_val = 0
        sum_dif = 0
        for item in df.iloc[index]:    #Iterate over every col
            sum_val = sum_val + int(item == True)   #value with a 1 if the row includes a True
            sum_dif = sum_val*dif.iloc[index]   #value with the value of dif_sqrd if it is 1
            #if item == True:      
            sum_vect[counter] = sum_val     #vector saying that each row contains _ units
            sum_dif_vect[counter] = sum_dif     #vector saying that each row contains 

            
        counter += 1
    if sum(sum_vect) == 0:
        srme = 0
    else: 
        srme = math.sqrt(sum(sum_dif_vect)/sum(sum_vect))
    return sum(sum_vect), srme


#%%----------Functions Model Creation----------##
#Calculation of the evaluation metrics
def evalMetrics(y_val, y_pred):
    mae = metrics.mean_absolute_error(y_val, y_pred)
    mse = metrics.mean_squared_error(y_val, y_pred)
    rmse = mse**0.5 
    r2 = metrics.r2_score(y_val, y_pred)
    print("\n--Test results")
    print(f"""
    MAE: \t{mae:.2f}
    RMSE: \t{rmse:.2f}
    r2: \t{r2:.2f}
    """)
    return r2, mae, rmse

#Visualize data comparing model vs reality
def plot_PredReal(y_val, y_pred):
    plt.plot([0, max(max(y_val), max(y_pred))], [0, max(max(y_val), max(y_pred))], 'k-.')
    plt.scatter(y_val, y_pred)
    plt.ylabel('Predicted HV')
    plt.xlabel('Actual HV')
    plt.title("Predicted vs Actual Values")
    plt.show()  

def showResults(y_test, y_pred):
    fig, axs = plt.subplots(ncols=2, figsize=(8, 4))
    PredictionErrorDisplay.from_predictions(
        y_test,
        y_pred=y_pred,
        kind="actual_vs_predicted",
        subsample=100,
        ax=axs[0],
        random_state=0,
    )
    PredictionErrorDisplay.from_predictions(
        y_test,
        y_pred=y_pred,
        kind="residual_vs_predicted",
        subsample=100,
        ax=axs[1],
        random_state=0,
    )
    plt.tight_layout()
    plt.show() 
       

def OptimumModelResutls(grid_search, num_params, search): 
    optimum_model_results = pd.Series(grid_search.loc[search.best_index_])
    optimum_model_results = pd.DataFrame({"Fold": [1, 2, 3, 4, 5], 
                        "Test MAE": pd.to_numeric(optimum_model_results[5+num_params:10+num_params].values)*-1, 
                        "Train MAE ": pd.to_numeric(optimum_model_results[13+num_params:18+num_params].values)*-1, 
                        "Test RMSE": pd.to_numeric(optimum_model_results[20+num_params:25+num_params].values)*-1,
                        "Train RMSE": pd.to_numeric(optimum_model_results[28+num_params:33+num_params].values)*-1, 
                        "Test R2": pd.to_numeric(optimum_model_results[35+num_params:40+num_params].values), 
                        "Train R2": pd.to_numeric(optimum_model_results[43+num_params:48+num_params].values), 
                        })
    optimum_model_results = optimum_model_results.round(2)
    
    print("The average MAE is " + str(optimum_model_results["Test MAE"].mean()) + "The STD is " + str(optimum_model_results["Test MAE"].values.std()))
    print("The average RMSE is " + str(optimum_model_results["Test RMSE"].mean()) + "The STD is " + str(optimum_model_results["Test RMSE"].values.std()))
    print("The average R2 is " + str(optimum_model_results["Test R2"].mean()) + "The STD is " + str(optimum_model_results["Test R2"].values.std()))
    return optimum_model_results
    
