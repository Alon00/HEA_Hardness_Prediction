# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 10:38:59 2023

@author: alonp
"""
"""##----------Code  Documentation----------##
File to upload, clean and prepare all of the data for the Ml algorithms

- Description: Combine several HEA Databases to have one cohesive table
- Objective: Obtain the best database possible for HV prediction
- Problem type: Data mining
- Input: 
    Data source: 2 databases from the literature
    Features: all experimental values
- Output:
    Database: 2 databases, one with fractional composition and another with CBFV
"""
#%%----------Import libraries----------##
import pandas as pd    #Pandas - library for data analysis and cleanup before training the model
import matplotlib.pyplot as plt    #matplot is used for data visualization 
from CBFV import composition     #Lets easilly create composition vectors based on several databases
import pymatgen.util.plotting as pymat_plt     #poder generar un periodic table heatmap


#%%----------Load data----------##
#Define data path
path1 = r"C:\Users\alonp\OneDrive\Desktop\TFM - Alonso Cuartero\AI - HEA\Article\Code\Data\Raw_data\Data1_Expanded dataset of mechanical properties and observed phases of multi-principal element alloys.csv"
path4 = r"C:\Users\alonp\OneDrive\Desktop\TFM - Alonso Cuartero\AI - HEA\Article\Code\Data\Raw_data\Data4_Machine learning assisted design of high entropy alloys with desired property.csv"
path_elements = r"C:\Users\alonp\OneDrive\Desktop\TFM - Alonso Cuartero\AI - HEA\Article\Code\Data\Raw_data\Elements_Periodic_Table.csv"

#Define column names
columns1 = ["Reference_ID", "Formula", "Microstructure", "Processing", "FCC_BCC_Other", "Grain_size", "Exp_Density", "Calc_Density", "Hardness_HV", "Test_type", "Tests_Temp", "YS", "UTS", "Elong", "Plas_Elong", "Elastic_Mod", "Calc_Elastic_Mod", "O_content", "N_content", "C_content", "DOI", "Year", "Title"]
columns4 = ["#", "Al", "Co", "Cr", "Cu", "Fe", "Ni", "Hardness_HV"]


#Create dataframes
df1 = pd.read_csv(path1, header = 0, names = columns1)

df4 = pd.read_csv(path4, header = 0, names = columns4).drop(index = [0]).reset_index(drop = True) #Few property data
df_elements = pd.read_csv(path_elements)

del path1; del path4; del path_elements
del columns1; del columns4


#%%----------FUNCTIONS----------##
import Functions as f

  
#%%----------DF1----------##
#Get rid of the columns that are less than 30% full
df1_percent_full = f.columnFullness(df1)
df1_percent_full = df1_percent_full[df1_percent_full >= 0.3].index  #Select columns with over 30% fullness
df1 = df1[df1_percent_full]
del df1_percent_full

#Remove more useless columns
useless_cols = ["Reference_ID", "Test_type", "DOI", "Year", "Title"]
df1 = df1.drop(columns = useless_cols)
#Define the data type to each column
df1 = df1.astype({'Formula': str, 'Microstructure': str, 'Processing': str, 'FCC_BCC_Other': str, 'Calc_Density': float, 'Hardness_HV': float, 'Tests_Temp': float, 'YS': float, 'UTS': float, 'Elong': float, 'Calc_Elastic_Mod': float})     #Define the data type to each column
df1_percent_full = f.columnFullness(df1)

#Remove columns that include experimental data since it will not be availible to make predictions it is not a valid parameter to train on
#Only paramenters that are commonly availible or theretically calculable can be used
experimental_cols = ['YS', 'UTS', 'Elong', 'Calc_Elastic_Mod']
df1 = df1.drop(columns = experimental_cols)
del experimental_cols

#Remove all rows that don't inclide a hardness value
df1 = df1[df1["Hardness_HV"].notnull()].reset_index(drop = True)
#Calculate the ammount of non-NaN values in the columns
df1_percent_full = f.columnFullness(df1)

#Format the "Formula" column 
df1["Formula"] = f.formatFormula(df1["Formula"])

#Remove samples tested at high temperatures
df1 = df1[(df1["Tests_Temp"].isnull()) | (df1["Tests_Temp"] < 100)].reset_index(drop = True)

#Select only homogeneous processing routes
df1 = df1[df1["Processing"].isin(["CAST", "ANNEAL", "POWDER"])].reset_index(drop = True)

#Remove duplicate formulas with a 10% margin, we do this to avoid having too many data from one family
vector_doubles = pd.Series(f.repeatedFormula(df1["Formula"]))
df1 = df1[~vector_doubles]  #Since the vector includes True for repeated values, we need to inver it with "~"


#%%----------DF4----------##
df4 = df4.drop(columns = "#")
df4["Hardness_HV"][129:137] = [207, 576, 415, 602, 155, 139, 430, 516]  #Remove some erroneous data
df4 = df4.astype({"Al": float, "Co": float, "Cr": float, "Cu": float, "Fe": float, "Ni": float, "Hardness_HV": float})     #Define the data type to each column


#Create the formula based of the fractional vector representation
df4_Formula = f.fractionalFormulaCreation(df4[df4.columns[:-1]])  #Input as argumen a df with only the fractional vectors
df4.insert(0, "Formula", df4_Formula)   #Add the newly created vector to the first index of the df
del df4_Formula

#Add a column with the processing route 
df4["Processing"] = ["CAST"]*len(df4["Hardness_HV"])

#Format the formula 
df4["Formula"] = f.formatFormula(df4["Formula"])
#Having the appropiate formual format, the fractional representation becomes useless
useless_cols = ['Al', 'Co', 'Cr', 'Cu', 'Fe', 'Ni']
df4 = df4.drop(columns = useless_cols)

#Remove duplicate formulas with a 10% margin, we do this to avoid having too many data from one family
vector_doubles = pd.Series(f.repeatedFormula(df4["Formula"]))
df4 = df4[~vector_doubles]  #Since the vector includes True for repeated values, we need to inver it with "~"


#%%----------DF1+DF4----------##
#Join the dataframes to create the final tadabase
df1 = df1[["Formula", "Hardness_HV", "Processing"]]

df = pd.concat([df1, df4], ignore_index = True)

#When processing route wants to be ignored: 
df = df.drop(columns=["Processing"])


#Remove duplicate formulas with a 10% margin, we do this to avoid having too many data from one family
vector_doubles = pd.Series(f.repeatedFormula(df["Formula"]))
df = df[~vector_doubles].reset_index(drop = True)  #Since the vector includes True for repeated values, we need to inver it with "~"


#Create fractional variation of the df
df_fractional = f.createFractional(df["Formula"])

#Combine the dataframe with the fractional
FRACTIONAL_DF = pd.concat([df, df_fractional], axis = 1).reset_index(drop = True)


#Create a Composition Based Feature Vector using Oliynyk
df_CBFV, df_Hardness, formulae, skipped = composition.generate_features(
    df.rename(columns = {"Formula": "formula", "Hardness_HV": "target"}),  #Change column names so it matches teh necessary notation
    elem_prop = 'oliynyk')
del formulae; del skipped   #Remove unwanted byproducts of CBFV generation


#Combine the dataframe with the CBFV
CBFV_DF = pd.concat([df, df_CBFV], axis = 1).reset_index(drop = True)


#%%----------Plotting----------##
#Create a Periodic Table Heatmap that displays how many times an element is used 
df_just_elements = f.justElements(df["Formula"])  #Creates a df with a column per element in the alloy
numofElements = f.countElements(df_just_elements, df_elements["Symbol"])  #Count how many times an elelent appears
periodic_input = dict(zip(df_elements["Symbol"], numofElements))    #Create a dictionary with the key being the element and the value how many times it appears
periodic_input = {key:val for key, val in periodic_input.items() if val != 0}   #Remove all the elements that don't appear
pymat_plt.periodic_table_heatmap(periodic_input, cbar_label='Number of appearances')    #Represent a perioric table heat map with the frecuency of elements
plt.show()
del numofElements; del periodic_input


#Create a Histogram to visualize how many ellements the alloys have
hist_num_of_elements = len(df_just_elements.columns) - 1 - df_just_elements.apply(lambda x: x.isnull().sum(), axis='columns')   #Create a vector that counts how many times elements each alloy has
plt.hist(hist_num_of_elements, 
          bins = range(min(hist_num_of_elements.unique()), max(hist_num_of_elements.unique())), 
          width = 1, edgecolor = "white", linewidth = 0.7)
plt.xlabel("Number of elements")
plt.ylabel("Frecuency")
plt.show()
del df_just_elements; del hist_num_of_elements

#Create a histogram to visualize the hardness values
plt.hist(df.Hardness_HV, edgecolor = "white", linewidth = 0.7)
plt.xlabel("Hardness value")
plt.ylabel("Frecuency")
plt.show()


#%%----------Examine possible outliers----------##
#Create a boxplot to visualize the hardness values
plt.boxplot(df.Hardness_HV)
plt.xlabel("Hardness Value")
plt.ylabel("Hardness")
plt.show()


#Remove outliers 
CBFV_DF = CBFV_DF.drop(177).reset_index(drop = True)
FRACTIONAL_DF = FRACTIONAL_DF.drop(177).reset_index(drop = True)


#%%----------Export final dataframe----------##
#Exportar LOS df a cvs para no tener que correr este código cada vez 
CBFV_DF.to_csv(r'C:\Users\alonp\OneDrive\Desktop\TFM - Alonso Cuartero\AI - HEA\Article\Code\Data\Processed_data\CBFV_DF_Property.csv', index = False)
FRACTIONAL_DF.to_csv(r'C:\Users\alonp\OneDrive\Desktop\TFM - Alonso Cuartero\AI - HEA\Article\Code\Data\Processed_data\FRACTIONAL_DF_Property.csv', index = False)



#%%----------Expplore test results----------##

y_test = pd.read_csv(r'C:\Users\alonp\OneDrive\Desktop\TFM - Alonso Cuartero\AI - HEA\TFM_Python_Code\Results\Validation\y_test.csv')
y_pred = pd.read_csv(r'C:\Users\alonp\OneDrive\Desktop\TFM - Alonso Cuartero\AI - HEA\TFM_Python_Code\Results\Validation\y_pred.csv')
dif_sqrd = (y_pred["0"] - y_test["Hardness_HV"])**2

df_just_elements = f.justElements(y_test["Unnamed: 0"])  #Creates a df with a column per element in the alloy


numofElements, rmse_elements = f.countElements2(df_just_elements, df_elements["Symbol"], dif_sqrd)  #Count how many times an elelent appears
rmse_elements = pd.DataFrame(rmse_elements, columns = ["RMSE"])
rmse_elements = rmse_elements.reset_index(drop = True).set_index(df_elements["Element"])
periodic_input = dict(zip(df_elements["Symbol"], rmse_elements["RMSE"]))    #Create a dictionary with the key being the element and the value how many times it appears
periodic_input = {key:val for key, val in periodic_input.items() if val != 0}   #Remove all the elements that don't appear
pymat_plt.periodic_table_heatmap(periodic_input, cbar_label='RMSE [HV]')    #Represent a perioric table heat map with the frecuency of elements
plt.show()
del numofElements; del periodic_input


