import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import  train_test_split
from sklearn.linear_model import LinearRegression , LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score , precision_score ,f1_score
from sklearn.preprocessing import OneHotEncoder , StandardScaler
import os
df = sns.load_dataset("titanic")

data_dir = "data" 
os.makedirs(data_dir , exist_ok = True)
file_path = os.path.join(data_dir , "sample.csv")
df.to_csv(data_dir , index = False)