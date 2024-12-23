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
df.to_csv(file_path , index = False)

df.isnull().sum()

df = df.drop("deck" , axis = 1)

df = df.dropna()
df = df.drop_duplicates()

x = df.drop("survived" , axis = 1)

y= df["survived"]


cat_col = x.select_dtypes(include=["object"]).columns.tolist()
num_col = x.select_dtypes(include=[np.number]).columns.tolist()

x_train , x_test ,y_train , y_test = train_test_split(x, y, test_size=0.20 , random_state = 42)

transformer = ColumnTransformer(transformers=[("cat" , OneHotEncoder() , cat_col), ("num" , StandardScaler() , num_col)])

pipeline = Pipeline(steps=[("transformer" , transformer), ("classifier" , LogisticRegression())])
