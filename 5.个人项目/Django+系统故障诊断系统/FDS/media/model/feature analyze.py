import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
input = pd.read_csv('./datashet/output.csv')
input = input.iloc[:,1:]
raw = input.iloc[:,-1]
rows_with_Zero=[]
rows_with_One=[]
rows_with_Two=[]
rows_with_Three = []
rows_with_Four = []
rows_with_Five = []
for j in range(0,len(raw),1):
    if raw[j]==0:
        rows_with_Zero.append(j)
        continue
    elif raw[j]==1:
        rows_with_One.append(j)
        continue
    elif raw[j]==2:
        rows_with_Two.append(j)
        continue
    elif raw[j]==3:
        rows_with_Three.append(j)
        continue
    elif raw[j]==4:
        rows_with_Four.append(j)
        continue
    else:rows_with_Five.append(j)

data_0 = input.iloc[rows_with_Zero]
data_1 = input.iloc[rows_with_One]
data_2 = input.iloc[rows_with_Two]
data_3 = input.iloc[rows_with_Three]
data_4 = input.iloc[rows_with_Four]
data_5 = input.iloc[rows_with_Five]
All_concat = pd.concat([data_0, data_1,data_2,data_3,data_4,data_5])
All_concat = All_concat.reset_index(drop=True)
All_concat.to_csv('./datashet/data_all_reshape.csv')

num=700
data_0 = data_0.sample(n=num, random_state=2023)
data_1 = data_1.sample(n=num, random_state=2023)
data_2 = data_2.sample(n=num, random_state=2023)
data_3 = data_3.sample(n=600, random_state=2023)
data_4 = data_4.sample(n=550, random_state=2023)
data_5 = data_5.sample(n=600, random_state=2023)


df_concat = pd.concat([data_0, data_1,data_2,data_3,data_4,data_5])
df_concat = df_concat.reset_index(drop=True)
df_concat.to_csv('./datashet/data_reshape.csv')
