import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
# 假设你有一个特征矩阵 X 和对应的目标变量 y
from sklearn.impute import KNNImputer
import json
# 划分数据集为训练集和测试集
data_1 = pd.read_csv('./datashet/output.csv')
data_2 = pd.read_csv('./datashet/validate_1000.csv')
data = pd.concat([data_1, data_2]).reset_index(drop=True)
data_list = data.values.tolist()
data_imputer = KNNImputer(n_neighbors=1)
df_result = data_imputer.fit_transform(data_list)
df_data = pd.DataFrame(df_result)
X = df_data.iloc[:,1:-1]
y = df_data.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2023)

# 创建随机森林分类器对象
rf_classifier = RandomForestClassifier(n_estimators=20)

# 训练随机森林分类器
rf_classifier.fit(X_train, y_train)

# 使用训练好的分类器进行预测
y_pred = rf_classifier.predict(X_test)

score=classification_report(y_test, y_pred)
print(score)

lines = score.split('\n')
precision=[]
recall=[]
for i in range(2,8):
    precision.append(float(lines[i].split()[1]))
    recall.append(float(lines[i].split()[2]))
precision_average = sum(precision) / len(precision)
recall_average = sum(recall) / len(recall)
F1_score=200*(precision_average*recall_average)/(precision_average+recall_average)

print(F1_score)

data_dict = {str(i): y_pred[i].tolist() for i in range(len(y_pred))}
file_path = './result/RF_data.json'
with open(file_path, 'w') as f:
    json.dump(data_dict, f)


# df = pd.read_csv('./datashet/test_2000_x.csv')
# df_X = df.iloc[:, :-1]
# df_X_list = df_X.values.tolist()
#
# df_imputer = KNNImputer(n_neighbors=1)
# df_result = df_imputer.fit_transform(df_X_list)
#
#
# df_y = pd.DataFrame(df_result)
#
# df.iloc[:,:-1]=df_y
# df_test=df.iloc[:,1:-1]
# df_pre=rf_classifier.predict(df_test)
# df_actual = df.iloc[:,-1]
# test_score = classification_report(df_actual, df_pre)
# print(test_score)
# lines = test_score.split('\n')
# precision=[]
# recall=[]
# for i in range(2,8):
#     precision.append(float(lines[i].split()[1]))
#     recall.append(float(lines[i].split()[2]))
# precision_average = sum(precision) / len(precision)
# recall_average = sum(recall) / len(recall)
# F1_score=2*(precision_average*recall_average)/(precision_average+recall_average)
#
# print(F1_score)
#
# data_dict = {str(i): df_pre[i].tolist() for i in range(len(df_pre))}
# file_path = './result/RF_data.json'
# with open(file_path, 'w') as f:
#     json.dump(data_dict, f)
