import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import json
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from xgboost.sklearn import XGBClassifier
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.impute import KNNImputer
import pickle
import sys
###读取数据###
data_1 = pd.read_csv('./datashet/train_10000.csv').iloc[:,1:]
data_2 = pd.read_csv('./datashet/validate_1000.csv').iloc[:,1:]

####将第一次给的10000个样本和第二次1000样本拼接起来####
data = pd.concat([data_1, data_2]).reset_index(drop=True)

###分离特征和标签###
X = data.iloc[:,:-1]
y = data.iloc[:,-1]

######划分训练集和测试集####
trainx, testx,trainlabel,testlabel = train_test_split(X,y, test_size=0.2, random_state=2023)

###为解决数据不平衡，分两个模型预测，第一个模型预测0和1，第二个模型预测剩下的故障类型###

###从trainX，trainlabel里面划分第一个模型要用到的训练集，测试集和第二模型要用到的训练集，测试集
trainlabel_1 = trainlabel.copy()
trainlabel_1[(trainlabel_1 != 0) & (trainlabel_1 != 1)] = 1
non_zero_indexes = trainlabel[trainlabel != 0].index
####第一个模型里面的测试集直接由trainX来替代，然后将trainlabel里面的不是0和1的全部换为1
trainx_1,testx_1,trainlabel_1_1,testlabel_1_1= train_test_split(trainx,trainlabel_1,test_size=0.3,random_state=2022)

####第二个模型要筛检掉trainlabel里面是0的数据，重组一个数据集
trainx_2 = trainx.loc[non_zero_indexes].copy()
trainlabel_2 = trainlabel.loc[non_zero_indexes].copy()

###第一个模型####
clf_first = XGBClassifier(n_estimators=85, objective='multi:softmax',num_class=2,
                    max_depth= 8,learning_rate = 0.2,colsample_bytree=0.55)
clf_first.fit(trainx_1,trainlabel_1_1)
y_pred = clf_first.predict(testx_1)
first_score = classification_report(testlabel_1_1, y_pred)
print(first_score)


trainx_second, testx_second, trainlabel_second, testlabel_second = train_test_split(trainx_2, trainlabel_2, test_size=0.3,random_state=2025)

####第二个模型####
clf_second = XGBClassifier(n_estimators = 85,colsample_bytree = 0.55, learning_rate = 0.2,
                           objective='multi:softmax',num_class=5,max_depth= 6)
clf_second.fit(trainx_second,trainlabel_second-1)
y_pred_second = clf_second.predict(testx_second)
second_score=classification_report(testlabel_second-1, y_pred_second)
print(second_score)
###至此训练结束

####网格搜索最优模型参数####
# n_estimators=[55,60,65,85,90,99]
# max_depth = [6,7,8]
# learning_rate = [0.2,0.6,0.8,0.7]
# colsample_bytree = [0.55,0.6,0.65,0.7]

# parameters = {
#               'learning_rate': learning_rate,
#               'colsample_bytree':colsample_bytree,
#               'max_depth': max_depth,
#               'n_estimators':n_estimators
#               }
#
# model = XGBClassifier( )
# # # 进行网格搜索
# clf = GridSearchCV(model, parameters, cv=3, scoring='accuracy',verbose=1,n_jobs=-1)
# clf=clf.fit(trainx_second,trainlabel_second-1)
# # # #%%网格搜索后的参数
# print(clf.best_params_)

####验证####
y_pred_0 = clf_first.predict(testx)
non_zero_indices = np.nonzero(y_pred_0)[0]
selected_rows_x = testx.iloc[non_zero_indices].copy()
y_pre_12345 = clf_second.predict(selected_rows_x)+1
predicted_reshape = np.zeros(len(y_pred_0))
predicted_reshape[non_zero_indices] = y_pre_12345

test_score = classification_report(testlabel, predicted_reshape)
print(test_score)

lines = test_score.split('\n')
precision=[]
recall=[]
for i in range(2,8):
    precision.append(float(lines[i].split()[1]))
    recall.append(float(lines[i].split()[2]))
precision_average = sum(precision) / len(precision)
recall_average = sum(recall) / len(recall)
F1_score=100*2*(precision_average*recall_average)/(precision_average+recall_average)

print(F1_score)