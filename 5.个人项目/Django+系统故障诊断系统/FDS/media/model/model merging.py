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
data = pd.read_csv('./datashet/data_all_reshape.csv')
raw_data = data.iloc[:,2:]

last_column = raw_data.columns[-1]

# 将最后一列中不为0和1的值更改为1
raw_data.loc[~raw_data[last_column].isin([0, 1]), last_column] = 1

X = raw_data.iloc[:,0:-1]
y = raw_data.iloc[:,-1]
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler.fit(X)
X_train_normalized = scaler.transform(X)
trainx, testx, trainlabel, testlabel = train_test_split(X_train_normalized, y, test_size=0.3, random_state=2023)
clf = XGBClassifier(n_estimators=99, objective='multi:softmax',num_class=2,
                    max_depth= 7,learning_rate = 0.2,colsample_bytree=0.65)
clf.fit(trainx,trainlabel)
y_pred = clf.predict(testx)
score=classification_report(testlabel, y_pred)
# print(score)

second_data = data.iloc[5144:,2:]
X_second = second_data.iloc[:,0:-1]
scaler_2 = MinMaxScaler(feature_range=(-1, 1))
scaler_2.fit(X_second)
X_train_normalized_2 = scaler.transform(X_second)
y_second = second_data.iloc[:,-1]-1
trainx_second, testx_second, trainlabel_second, testlabel_second = train_test_split(X_train_normalized_2, y_second, test_size=0.3, random_state=2025)
clf_second = XGBClassifier(n_estimators = 60,colsample_bytree = 0.6, learning_rate = 0.7, max_depth= 7)
clf_second.fit(trainx_second,trainlabel_second)
y_pred_second = clf_second.predict(testx_second)
score=classification_report(testlabel_second, y_pred_second)
# print(score)

trainx_1, testx_1, trainlabel_1, testlabel_1 = train_test_split(X_train_normalized, data.iloc[:,-1], test_size=0.3, random_state=2022)
df_pre_test=clf.predict(testx_1)
non_zero_indices_0 = np.nonzero(df_pre_test)[0]
selected_rows_12345 = testx_1[non_zero_indices_0]
df_pre_12 = clf_second.predict(selected_rows_12345)+1
df_pre_test[non_zero_indices_0]=df_pre_12
score=classification_report(testlabel_1, df_pre_test)
print('精度')
print(score)
# clf.fit(trainx,trainlabel)
# y_pred = clf.predict(testx)
# score=classification_report(testlabel, y_pred)
# print(score)

# learning_rate = [0.1,0.3,0.5,0.7]
# # subsample = [0.5,0.6,0.7]
# colsample_bytree = [0.55,0.6,0.65,0.7]
# # min_child_weight =[0.1,0.2,0.3,0.4,0.5,0.6,0.8,0.7]
# max_depth = [6,7,8]
# n_estimators=[50,60,70,80,90,96,100]
# parameters = {
#               'learning_rate': learning_rate,
#               # 'subsample': subsample,
#               'colsample_bytree':colsample_bytree,
#               'max_depth': max_depth,
#               # 'min_child_weight':min_child_weight
#               'n_estimators':n_estimators
#               }
# #
# model = XGBClassifier( )
# # # 进行网格搜索
# clf = GridSearchCV(model, parameters, cv=3, scoring='accuracy',verbose=1,n_jobs=-1)
# clf = clf.fit(trainx_second, trainlabel_second)
# # # #%%网格搜索后的参数
# print(clf.best_params_)


df = pd.read_csv('./datashet/validate_1000.csv')
df_X = df.iloc[:, :-1]
df_X_list = df_X.values.tolist()

df_imputer = KNNImputer(n_neighbors=1)
df_result = df_imputer.fit_transform(df_X_list)

df_y = pd.DataFrame(df_result)

df.iloc[:,:-1]=df_y
df_data=df.iloc[:,1:]

df_data_x = df_data.iloc[:,:-1]
scaler_3 = MinMaxScaler(feature_range=(-1, 1))
scaler_3.fit(df_data_x)
df_data_x_0 = scaler.transform(df_data_x)
df_data_y = df_data.iloc[:,-1]
compare=df_data_y
sereis = compare.where((compare == 0) | (compare == 1), 1)
df_pre_0=clf.predict(df_data_x_0)

non_zero_indices = np.nonzero(df_pre_0)[0]
print(len(non_zero_indices))
compare_s=classification_report(sereis, df_pre_0)
print(compare_s)
selected_rows = df_data.iloc[non_zero_indices]
selected_rows_x = selected_rows.iloc[:,:-1]
scaler_4 = MinMaxScaler(feature_range=(-1, 1))
scaler_4.fit(selected_rows_x)
selected_rows_x_1 = scaler.transform(selected_rows_x)
selected_rows_y = selected_rows.iloc[:,-1]
df_pre_12345 = clf_second.predict(selected_rows_x_1)+1

selected_rows['predicted'] = df_pre_12345

last_column = selected_rows.columns[-1]
last_column_values = selected_rows[last_column].values
last_column_indices = selected_rows[last_column].index.values

predicted_reshape = np.zeros(len(df_data_y))

predicted_reshape[last_column_indices] = last_column_values

Reshape_score=classification_report(df_data_y, predicted_reshape)
print(Reshape_score)

lines = Reshape_score.split('\n')
precision=[]
recall=[]
for i in range(2,8):
    precision.append(float(lines[i].split()[1]))
    recall.append(float(lines[i].split()[2]))
precision_average = sum(precision) / len(precision)
recall_average = sum(recall) / len(recall)
F1_score=2*(precision_average*recall_average)/(precision_average+recall_average)

print(F1_score)
# json_result = {}
# for key in range(len(df_data_y)):
#     # json_result[key] =str(df_pre[key])
#     json_result[key] = predicted_reshape[key]
#
# class NumpyEncoder(json.JSONEncoder):
#     def default(self, obj):
#         if isinstance(obj, np.integer):
#             return int(obj)
#         return super(NumpyEncoder, self).default(obj)
#
# with open('./result/Merging_XGB_Result.json', 'w') as file:
#     json.dump(json_result, file,cls=NumpyEncoder)

# 保存模型到磁盘文件
model_file = 'xgboost_clf.pkl'
pickle.dump(clf, open(model_file, 'wb'))
model_file_2 = 'xgboost_clf_2.pkl'
pickle.dump(clf_second, open(model_file, 'wb'))
# 获取模型文件的大小
file_size = sys.getsizeof(model_file)
file_size_2 = sys.getsizeof(model_file_2)
total = file_size+file_size_2
print("模型大小: {} bytes".format(total))