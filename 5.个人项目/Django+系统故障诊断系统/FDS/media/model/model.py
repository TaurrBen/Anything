import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import json
import sys
import pickle
from xgboost.sklearn import XGBClassifier
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.impute import KNNImputer

def train(filename):
    data = pd.read_csv(filename)
    raw_data = data.iloc[:, 2:]

    last_column = raw_data.columns[-1]

    # 将最后一列中不为0和1的值更改为1
    raw_data.loc[~raw_data[last_column].isin([0, 1]), last_column] = 1

    X = raw_data.iloc[:, 0:-1]
    y = raw_data.iloc[:, -1]
    trainx, testx, trainlabel, testlabel = train_test_split(X, y, test_size=0.3, random_state=2023)
    clf = XGBClassifier(n_estimators=99, objective='multi:softmax',num_class=2,
                    max_depth= 7,learning_rate = 0.2,colsample_bytree=0.65)
    clf.fit(trainx, trainlabel)

    second_data = data.iloc[5144:, 2:]
    X_second = second_data.iloc[:, 0:-1]
    y_second = second_data.iloc[:, -1] - 1
    trainx_second, testx_second, trainlabel_second, testlabel_second = train_test_split(X_second, y_second,
                                                                                        test_size=0.3,
                                                                                        random_state=2025)
    clf_second = XGBClassifier(objective='multi:softmax',num_class=5,n_estimators=60, colsample_bytree=0.6, learning_rate=0.7, max_depth=7, subsample=0.9)
    clf_second.fit(trainx_second, trainlabel_second)
    return clf,clf_second
# def test(X,y,clf,clf_second):
#     trainx_1, testx_1, trainlabel_1, testlabel_1 = train_test_split(X, y, test_size=0.3,random_state=20)
#     df_pre_test = clf.predict(testx_1)
#     non_zero_indices_0 = np.nonzero(df_pre_test)[0]
#     selected_rows_12345 = testx_1.iloc[non_zero_indices_0]
#     df_pre_12 = clf_second.predict(selected_rows_12345) + 1
#     df_pre_test[non_zero_indices_0] = df_pre_12
#     score = classification_report(testlabel_1, df_pre_test)
#     return score
def mytest_data_split(filename):
    df = pd.read_csv(filename)
    df_X = df.iloc[:, :-1]
    df_X_list = df_X.values.tolist()

    df_imputer = KNNImputer(n_neighbors=1)
    df_result = df_imputer.fit_transform(df_X_list)

    df_y = pd.DataFrame(df_result)

    df.iloc[:, :-1] = df_y
    df_data = df.iloc[:, 1:]

    df_data_x = df_data.iloc[:, :-1]
    df_data_y = df_data.iloc[:, -1]
    return df_data_x,df_data_y,df_data
def Get_F1_score(Reshape_score):
    lines = Reshape_score.split('\n')
    precision = []
    recall = []
    for i in range(2, 8):
        precision.append(float(lines[i].split()[1]))
        recall.append(float(lines[i].split()[2]))
    precision_average = sum(precision) / len(precision)
    recall_average = sum(recall) / len(recall)
    F1_score = 200 * (precision_average * recall_average) / (precision_average + recall_average)

    return F1_score
def Get_json(df_pre):
    data_dict = {str(i): df_pre[i].tolist() for i in range(len(df_pre))}
    file_path = './result/model_data.json'
    with open(file_path, 'w') as f:
        json.dump(data_dict, f)

def get_model_size(model_1,model_2):
    model_file = './result/xgboost_clf.pkl'
    pickle.dump(model_1, open(model_file, 'wb'))
    model_file_2 = './result/xgboost_clf_2.pkl'
    pickle.dump(model_2, open(model_file_2, 'wb'))
    # 获取模型文件的大小
    file_size = sys.getsizeof(model_file)
    file_size_2 = sys.getsizeof(model_file_2)
    total = file_size + file_size_2
    print("模型大小: {} bytes".format(total))


if __name__ == '__main__':
    file_train ='./datashet/data_all_reshape.csv'
    first_model,second_model=train(file_train)
    # data = pd.read_csv(file_train).iloc[:,-1]
    # Model_score=test(X,data,first_model,second_model)
    # print('模型精度')
    # print(Model_score)
    file_test = './datashet/validate_1000.csv'
    df_data_x,df_data_y,df_data=mytest_data_split(file_test)
    df_pre_0 = first_model.predict(df_data_x)

    non_zero_indices = np.nonzero(df_pre_0)[0]
    selected_rows = df_data.iloc[non_zero_indices]
    selected_rows_x = selected_rows.iloc[:, :-1]
    selected_rows_y = selected_rows.iloc[:, -1]
    df_pre_12345 = second_model.predict(selected_rows_x) + 1
    selected_rows['predicted'] = df_pre_12345

    last_column = selected_rows.columns[-1]
    last_column_values = selected_rows[last_column].values
    last_column_indices = selected_rows[last_column].index.values
    predicted_reshape = np.zeros(len(df_data_y))

    predicted_reshape[last_column_indices] = last_column_values

    Reshape_score = classification_report(df_data_y, predicted_reshape)
    print(Reshape_score)
    F1_Score=Get_F1_score(Reshape_score)
    print('F1_score:',F1_Score)
    Get_json(predicted_reshape)
    get_model_size(first_model,second_model)
