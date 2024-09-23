import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import json
import sys
import pickle
import matplotlib.pyplot as plt
from xgboost.sklearn import XGBClassifier
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.impute import KNNImputer
####模型训练####
def trian(file_one,file_two):
    ###读取数据###
    data_1 = pd.read_csv(file_one).iloc[:, 1:]
    data_2 = pd.read_csv(file_two).iloc[:, 1:]

    ####将第一次给的10000个样本和第二次1000样本拼接起来####
    data = pd.concat([data_1, data_2]).reset_index(drop=True)
    #####利用KNN补缺失值
    data_list = data.values.tolist()
    data_imputer = KNNImputer(n_neighbors=1)
    df_result = data_imputer.fit_transform(data_list)
    df_data = pd.DataFrame(df_result)
    ###分离特征和标签###
    X = df_data.iloc[:, :-1]
    y = df_data.iloc[:, -1]

    ######划分训练集和测试集####
    trainx, testx, trainlabel, testlabel = train_test_split(X, y, test_size=0.2, random_state=2023)

    ###为解决数据不平衡，分两个模型预测，第一个模型预测0和1，第二个模型预测剩下的故障类型###

    ###从trainX，trainlabel里面划分第一个模型要用到的训练集，测试集和第二模型要用到的训练集，测试集
    trainlabel_1 = trainlabel.copy()
    trainlabel_1[(trainlabel_1 != 0) & (trainlabel_1 != 1)] = 1
    non_zero_indexes = trainlabel[trainlabel != 0].index
    ####第一个模型里面的测试集直接由trainX来替代，然后将trainlabel里面的不是0和1的全部换为1
    trainx_1, testx_1, trainlabel_1_1, testlabel_1_1 = train_test_split(trainx, trainlabel_1, test_size=0.3,
                                                                        random_state=2022)

    ####第二个模型要筛检掉trainlabel里面是0的数据，重组一个数据集
    trainx_2 = trainx.loc[non_zero_indexes].copy()
    trainlabel_2 = trainlabel.loc[non_zero_indexes].copy()

    ###第一个模型####
    clf_first = XGBClassifier(n_estimators=85, objective='multi:softmax', num_class=2,
                              max_depth=8, learning_rate=0.2, colsample_bytree=0.55)
    clf_first.fit(trainx_1, trainlabel_1_1)
    # y_pred = clf_first.predict(testx_1)
    # first_score = classification_report(testlabel_1_1, y_pred)
    # print(first_score)
    trainx_second, testx_second, trainlabel_second, testlabel_second = train_test_split(trainx_2, trainlabel_2,test_size=0.3,random_state=2025)

    ####第二个模型####
    clf_second = XGBClassifier(n_estimators=85, colsample_bytree=0.55, learning_rate=0.2,
                               objective='multi:softmax', num_class=5, max_depth=6)
    clf_second.fit(trainx_second, trainlabel_second - 1)
    # y_pred_second = clf_second.predict(testx_second)
    # second_score = classification_report(testlabel_second - 1, y_pred_second)
    # print(second_score)
    return clf_first,clf_second,testx,testlabel
###验证集测验
def validate(clf_first, clf_second, testx):
    ###首先预测0类有多少###
    y_pred_0 = clf_first.predict(testx)

    ###剔除0类，用第二个模型预测剩下类别
    non_zero_indices = np.nonzero(y_pred_0)[0]
    selected_rows_x = testx.iloc[non_zero_indices].copy()
    y_pre_12345 = clf_second.predict(selected_rows_x) + 1

    ###将两个预测结果拼接起来
    predicted_reshape = np.zeros(len(y_pred_0))
    predicted_reshape[non_zero_indices] = y_pre_12345
    # test_score = classification_report(testlabel, predicted_reshape)
    # print(test_score)
    return predicted_reshape
###提取指标
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
#####测试集训练####
def unkonwn_test(clf_first, clf_second,filename):
    ####测试集数据读入####
    df = pd.read_csv(filename)
    df_X = df.iloc[:, 1:]
    df_X_list = df_X.values.tolist()
    ####用KNNImputer填充缺失值####
    df_imputer = KNNImputer(n_neighbors=1)
    df_result = df_imputer.fit_transform(df_X_list)
    df_y = pd.DataFrame(df_result)
    ###补其缺少值后，进行测试####
    result=validate(clf_first, clf_second, df_y)
    ####返回结果####
    return result
####结果文件保存#####
def Get_json(df_pre):
    data_dict = {str(i): df_pre[i].tolist() for i in range(len(df_pre))}
    file_path = './media/model/result/final_model_data.json'
    with open(file_path, 'w') as f:
        json.dump(data_dict, f)
###保存模型，并获取模型大小
def get_model_size(model_1,model_2):
    model_file = './media/model/result/final_xgboost_clf.pkl'
    pickle.dump(model_1, open(model_file, 'wb'))
    model_file_2 = './media/model/result/final_xgboost_clf_2.pkl'
    pickle.dump(model_2, open(model_file_2, 'wb'))
    # 获取模型文件的大小
    file_size = sys.getsizeof(model_file)
    file_size_2 = sys.getsizeof(model_file_2)
    total = file_size + file_size_2
    return total
    # print("模型大小: {} bytes".format(total))
###数据可视化
def visual(arr):
    # 统计数字出现的次数
    unique_vals, counts = np.unique(arr, return_counts=True)

    # 打印每个数字及其出现次数
    for val, count in zip(unique_vals, counts):
        print(f"{int(val)}: {int(count)}")
    # # 生成柱形图
    # plt.bar(unique_vals, counts)
    #
    # # 显示数值标签
    # for i, value in enumerate(counts):
    #     plt.text(i, value, str(value), ha='center', va='bottom')
    #
    # # 设置图形属性
    # plt.title('Bar Chart with labels')
    # plt.savefig('./media/model/result/bar_chart.png')
    # # 显示图形
    # plt.show()

def modelTrain(file_one,file_two):
    ###初始文件读取####
    # file_one = './datashet/train_10000.csv'
    # file_two = './datashet/validate_1000.csv'
    # ###测试集读取###
    # test_file_name ='./datashet/test_2000_x.csv'
    ###导入初始文件训练，返回模型###
    clf_first, clf_second, validatex, validatelabel = trian(file_one,file_two)
    ####验证结果获取
    validate_result = validate(clf_first, clf_second, validatex)
    ###验证结果指标获取及可视化####
    test_score = classification_report(validatelabel, validate_result)
    F1_Score = Get_F1_score(test_score)
    print("validate set  indicator ：")
    print(test_score)
    print('validate set F1: ',F1_Score)
    total = get_model_size(clf_first, clf_second)
    print("Model size: {} bytes".format(total))

    return clf_first, clf_second,test_score,F1_Score,total

def modelTest(clf_first, clf_second, test_file_name):
    testx_result = unkonwn_test(clf_first, clf_second, test_file_name)
    ###可视化结果获取
    # visual(testx_result)
    ####保存测试结果
    print(testx_result)
    Get_json(testx_result)
    unique_vals, counts = np.unique(testx_result, return_counts=True)
    print(unique_vals, counts)
    data1 = [{'name': '0', 'value': counts[0]},
             {'name': '1', 'value': counts[1]},
             {'name': '2', 'value': counts[2]},
             {'name': '3', 'value': counts[3]},
             {'name': '4', 'value': counts[4]},
             {'name': '5', 'value': counts[5]}]
    return  data1

if __name__ == '__main__':
    file_one = './datashet/train_10000.csv'
    file_two = './datashet/validate_1000.csv'
    test_file_name = './datashet/test_2000_x.csv'
    clf_first, clf_second = modelTrain(file_one,file_two)
    modelTest(clf_first, clf_second, test_file_name)
