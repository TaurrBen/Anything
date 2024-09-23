import numpy as np
import pandas as pd
import os

from IPython.core.display_functions import display
from IPython.display import clear_output
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
import seaborn as sns
import datetime as dt
# Making sklearn pipeline outputs as dataframe:-
from sklearn import set_config

set_config(transform_output="pandas")
pd.set_option('display.max_columns', 200)
pd.set_option('display.max_rows', 200)
random_state = 202309

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.feature_selection import mutual_info_classif
from scipy import stats
from sklearn.decomposition import PCA

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, KFold, RepeatedKFold, cross_val_score

# ML Model training:-
# from xgboost import DMatrix, XGBRegression
# from lightgbm import LGBMClassifier, log_evaluation, early_stopping
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
# from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,AdaBoostClassifier,BaggingClassifier,HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.cluster import KMeans, AgglomerativeClustering, Birch, MiniBatchKMeans, SpectralClustering
from sklearn.mixture import GaussianMixture

print(os.getcwd())
for dirname, _, filenames in os.walk('./数据'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import torch
import gpytorch

table1 = pd.read_excel('./数据\表1-患者列表及临床信息.xlsx')
display(table1.head(5))
table2 = pd.read_excel('./数据\表2-患者影像信息血肿及水肿的体积及位置.xlsx')
display(table2.head(5))
table3_ED = pd.read_excel('./数据\表3-患者影像信息血肿及水肿的形状及灰度分布.xlsx',sheet_name='ED')
display(table3_ED.head(5))
table3_Hemo = pd.read_excel('./数据\表3-患者影像信息血肿及水肿的形状及灰度分布.xlsx',sheet_name='Hemo')
display(table3_Hemo.head(5))
table4 = pd.read_excel('./数据\表4-答案文件.xlsx')
display(table4.head(5))
table5 = pd.read_excel('./数据\附表1-检索表格-流水号vs时间.xlsx')
display(table5.head(5))
clear_output()
Problem1a = pd.DataFrame()
Problem1a['发病到首次影像检查时间间隔'] = table1[table1.index<100]['发病到首次影像检查时间间隔']
Problem1a['resT'] = [pd.Timedelta(hours=i) for i in (48 - table1[table1.index<100]['发病到首次影像检查时间间隔'])]
Problem1a = pd.concat([Problem1a,table5[table5.index<100]], axis=1)
Problem1a = pd.concat([Problem1a,table2.iloc[range(0,100),range(2,9*23+1,23)]], axis=1)
Problem1a['发病时间'] = [i - pd.Timedelta(hours=j) for i,j in zip(pd.to_datetime(Problem1a['入院首次检查时间点']),table1[table1.index<100]['发病到首次影像检查时间间隔'])]
Problem1a['isExpansion_count'] = 0
# display(Problem1a['resT'])
for index in range(1,9):
#     display((pd.to_datetime(Problem1a['随访{}时间点'.format(index)]) - pd.to_datetime(Problem1a['入院首次检查时间点'])))
#     display((pd.to_datetime(Problem1a['随访{}时间点'.format(index)]) - pd.to_datetime(Problem1a['入院首次检查时间点'])) < Problem1a['resT'])
#     display(abs(Problem1a['HM_volume.{}'.format(index)]-Problem1a['HM_volume']) >= 6000)
#     display((abs(Problem1a['HM_volume.{}'.format(index)]-Problem1a['HM_volume'])/Problem1a['HM_volume']) >= 0.33)
    Problem1a['deltaT{}'.format(index)] = (pd.to_datetime(Problem1a['随访{}时间点'.format(index)]) - pd.to_datetime(Problem1a['入院首次检查时间点'])) < Problem1a['resT']
    Problem1a['deltaVolume{}'.format(index)] = (Problem1a['HM_volume.{}'.format(index)]-Problem1a['HM_volume']).fillna(0)
    Problem1a['deltaRelativeVolume{}'.format(index)] = ((Problem1a['HM_volume.{}'.format(index)]-Problem1a['HM_volume'])/Problem1a['HM_volume']).fillna(0)
    Problem1a['deltaV{}'.format(index)] = np.logical_or(Problem1a['deltaVolume{}'.format(index)]>=6000,Problem1a['deltaRelativeVolume{}'.format(index)]>=0.33)
    Problem1a['isExpansion{}'.format(index)] = np.logical_and(Problem1a['deltaT{}'.format(index)],Problem1a['deltaV{}'.format(index)])
    Problem1a['isExpansion_count'] = Problem1a['isExpansion_count'] + Problem1a['isExpansion{}'.format(index)]
    if index > 1:
        pass
Problem1a['isExpansion'] = [int(i) for i in Problem1a['isExpansion_count']>=1]
def calExpansionTime(Problem1a):
    for row in range(0,100):
        if Problem1a.loc[row,'isExpansion']==True:
            for index in range(1,9):
                if Problem1a.loc[row,'isExpansion{}'.format(index)]==True:
                    Problem1a.loc[row,'isExpansion_time'] = (Problem1a.loc[row,'随访{}时间点'.format(index)]-Problem1a.loc[row,'发病时间'])*24
                    break;
calExpansionTime(Problem1a)
Problem1a.to_excel('Problem1a.xlsx')
# Problem1a['isExpansion1'] = sum(Problem1a['isExpansion{}'.format(index)])
# display(Problem1a.loc[range(0,100),['deltaT{}'.format(index) for index in range(1,9)]].head(5))
# display(Problem1a.loc[range(0,100),['deltaVolume{}'.format(index) for index in range(1,9)]])
# display(Problem1a.loc[range(0,100),['deltaRelativeVolume{}'.format(index) for index in range(1,9)]])
# display(Problem1a.loc[range(0,100),['deltaV{}'.format(index) for index in range(1,9)]].head(5))
# display(Problem1a.loc[range(0,100),['isExpansion{}'.format(index) for index in range(1,9)]])
# display(Problem1a['isExpansion'])
# dt.datetime.strptime(Problem1a['随访{}时间点'.format(index)],"%Y/%m/%d %H-%M")
Problem1b = table1
Problem1b = pd.concat([Problem1b,table2.iloc[:,range(2,24)]],axis=1)
Problem1b.loc[:,'收缩压']=[int(i.split('/')[0]) for i in Problem1b['血压'].to_list()]
Problem1b.loc[:,'舒张压']=[int(i.split('/')[1]) for i in Problem1b['血压'].to_list()]
def findRow(Problem1b,table,row):
    for row1,num in enumerate(table['流水号']):
        if num==Problem1b.loc[row,'入院首次影像检查流水号']:
            return table.iloc[row1,range(2,33)]
pdtemp = pd.DataFrame(columns=table3_ED.columns[2:])
pdtemp1 = pd.DataFrame(columns=table3_Hemo.columns[2:])
for row in range(0,160):
    pdtemp.loc[row] = findRow(Problem1b,table3_ED,row)
    pdtemp1.loc[row] = findRow(Problem1b,table3_Hemo,row)

pdtemp.columns = pdtemp.columns + '_ED'
pdtemp1.columns = pdtemp1.columns + '_Hemo'
Problem1b = pd.concat([Problem1b,pdtemp,pdtemp1],axis=1)
Problem1b['target'] = Problem1a['isExpansion']
Problem1b['性别']=Problem1b['性别'].map({'男':1,'女':0}).astype(np.int8)
Problem1b.to_excel('Problem1b_1.xlsx')
display(Problem1b.head(5))
clear_output()
drop_columns = ['ID','90天mRS','数据集划分','入院首次影像检查流水号','血压']
display(Problem1b.drop(columns=drop_columns).head(5))
clear_output()
train_df = Problem1b.drop(columns=drop_columns).loc[range(0,100)]
test1_df = Problem1b.drop(columns=drop_columns).loc[range(100,130)]
test2_df = Problem1b.drop(columns=drop_columns).loc[range(130,160)]
display(train_df)
display(test1_df)
display(test2_df)
clear_output()
def summary_df(train_df,test1_df,test2_df):
    summary = pd.DataFrame(train_df.dtypes, columns=['dtypes'])
    summary['train_missing#'] = train_df.isna().sum()
    summary['train_missing%'] = (train_df.isna().sum())/len(train_df)
    summary['train_uniques'] = train_df.nunique().values
    summary['train_count'] = train_df.count().values
    # summary['train_skew'] = train_df.skew().values
    summary['test1_missing#'] = test1_df.isna().sum()
    summary['test1_missing%'] = (test1_df.isna().sum())/len(test1_df)
    summary['test1_uniques'] = test1_df.nunique().values
    summary['test1_count'] = test1_df.count().values
    # summary['test1_skew'] = test1_df.skew().values
    summary['test2_missing#'] = test2_df.isna().sum()
    summary['test2_missing%'] = (test2_df.isna().sum())/len(test2_df)
    summary['test2_uniques'] = test2_df.nunique().values
    summary['test2_count'] = test2_df.count().values
    # summary['test2_skew'] = test2_df.skew().values
    return summary
summary_df(train_df,test1_df,test2_df).style.background_gradient(cmap='Blues')
clear_output()
cat_cols = ['性别','脑出血前mRS评分','高血压病史','卒中病史','糖尿病史','房颤史','冠心病史','吸烟史','饮酒史']
num_cols = ['年龄', '发病到首次影像检查时间间隔', '收缩压','舒张压', 'HM_volume', 'HM_ACA_R_Ratio', 'HM_MCA_R_Ratio',
       'HM_PCA_R_Ratio', 'HM_Pons_Medulla_R_Ratio', 'HM_Cerebellum_R_Ratio',
       'HM_ACA_L_Ratio', 'HM_MCA_L_Ratio', 'HM_PCA_L_Ratio',
       'HM_Pons_Medulla_L_Ratio', 'HM_Cerebellum_L_Ratio', 'ED_volume',
       'ED_ACA_R_Ratio', 'ED_MCA_R_Ratio', 'ED_PCA_R_Ratio',
       'ED_Pons_Medulla_R_Ratio', 'ED_Cerebellum_R_Ratio', 'ED_ACA_L_Ratio',
       'ED_MCA_L_Ratio', 'ED_PCA_L_Ratio', 'ED_Pons_Medulla_L_Ratio',
       'ED_Cerebellum_L_Ratio', 'original_shape_Elongation_ED',
       'original_shape_Flatness_ED', 'original_shape_LeastAxisLength_ED',
       'original_shape_MajorAxisLength_ED',
       'original_shape_Maximum2DDiameterColumn_ED',
       'original_shape_Maximum2DDiameterRow_ED',
       'original_shape_Maximum2DDiameterSlice_ED',
       'original_shape_Maximum3DDiameter_ED', 'original_shape_MeshVolume_ED',
       'original_shape_MinorAxisLength_ED', 'original_shape_Sphericity_ED',
       'original_shape_SurfaceArea_ED', 'original_shape_SurfaceVolumeRatio_ED',
       'original_shape_VoxelVolume_ED',
       'NCCT_original_firstorder_10Percentile_ED',
       'NCCT_original_firstorder_90Percentile_ED',
       'NCCT_original_firstorder_Energy_ED',
       'NCCT_original_firstorder_Entropy_ED',
       'NCCT_original_firstorder_InterquartileRange_ED',
       'NCCT_original_firstorder_Kurtosis_ED',
       'NCCT_original_firstorder_Maximum_ED',
       'NCCT_original_firstorder_MeanAbsoluteDeviation_ED',
       'NCCT_original_firstorder_Mean_ED',
       'NCCT_original_firstorder_Median_ED',
       'NCCT_original_firstorder_Minimum_ED',
       'NCCT_original_firstorder_Range_ED',
       'NCCT_original_firstorder_RobustMeanAbsoluteDeviation_ED',
       'NCCT_original_firstorder_RootMeanSquared_ED',
       'NCCT_original_firstorder_Skewness_ED',
       'NCCT_original_firstorder_Uniformity_ED',
       'NCCT_original_firstorder_Variance_ED',
       'original_shape_Elongation_Hemo', 'original_shape_Flatness_Hemo',
       'original_shape_LeastAxisLength_Hemo',
       'original_shape_MajorAxisLength_Hemo',
       'original_shape_Maximum2DDiameterColumn_Hemo',
       'original_shape_Maximum2DDiameterRow_Hemo',
       'original_shape_Maximum2DDiameterSlice_Hemo',
       'original_shape_Maximum3DDiameter_Hemo',
       'original_shape_MeshVolume_Hemo', 'original_shape_MinorAxisLength_Hemo',
       'original_shape_Sphericity_Hemo', 'original_shape_SurfaceArea_Hemo',
       'original_shape_SurfaceVolumeRatio_Hemo',
       'original_shape_VoxelVolume_Hemo',
       'NCCT_original_firstorder_10Percentile_Hemo',
       'NCCT_original_firstorder_90Percentile_Hemo',
       'NCCT_original_firstorder_Energy_Hemo',
       'NCCT_original_firstorder_Entropy_Hemo',
       'NCCT_original_firstorder_InterquartileRange_Hemo',
       'NCCT_original_firstorder_Kurtosis_Hemo',
       'NCCT_original_firstorder_Maximum_Hemo',
       'NCCT_original_firstorder_MeanAbsoluteDeviation_Hemo',
       'NCCT_original_firstorder_Mean_Hemo',
       'NCCT_original_firstorder_Median_Hemo',
       'NCCT_original_firstorder_Minimum_Hemo',
       'NCCT_original_firstorder_Range_Hemo',
       'NCCT_original_firstorder_RobustMeanAbsoluteDeviation_Hemo',
       'NCCT_original_firstorder_RootMeanSquared_Hemo',
       'NCCT_original_firstorder_Skewness_Hemo',
       'NCCT_original_firstorder_Uniformity_Hemo',
       'NCCT_original_firstorder_Variance_Hemo']
method_cols =  ['脑室引流', '止血治疗', '降颅压治疗', '降压治疗', '镇静、镇痛治疗',
       '止吐护胃', '营养神经']
target = 'target'
feature = cat_cols + num_cols + method_cols


def chi_squared_test(df, input_var, target_var, significance_level=0.05):
    contingency_table = pd.crosstab(pd.DataFrame(df[input_var], columns=input_var), df[target_var])
    chi2, p, _, _ = stats.chi2_contingency(contingency_table)

    if p < significance_level:
        print(f'\033[32m{input_var} has a significant relationship with the target variable.\033[0m')
    else:
        print(f'\033[31m{input_var} does not have a significant relationship with the target variable.\033[0m')


for cat_col in cat_cols:
    chi_squared_test(train_df, cat_col, target)