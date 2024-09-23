import math

import numpy
import torch
import pandas as pd
import matplotlib.pyplot as plt
import CMAPSS_Dataloader

def train_actual_predicted(read_path, sub_dataset, window_size, max_life, fusion, uncertainty, train_output, alpha_grid, alpha_low, alpha_high, _COLORS):
    
    train_data = CMAPSS_Dataloader.load_dataset(read_path, sub_dataset, max_life, fusion, train=True, test=False, finaltest=False)
    train_data_gp = train_data.groupby('engine_id', sort=False)
    train_data_reduce = []
    
    for engine_id, data in train_data_gp:
        data = data.iloc[window_size-1:,:]
        train_data_reduce.append(data)
        
    train_data = pd.concat(train_data_reduce, ignore_index=True)
    
    train_data['Predicted RUL'] = train_output[0]
    train_data['Predicted RUL Std'] = train_output[1]#[0:train_data.shape[0]]
    if uncertainty:
        train_data['Predicted RUL Epistemic Uncertainty'] = train_output[2]#[0:train_data.shape[0]]
        train_data['Predicted RUL Aleatoric Uncertainty'] = train_output[3]#[0:train_data.shape[0]]
    
    train_rul_group = train_data.groupby('engine_id', sort = False)
    max_plots=[]
        
    for engine_id, data in train_rul_group:
        actual_rul = data[['cycle', 'RUL']]
        actual_rul = actual_rul.set_index('cycle')
        # actual_rul['RUL'] = actual_rul['RUL'].apply(lambda x:math.exp(x)-1)
        predicted_rul1 = data[['cycle','Predicted RUL']]
        predicted_rul1 = predicted_rul1.set_index('cycle')
        predicted_rul = data[['cycle','Predicted RUL']]
        predicted_rul = predicted_rul.set_index('cycle')
        # predicted_rul['Predicted RUL'] = predicted_rul['Predicted RUL'].apply(lambda x: math.exp(x) - 1)
        predicted_rul_std = data[['cycle', 'Predicted RUL Std']]
        # if uncertainty:
        #     predicted_rul_std = data[['cycle', 'Predicted RUL Aleatoric Uncertainty']]
        predicted_rul_std = predicted_rul_std.set_index('cycle')
        predicted_rul_std.columns = ['Predicted RUL']
        factor = 1
        upper_rul = predicted_rul1 + factor * predicted_rul_std
        lower_rul = predicted_rul1 - factor * predicted_rul_std
        # upper_rul['Predicted RUL'] = upper_rul['Predicted RUL'].apply(lambda x: math.exp(x) - 1)
        # lower_rul['Predicted RUL'] = lower_rul['Predicted RUL'].apply(lambda x: math.exp(x) - 1)

        fig=plt.figure(figsize=(7,7))
        ax=plt.subplot(1,1,1)
        plt.grid(visible=True, which="major", axis="both", alpha=alpha_grid)
        plt.xlabel("Time (Cycle)", fontsize = 24)
        plt.ylabel("RUL", fontsize = 24)
        plt.xlim(data['cycle'].min(), data['cycle'].max())
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.plot(predicted_rul, color=_COLORS[0], alpha=alpha_high, linestyle="solid",linewidth=4, label='Predicted RUL')
        plt.plot(actual_rul, color=_COLORS[1], alpha=alpha_high, linestyle="dashed", linewidth=4, label='Actual RUL')
        plt.fill_between(data['cycle'],y1=lower_rul['Predicted RUL'].tolist(), y2=upper_rul['Predicted RUL'].tolist(), color=_COLORS[2], alpha=alpha_high, label=str(factor)+'-sigma confidence')
        ax.set_title('FD'+sub_dataset +" Train Engine Unit-" + str(engine_id), fontsize = 26)
        ax.legend(loc="upper right", fontsize = 16)
        max_plots.append(engine_id)
        
        if len(max_plots) >= 1:
            break

        fig.tight_layout()
    plt.show()
    return None

def test_actual_predicted(read_path, sub_dataset, window_size, max_life, fusion, uncertainty, test_output, alpha_grid, alpha_low, alpha_high, _COLORS):
    
    test_data = CMAPSS_Dataloader.load_dataset(read_path, sub_dataset, max_life, fusion, train=False, test=True, finaltest=False)
    test_data_gp = test_data.groupby('engine_id', sort=False)
    test_data_reduce = []
    
    for engine_id, data in test_data_gp:
        data = data.iloc[window_size-1:,:]
        test_data_reduce.append(data)
        
    test_data = pd.concat(test_data_reduce, ignore_index=True)
    
    test_data['Predicted RUL'] = test_output[0]#[0:test_data.shape[0]]
    test_data['Predicted RUL Std'] = test_output[1]#[0:test_data.shape[0]]
    if uncertainty:
        test_data['Predicted RUL Epistemic Uncertainty'] = test_output[2]#[0:test_data.shape[0]]
        test_data['Predicted RUL Aleatoric Uncertainty'] = test_output[3]#[0:test_data.shape[0]]

    test_rul_group = test_data.groupby('engine_id', sort = False)
    max_plots=[]
        
    for engine_id, data in test_rul_group:
        actual_rul = data[['cycle', 'RUL']]
        actual_rul = actual_rul.set_index('cycle')
        # actual_rul['RUL'] = actual_rul['RUL'].apply(lambda x: math.exp(x) - 1)
        predicted_rul1 = data[['cycle', 'Predicted RUL']]
        predicted_rul1 = predicted_rul1.set_index('cycle')
        predicted_rul = data[['cycle', 'Predicted RUL']]
        predicted_rul = predicted_rul.set_index('cycle')
        # predicted_rul['Predicted RUL'] = predicted_rul['Predicted RUL'].apply(lambda x: math.exp(x) - 1)
        predicted_rul_std = data[['cycle', 'Predicted RUL Std']]
        # if uncertainty:
        #     predicted_rul_std = data[['cycle', 'Predicted RUL Aleatoric Uncertainty']]
        predicted_rul_std = predicted_rul_std.set_index('cycle')
        predicted_rul_std.columns = ['Predicted RUL']
        factor = 1
        upper_rul = predicted_rul1 + factor * predicted_rul_std
        lower_rul = predicted_rul1 - factor * predicted_rul_std
        # upper_rul['Predicted RUL'] = upper_rul['Predicted RUL'].apply(lambda x: math.exp(x) - 1)
        # lower_rul['Predicted RUL'] = lower_rul['Predicted RUL'].apply(lambda x: math.exp(x) - 1)
        
        fig=plt.figure(figsize=(7,7))
        ax=plt.subplot(1,1,1)
        plt.grid(visible=True, which="major", axis="both", alpha=alpha_grid)
        plt.xlabel("Time (Cycle)", fontsize = 24)
        plt.ylabel("RUL", fontsize = 24)
        plt.xlim(data['cycle'].min(), data['cycle'].max())
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.plot(predicted_rul, color=_COLORS[0], alpha=alpha_high, linestyle="solid",linewidth=4, label='Predicted RUL')
        plt.plot(actual_rul, color=_COLORS[1], alpha=alpha_high, linestyle="dashed", linewidth=4, label='Actual RUL')
        plt.fill_between(data['cycle'],y1=lower_rul['Predicted RUL'].tolist(), y2=upper_rul['Predicted RUL'].tolist(), color=_COLORS[2], alpha=alpha_high, label=str(factor)+'-sigma confidence')
        ax.set_title('FD'+sub_dataset +" Test Engine Unit-" + str(engine_id), fontsize = 26)
        ax.legend(loc="upper right", fontsize = 16)
        max_plots.append(engine_id)
        
        if len(max_plots) >= 1:
            break

        fig.tight_layout()
    plt.show()
    return None

def loss_plot(sub_dataset, train_loss_epoch, test_loss_epoch, alpha_grid, alpha_low, alpha_high, _COLORS):
   
   fig=plt.figure(figsize=(7,7))
   plt.grid(visible=True, which="both", axis="both", alpha=alpha_grid)
   plt.xlabel('Epochs', fontsize=24)
   plt.ylabel('RMSE', fontsize=24)
   plt.xticks(fontsize=18)
   plt.yticks(fontsize=18)
   plt.plot(train_loss_epoch, color=_COLORS[0], alpha=alpha_low, linestyle="solid", label='Train Loss')
   plt.plot(test_loss_epoch, color=_COLORS[1], alpha=alpha_low, linestyle="solid", label='Test Loss')            
   fig.tight_layout()
   plt.show()
   return None
  
def score_func(pred_rul, actual_rul):
    Si = []
    for i in range(0,pred_rul.shape[0]):
        di = pred_rul[i] - actual_rul[i]
        
        if di < 0:
            inter = torch.exp((-di)/13) - 1
        else:
            inter = torch.exp(di/10) - 1
        Si.append(inter)
    s = torch.sum(torch.stack(Si))
    return s

def fianltest_actual_vs_predicted(read_path, sub_dataset, window_size, max_life, fusion, uncertainty, finaltest_output, alpha_grid,
                                  alpha_low, alpha_high, _COLORS, finalte_dataset):
    
    finaltest_data = CMAPSS_Dataloader.load_dataset(read_path, sub_dataset,max_life, fusion, train=False, test=False, finaltest=True)
    finaltest_data_groupped = finaltest_data.groupby('engine_id', sort = False)
    finaltest_data_reduced = []

    for engine_id, data in finaltest_data_groupped:  
        data = data.iloc[window_size-1 : , :]
        finaltest_data_reduced.append(data)

    finaltest_data = pd.concat(finaltest_data_reduced, ignore_index=True)
    finaltest_data['Predicted RUL'] =  finaltest_output[0]#[0:finaltest_data.shape[0]]
    finaltest_data['Predicted RUL Std'] = finaltest_output[1]#[0:finaltest_data.shape[0]]
    if uncertainty:
        finaltest_data['Predicted RUL Epistemic Uncertainty'] = finaltest_output[2]#[0:finaltest_data.shape[0]]
        finaltest_data['Predicted RUL Aleatoric Uncertainty'] = finaltest_output[3]#[0:finaltest_data.shape[0]]
        finaltest_data['Predicted RUL Total Uncertainty'] = finaltest_data['Predicted RUL Epistemic Uncertainty'] + finaltest_data['Predicted RUL Aleatoric Uncertainty']
    
    finaltest_rul_group = finaltest_data.groupby('engine_id', sort = False)
    
    if sub_dataset == "001":
        plot_ids=[24]
    if sub_dataset == "002":
        plot_ids=[5]
    if sub_dataset == "003":
        plot_ids=[3]
    if sub_dataset == "004":
        plot_ids=[32]

    for engine_id, data in finaltest_rul_group:
        if engine_id in plot_ids:
            actual_rul = data[['cycle', 'RUL']]
            actual_rul = actual_rul.set_index('cycle')
            # actual_rul['RUL'] = actual_rul['RUL'].apply(lambda x: math.exp(x) - 1)
            predicted_rul1 = data[['cycle','Predicted RUL']]
            predicted_rul1 = predicted_rul1.set_index('cycle')
            predicted_rul = data[['cycle', 'Predicted RUL']]
            predicted_rul = predicted_rul.set_index('cycle')
            # predicted_rul['Predicted RUL'] = predicted_rul['Predicted RUL'].apply(lambda x: math.exp(x) - 1)

            Epistemic_Uncertainty = data[['cycle', 'Predicted RUL Epistemic Uncertainty']]
            Epistemic_Uncertainty = Epistemic_Uncertainty.set_index('cycle')
            Aleatoric_Uncertainty = data[['cycle', 'Predicted RUL Aleatoric Uncertainty']]
            Aleatoric_Uncertainty = Aleatoric_Uncertainty.set_index('cycle')
            Total_Uncertainty = data[['cycle', 'Predicted RUL Total Uncertainty']]
            Total_Uncertainty = Total_Uncertainty.set_index('cycle')
            predicted_rul_std = data[['cycle', 'Predicted RUL Std']]
            # if uncertainty:
            #     predicted_rul_std = data[['cycle', 'Predicted RUL Aleatoric Uncertainty']]
            predicted_rul_std = predicted_rul_std.set_index('cycle')
            predicted_rul_std.columns = ['Predicted RUL']

            fig=plt.figure(figsize=(12,12))
            plt.subplots_adjust(wspace=0.3, hspace=0.3)
            ax=fig.add_subplot(4,1,1)
            ax.grid(visible=True, which="both", axis="both", alpha=alpha_grid)
            ax.set_ylabel("RUL",fontsize=12)
            ax.set_xlim(data['cycle'].min(), data['cycle'].max())
            ax.plot(predicted_rul, color=_COLORS[0], alpha=alpha_high, linestyle="solid", linewidth=4, label='Predicted RUL')
            ax.plot(actual_rul, color=_COLORS[1], alpha=alpha_high, linestyle="dashed", linewidth=4, label='Actual RUL')
            # for i,factor in enumerate([1]):
            #     upper_rul = predicted_rul1 + factor * predicted_rul_std
            #     lower_rul = predicted_rul1 - factor * predicted_rul_std
            #     upper_rul['Predicted RUL'] = upper_rul['Predicted RUL'].apply(lambda x: math.exp(x) - 1)
            #     lower_rul['Predicted RUL'] = lower_rul['Predicted RUL'].apply(lambda x: math.exp(x) - 1)
            #     ax.fill_between(data['cycle'],y1=lower_rul['Predicted RUL'].tolist(), y2=upper_rul['Predicted RUL'].tolist()
            #                     , alpha=alpha_low, label=str(factor)+'-sigma confidence')
            #     print("upper:lower",upper_rul,lower_rul)
            ax.set_title('FD'+sub_dataset+" Final Test Engine Unit-" + str(engine_id), fontsize=12)
            ax.legend(loc="upper right", fontsize = 10)
            ##################
            ax=fig.add_subplot(4,1,2)
            ax.grid(visible=True, which="both", axis="both", alpha=alpha_grid)
            ax.set_ylabel("RUL EU", fontsize=12)
            ax.set_xlim(data['cycle'].min(), data['cycle'].max())
            ax.plot(Epistemic_Uncertainty, color=_COLORS[0], alpha=alpha_high, linestyle="solid", linewidth=4,
                     label='Predicted RUL Epistemic Uncertainty')
            ax.legend(loc="upper right", fontsize=10)
            ##################
            ax=fig.add_subplot(4,1,3)
            ax.grid(visible=True, which="both", axis="both", alpha=alpha_grid)
            ax.set_ylabel("RUL AU", fontsize=12)
            ax.set_xlim(data['cycle'].min(), data['cycle'].max())
            ax.plot(Aleatoric_Uncertainty, color=_COLORS[0], alpha=alpha_high, linestyle="solid", linewidth=4,
                     label='Predicted RUL Aleatoric Uncertainty')
            ax.legend(loc="upper right", fontsize=10)
            ##################
            ax = fig.add_subplot(4, 1, 4)
            ax.grid(visible=True, which="both", axis="both", alpha=alpha_grid)
            ax.set_xlabel("Time (Cycle)", fontsize=12)
            ax.set_ylabel("RUL TU", fontsize=12)
            ax.set_xlim(data['cycle'].min(), data['cycle'].max())
            ax.plot(Total_Uncertainty, color=_COLORS[0], alpha=alpha_high, linestyle="solid", linewidth=4,
                    label='Predicted RUL Total Uncertainty')
            ax.legend(loc="upper right", fontsize=10)
            plt.show()

    rul_score = []
    for engine_id, data in finaltest_rul_group:
        last_rul = data["Predicted RUL"].iloc[-1]
        rul_score.append(math.exp(last_rul)-1)


    true_rul = pd.read_csv(read_path + "RUL_FD"+sub_dataset + ".txt",delimiter="\s+", header=None)
    true_rul.columns = ['RUL']
    ture_rul = torch.FloatTensor(true_rul['RUL'].values)
    rul_socre =  torch.FloatTensor(rul_score)
    print("The Final rmse is:",torch.sqrt(torch.nn.MSELoss()(rul_socre, ture_rul)).item())
    score = score_func(rul_socre, ture_rul)
    print("The Final Score is:", score.item())
     
    return score


