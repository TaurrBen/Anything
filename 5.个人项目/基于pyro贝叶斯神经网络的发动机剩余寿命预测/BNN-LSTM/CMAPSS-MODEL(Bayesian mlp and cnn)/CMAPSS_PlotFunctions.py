import math

import numpy
import pyro
import torch
import pandas as pd
import matplotlib.pyplot as plt

import CMAPSS_Dataloader
import numpy
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import CMAPSS_Dataloader
from sklearn.metrics import mean_squared_error
import CMAPSS_CONFIG as config

def train_actual_predicted(read_path, sub_dataset, window_size, max_life, train_output, alpha_grid, alpha_low,
                           alpha_high, _COLORS):
    train_data = CMAPSS_Dataloader.load_dataset(read_path, sub_dataset, max_life, train=True, test=False,
                                                finaltest=False)
    train_data_gp = train_data.groupby('engine_id', sort=False)
    train_data_reduce = []

    for engine_id, data in train_data_gp:
        data = data.iloc[window_size - 1:, :]
        train_data_reduce.append(data)

    train_data = pd.concat(train_data_reduce, ignore_index=True)


    train_data['Predicted RUL'] = train_output[0]
    train_data['Predicted RUL Std'] = train_output[1]
    train_data_rul_stack = torch.stack(train_output[2])
    train_data['Predicted RUL1'] = torch.mean(train_data_rul_stack,1)

    train_rul_group = train_data.groupby('engine_id', sort=False)
    max_plots = []

    for engine_id, data in train_rul_group:
        actual_rul = data[['cycle', 'RUL']]
        actual_rul = actual_rul.set_index('cycle')
        if config.transform_ln:
            actual_rul['RUL'] = actual_rul['RUL'].apply(lambda x: math.exp(x) - 1)
        predicted_rul1 = data[['cycle', 'Predicted RUL']]
        predicted_rul1 = predicted_rul1.set_index('cycle')
        predicted_rul = data[['cycle', 'Predicted RUL']]
        predicted_rul = predicted_rul.set_index('cycle')
        if config.transform_ln:
            predicted_rul['Predicted RUL'] = predicted_rul['Predicted RUL'].apply(lambda x: math.exp(x) - 1)
        predicted_rul_std = data[['cycle', 'Predicted RUL Std']]
        predicted_rul_std = predicted_rul_std.set_index('cycle')
        predicted_rul_std.columns = ['Predicted RUL']


        fig = plt.figure(figsize=(7, 7))
        ax = plt.subplot(1, 1, 1)
        plt.grid(visible=True, which="major", axis="both", alpha=alpha_grid)
        plt.xlabel("Time (Cycle)", fontsize=24)
        plt.ylabel("RUL", fontsize=24)
        plt.xlim(data['cycle'].min(), data['cycle'].max())
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.plot(predicted_rul, color=_COLORS[0], alpha=alpha_high, linestyle="solid", linewidth=4,
                 label='Predicted RUL')
        plt.plot(actual_rul, color=_COLORS[1], alpha=alpha_high, linestyle="dashed", linewidth=4, label='Actual RUL')
        for i, factor in enumerate([1, 2, 3]):
            upper_rul = predicted_rul1 + factor * predicted_rul_std
            lower_rul = predicted_rul1 - factor * predicted_rul_std
            if config.transform_ln:
                upper_rul['Predicted RUL'] = upper_rul['Predicted RUL'].apply(lambda x: math.exp(x) - 1)
                lower_rul['Predicted RUL'] = lower_rul['Predicted RUL'].apply(lambda x: math.exp(x) - 1)
            plt.fill_between(data['cycle'], y1=lower_rul['Predicted RUL'].tolist(), y2=upper_rul['Predicted RUL'].tolist(),
                             color=_COLORS[2], alpha=alpha_low, label=str(factor)+'-sigma confidence')
        ax.set_title('FD' + sub_dataset + " Train Engine Unit-" + str(engine_id), fontsize=26)
        ax.legend(loc="upper right", fontsize=16)
        max_plots.append(engine_id)

        if len(max_plots) >= 1:
            break

        fig.tight_layout()
        plt.savefig('./image/train'+str(engine_id)+'.png')
    plt.show()
    return None


def test_actual_predicted(read_path, sub_dataset, window_size, max_life, test_output, alpha_grid, alpha_low, alpha_high,
                          _COLORS):
    test_data = CMAPSS_Dataloader.load_dataset(read_path, sub_dataset, max_life, train=False, test=True,
                                               finaltest=False)
    test_data_gp = test_data.groupby('engine_id', sort=False)
    test_data_reduce = []

    for engine_id, data in test_data_gp:
        data = data.iloc[window_size - 1:, :]
        test_data_reduce.append(data)

    test_data = pd.concat(test_data_reduce, ignore_index=True)

    test_data['Predicted RUL'] = test_output[0]
    test_data['Predicted RUL Std'] = test_output[1]
    test_data_rul_stack = torch.stack(test_output[2])
    test_data['Predicted RUL1'] = torch.mean(test_data_rul_stack, 1)

    test_rul_group = test_data.groupby('engine_id', sort=False)
    max_plots = []

    for engine_id, data in test_rul_group:
        actual_rul = data[['cycle', 'RUL']]
        actual_rul = actual_rul.set_index('cycle')
        if config.transform_ln:
            actual_rul['RUL'] = actual_rul['RUL'].apply(lambda x: math.exp(x) - 1)
        predicted_rul1 = data[['cycle', 'Predicted RUL']]
        predicted_rul1 = predicted_rul1.set_index('cycle')
        predicted_rul = data[['cycle', 'Predicted RUL']]
        predicted_rul = predicted_rul.set_index('cycle')
        if config.transform_ln:
            predicted_rul['Predicted RUL'] = predicted_rul['Predicted RUL'].apply(lambda x: math.exp(x) - 1)
        predicted_rul_std = data[['cycle', 'Predicted RUL Std']]
        predicted_rul_std = predicted_rul_std.set_index('cycle')
        predicted_rul_std.columns = ['Predicted RUL']

        fig = plt.figure(figsize=(7, 7))
        ax = plt.subplot(1, 1, 1)
        plt.grid(visible=True, which="major", axis="both", alpha=alpha_grid)
        plt.xlabel("Time (Cycle)", fontsize=24)
        plt.ylabel("RUL", fontsize=24)
        plt.xlim(data['cycle'].min(), data['cycle'].max())
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.plot(predicted_rul, color=_COLORS[0], alpha=alpha_high, linestyle="solid", linewidth=4,
                 label='Predicted RUL')
        plt.plot(actual_rul, color=_COLORS[1], alpha=alpha_high, linestyle="dashed", linewidth=4, label='Actual RUL')
        for i, factor in enumerate([1, 2, 3]):
            upper_rul = predicted_rul1 + factor * predicted_rul_std
            lower_rul = predicted_rul1 - factor * predicted_rul_std
            if config.transform_ln:
                upper_rul['Predicted RUL'] = upper_rul['Predicted RUL'].apply(lambda x: math.exp(x) - 1)
                lower_rul['Predicted RUL'] = lower_rul['Predicted RUL'].apply(lambda x: math.exp(x) - 1)
            plt.fill_between(data['cycle'], y1=lower_rul['Predicted RUL'].tolist(), y2=upper_rul['Predicted RUL'].tolist(),
                             color=_COLORS[2], alpha=alpha_high, label=str(factor)+'-sigma confidence')
        ax.set_title('FD' + sub_dataset + " Test Engine Unit-" + str(engine_id), fontsize=26)
        ax.legend(loc="upper right", fontsize=16)
        max_plots.append(engine_id)

        if len(max_plots) >= 1:
            break

        fig.tight_layout()
        plt.savefig('./image/test'+str(engine_id)+'.png')
    plt.show()
    return None


def loss_plot(sub_dataset, train_loss_epoch, test_loss_epoch, alpha_grid, alpha_low, alpha_high, _COLORS):
    fig = plt.figure(figsize=(8, 4))
    ax = plt.subplot(1, 2, 1)
    ax.grid(visible=True, which="both", axis="both", alpha=alpha_grid)
    ax.set_xlabel('Epochs', fontsize=24)
    ax.set_ylabel('ELBO', fontsize=24)
    ax.plot(train_loss_epoch, color=_COLORS[0], alpha=alpha_low, linestyle="solid", label='Train Loss')
    ax.legend(loc="upper right", fontsize=10)
    ax = plt.subplot(1, 2, 2)
    ax.grid(visible=True, which="both", axis="both", alpha=alpha_grid)
    ax.set_xlabel('Epochs', fontsize=24)
    ax.set_ylabel('RMSE', fontsize=24)
    ax.plot(test_loss_epoch, color=_COLORS[1], alpha=alpha_low, linestyle="solid", label='Test Loss')
    ax.legend(loc="upper right", fontsize=10)
    fig.tight_layout()
    plt.show()
    return None

def score_func(pred_rul, actual_rul):
    Si = []
    for i in range(0, pred_rul.shape[0]):
        di = pred_rul[i] - actual_rul[i]

        if di < 0:
            inter = torch.exp((-di) / 13) - 1
        else:
            inter = torch.exp(di / 10) - 1
        Si.append(inter)
    s = torch.sum(torch.stack(Si))
    return s

def param_plot(model,guide,finaltest_loader,num_samples=100):

    return_sites = ['fc1.weight','fc1.bias',
                   'lstm.weight_ih_l0','lstm.weight_hh_l0',
                   'lstm.bias_ih_l0','lstm.bias_hh_l0',
                   'lstm.weight_ih_l0_reverse', 'lstm.weight_hh_l0_reverse',
                   'lstm.bias_ih_l0_reverse', 'lstm.bias_hh_l0_reverse',
                   'linearSeq.0.weight','linearSeq.0.bias',
                   'linearSeq.2.weight','linearSeq.2.bias']

    return_sites = ['linearSeq.2.weight']
    predictive = pyro.infer.Predictive(model=model, guide=guide, num_samples=num_samples,return_sites=return_sites)
    for i, (data_fte) in enumerate(finaltest_loader):
        samples = predictive(data_fte.float())
        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(1, 1, 1)
        ax.grid(visible=True, which="both", axis="both", alpha=0.4)
        ax.set_ylabel("Prob", fontsize=12)
        ##第四个维度-1自己改得到不同的
        sns.histplot(data=samples['linearSeq.2.weight'][:,-1,-1,-1].view(num_samples,1), ax=ax, kde=True, stat='probability', label='Prob dist')
        ax.legend(loc="upper right", fontsize=10)
        plt.show()
        break


import CMAPSS_Dataloader
from sklearn.metrics import mean_squared_error


def fianltest_actual_vs_predicted(read_path, sub_dataset, window_size, max_life, finaltest_output, alpha_grid,
                                  alpha_low, alpha_high, _COLORS):
    finaltest_data = CMAPSS_Dataloader.load_dataset(read_path, sub_dataset, max_life, train=False, test=False,
                                                    finaltest=True)
    finaltest_data_groupped = finaltest_data.groupby('engine_id', sort=False)
    finaltest_data_reduced = []

    for engine_id, data in finaltest_data_groupped:
        data = data.iloc[window_size - 1:, :]
        finaltest_data_reduced.append(data)

    finaltest_data = pd.concat(finaltest_data_reduced, ignore_index=True)

    finaltest_data['Predicted RUL'] = finaltest_output[0]
    finaltest_data['Predicted RUL Std'] = finaltest_output[1]
    finaltest_data_rul_stack = torch.stack(finaltest_output[2])
    finaltest_data['Predicted RUL1'] = torch.mean(finaltest_data_rul_stack, 1)

    finaltest_rul_group = finaltest_data.groupby('engine_id', sort=False)
    plot_ids = []
    if sub_dataset == "001":
        # plot_ids = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 24, 25, 26, 27, 28, 29, 30, 31,
        #             32, 33, 35, 37, 39, 41, 42, 43, 44, 45, 46, 47, 51, 52, 53, 57, 60, 61, 63, 66, 70, 74, 80, 81, 92,
        #             83, 90, 91, 93, 92, 94]
        plot_ids = [24]
    if sub_dataset == "002":
        plot_ids = [5]
    if sub_dataset == "003":
        plot_ids = [3]
    if sub_dataset == "004":
        plot_ids = [32]

    for engine_id, data in finaltest_rul_group:
        if engine_id in plot_ids:
            actual_rul = data[['cycle', 'RUL']]
            actual_rul = actual_rul.set_index('cycle')
            if config.transform_ln:
                actual_rul['RUL'] = actual_rul['RUL'].apply(lambda x: math.exp(x) - 1)
            predicted_rul1 = data[['cycle', 'Predicted RUL']]
            predicted_rul1 = predicted_rul1.set_index('cycle')
            predicted_rul = data[['cycle', 'Predicted RUL']]
            predicted_rul = predicted_rul.set_index('cycle')
            if config.transform_ln:
                predicted_rul['Predicted RUL'] = predicted_rul['Predicted RUL'].apply(lambda x: math.exp(x) - 1)
            predicted_rul_std = data[['cycle', 'Predicted RUL Std']]
            predicted_rul_std = predicted_rul_std.set_index('cycle')
            predicted_rul_std.columns = ['Predicted RUL']
            #########
            index = data.index
            finaltest_data_rul = finaltest_data_rul_stack[index[-1]]
            if config.transform_ln:
                finaltest_data_rul = torch.exp(finaltest_data_rul) - 1

            fig = plt.figure(figsize=(12, 12))
            plt.subplots_adjust(wspace=0.3, hspace=0.3)
            ax = fig.add_subplot(1, 1, 1)
            ax.grid(visible=True, which="both", axis="both", alpha=alpha_grid)
            ax.set_ylabel("RUL", fontsize=12)
            ax.set_xlim(data['cycle'].min(), data['cycle'].max())
            ax.plot(predicted_rul, color=_COLORS[0], alpha=alpha_high, linestyle="solid", linewidth=4,
                    label='Predicted RUL')
            ax.plot(actual_rul, color=_COLORS[1], alpha=alpha_high, linestyle="dashed", linewidth=4, label='Actual RUL')
            for i, factor in enumerate([1, 2, 3]):
                upper_rul = predicted_rul1 + factor * predicted_rul_std
                lower_rul = predicted_rul1 - factor * predicted_rul_std
                if config.transform_ln:
                    upper_rul['Predicted RUL'] = upper_rul['Predicted RUL'].apply(lambda x: math.exp(x) - 1)
                    lower_rul['Predicted RUL'] = lower_rul['Predicted RUL'].apply(lambda x: math.exp(x) - 1)
                ax.fill_between(data['cycle'], y1=lower_rul['Predicted RUL'].tolist(),
                                y2=upper_rul['Predicted RUL'].tolist()
                                , alpha=alpha_low, label=str(factor) + '-sigma confidence')
            ax.set_title('FD' + sub_dataset + " Final Test Engine Unit-" + str(engine_id), fontsize=12)
            ax.legend(loc="upper right", fontsize=10)
            fig.tight_layout()
            ##########
            fig1 = plt.figure(figsize=(4, 5))
            ax = fig1.add_subplot(1, 1, 1)
            ax.grid(visible=True, which="both", axis="both", alpha=alpha_grid)
            ax.set_ylabel("Prob", fontsize=12)
            sns.histplot(data=finaltest_data_rul,ax=ax,kde=True,stat='probability',label='Prob dist')
            ax.vlines(actual_rul['RUL'].iloc[-1],ymin=0,ymax=1,label='Actual RUL')
            ax.vlines(predicted_rul['Predicted RUL'].iloc[-1],ymin=0,ymax=1,label='Predicted RUL')
            ax.legend(loc="upper right", fontsize=10)
            plt.show()
    rul_score = []
    true_rul = []
    #not transform
    rul_mean = []
    rul_std = []
    for engine_id, data in finaltest_rul_group:
        true_last_rul = data['RUL'].iloc[-1]
        last_rul = data["Predicted RUL"].iloc[-1]
        last_rul_std = data['Predicted RUL Std'].iloc[-1]

        if config.transform_ln:
            rul_score.append(math.exp(last_rul) - 1)
            true_rul.append(math.exp(true_last_rul) - 1)
        else:
            rul_score.append(last_rul)
            true_rul.append(true_last_rul)
        rul_mean.append(last_rul)
        rul_std.append(last_rul_std)


    # true_rul = pd.read_csv(read_path + "RUL_FD" + sub_dataset + ".txt", delimiter="\s+", header=None)
    # true_rul.columns = ['RUL']
    # ture_rul = torch.FloatTensor(true_rul['RUL'].values)

    rul_df = pd.concat([pd.Series(true_rul),pd.Series(rul_mean), pd.Series(rul_std)], axis=True, ignore_index=False)
    rul_df.columns = ['RUL', 'Predicted RUL', 'Predicted RUL Std']
    rul_df_sort = rul_df.sort_values(by=["RUL"],ascending=True).reset_index()

    ###100 sorted res
    fig = plt.figure(figsize=(18, 8))
    ax = fig.add_subplot(1, 1, 1)
    plt.grid(visible=True, which="major", axis="both", alpha=alpha_grid)
    plt.xlabel("id", fontsize=24)
    plt.ylabel("RUL", fontsize=24)
    for i, factor in enumerate([1, 2, 3]):
        actual_rul = rul_df_sort[['RUL']]
        if config.transform_ln:
            actual_rul['RUL'] = actual_rul['RUL'].apply(lambda x: math.exp(x) - 1)
        predicted_rul1 = rul_df_sort[['Predicted RUL']]
        predicted_rul = rul_df_sort[['Predicted RUL']]
        if config.transform_ln:
            predicted_rul['Predicted RUL'] = predicted_rul['Predicted RUL'].apply(lambda x: math.exp(x) - 1)
        predicted_rul_std = rul_df_sort[['Predicted RUL Std']]
        predicted_rul_std.columns = ['Predicted RUL']
        upper_rul = predicted_rul1 + factor * predicted_rul_std
        lower_rul = predicted_rul1 - factor * predicted_rul_std
        if config.transform_ln:
            upper_rul['Predicted RUL'] = upper_rul['Predicted RUL'].apply(lambda x: math.exp(x) - 1)
            lower_rul['Predicted RUL'] = lower_rul['Predicted RUL'].apply(lambda x: math.exp(x) - 1)
        ax.plot(actual_rul, linewidth=4, label='Actual RUL')
        ax.plot(predicted_rul, linewidth=4, label='Predicted RUL')
        ax.plot(upper_rul['Predicted RUL'].tolist(), linewidth=4,label='Upper')
        ax.plot(lower_rul['Predicted RUL'].tolist(), linewidth=4, label='Lower')
        # ax.set_title('FD' + sub_dataset + " Test Engine Unit-" + str(engine_id), fontsize=26)
        ax.legend(loc="upper right", fontsize=16)
        plt.show()


    rul_socre = torch.FloatTensor(rul_score)
    ture_rul = torch.FloatTensor(true_rul)
    mse = mean_squared_error(ture_rul.numpy(), rul_socre.numpy())
    rmse = np.sqrt(mse)
    print("mse", mse, "rmse", rmse)
    score = score_func(rul_socre, ture_rul)
    print("The Final Score is:", score.item())
    score = score_func(rul_socre, ture_rul)
    print("The Final Score is:", score.item())

    return score


