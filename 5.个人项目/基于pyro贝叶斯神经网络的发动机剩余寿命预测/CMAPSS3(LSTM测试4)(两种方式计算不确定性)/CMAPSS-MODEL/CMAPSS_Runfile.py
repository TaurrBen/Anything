import pyro

import CMAPSS_Dataloader
import CMAPSS_CNN
import CMAPSS_TrainLoop
import CMAPSS_PlotFunctions

sub_dataset="001"
window_size=30
max_life=125#125
scaler="mm" #"mm" max-min ;"ss"standard
scaler_range=(-1,1)
shuffle=True
fusion=True
uncertainty=True
batch_size=512#适用于你的内存大小
num_iter = 1
alpha_grid=0.4
alpha_low=0.6
alpha_high=0.9
_COLORS=["green", "teal","pink"]
read_path = r"..\C-MAPSS-Data/"

train_loader, test_loader, unshuffle_train_loader, finaltest_loader, tr_dataset, te_dataset, finalte_dataset=CMAPSS_Dataloader.LoadCMAPSS(read_path, sub_dataset,
                                                                                                                                           window_size, max_life, scaler,
                                                                                                                                           scaler_range, shuffle,
                                                                                                                                           batch_size,fusion)

# train_loss_epoch, test_loss_epoch, train_output, test_output, finaltest_output = CMAPSS_CNN.train(train_loader, test_loader, unshuffle_train_loader, finaltest_loader, num_iter)
# net = CMAPSS_CNN.MLP
net = CMAPSS_CNN.MLP
if fusion:
    net = CMAPSS_CNN.LSTM
model = CMAPSS_CNN.BayesianCNN(net)
train_loss_epoch, test_loss_epoch, train_output, test_output, finaltest_output = CMAPSS_CNN.train(model.model,model.guide,train_loader, test_loader,unshuffle_train_loader,finaltest_loader,num_iter,fusion, uncertainty)
# train_loss_epoch, test_loss_epoch, train_output, test_output, finaltest_output = model.fit(train_loader, test_loader,unshuffle_train_loader,finaltest_loader,num_iter)


# model, optimizer, loss_func = CMAPSS_CNN.Network()
#
# train_loss_epoch, test_loss_epoch, train_output, test_output, finaltest_output = CMAPSS_TrainLoop.Train(train_loader, test_loader,
#                                                                                        unshuffle_train_loader,
#                                                                                        finaltest_loader,
#                                                                                        model, optimizer,
#                                                                                        loss_func,
#                                                                                        num_epochs)
CMAPSS_PlotFunctions.train_actual_predicted(read_path, sub_dataset, window_size, max_life, fusion, uncertainty, train_output, alpha_grid, alpha_low, alpha_high, _COLORS)
CMAPSS_PlotFunctions.test_actual_predicted(read_path, sub_dataset, window_size, max_life, fusion, uncertainty, test_output, alpha_grid, alpha_low, alpha_high, _COLORS)
CMAPSS_PlotFunctions.loss_plot(sub_dataset, train_loss_epoch, test_loss_epoch, alpha_grid, alpha_low, alpha_high, _COLORS)
socre = CMAPSS_PlotFunctions.fianltest_actual_vs_predicted(read_path, sub_dataset, window_size, max_life, fusion, uncertainty, finaltest_output, alpha_grid, alpha_low, alpha_high, _COLORS, finalte_dataset)
#

