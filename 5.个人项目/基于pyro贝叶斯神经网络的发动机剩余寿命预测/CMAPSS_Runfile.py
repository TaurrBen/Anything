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
batch_size=256#适用于你的内存大小
num_iter = 2
alpha_grid=0.4
alpha_low=0.6
alpha_high=0.9
_COLORS=["green", "teal","pink"]
read_path = r"..\C-MAPSS-Data/"

train_loader, test_loader, unshuffle_train_loader, finaltest_loader=CMAPSS_Dataloader.LoadCMAPSS(read_path, sub_dataset,
                                                                                   window_size, max_life, scaler, 
                                                                                   scaler_range, shuffle,
                                                                                   batch_size)

# train_loss_epoch, test_loss_epoch, train_output, test_output, finaltest_output = CMAPSS_CNN.train(train_loader, test_loader, unshuffle_train_loader, finaltest_loader, num_iter)
# net = CMAPSS_CNN.CNN()
# net = CMAPSS_CNN.MLP()
net = CMAPSS_CNN.LSTM(batch_size=batch_size,seq_length=window_size,feature_length=14)
model = CMAPSS_CNN.BayesianCNN(net)
train_loss_epoch, test_loss_epoch, train_output, test_output, finaltest_output = CMAPSS_CNN.train(model.model,model.guide,train_loader, test_loader,unshuffle_train_loader,finaltest_loader,num_iter)
# train_loss_epoch, test_loss_epoch, train_output, test_output, finaltest_output = model.fit(train_loader, test_loader,unshuffle_train_loader,finaltest_loader,num_iter)


# model, optimizer, loss_func = CMAPSS_CNN.Network()
#
# train_loss_epoch, test_loss_epoch, train_output, test_output, finaltest_output = CMAPSS_TrainLoop.Train(train_loader, test_loader,
#                                                                                        unshuffle_train_loader,
#                                                                                        finaltest_loader,
#                                                                                        model, optimizer,
#                                                                                        loss_func,
#                                                                                        num_epochs)
CMAPSS_PlotFunctions.train_actual_predicted(read_path, sub_dataset, window_size, max_life, train_output, alpha_grid, alpha_low, alpha_high, _COLORS, batch_size)
CMAPSS_PlotFunctions.test_actual_predicted(read_path, sub_dataset, window_size, max_life, test_output, alpha_grid, alpha_low, alpha_high, _COLORS, batch_size)
CMAPSS_PlotFunctions.loss_plot(sub_dataset, train_loss_epoch, test_loss_epoch, alpha_grid, alpha_low, alpha_high, _COLORS)
socre = CMAPSS_PlotFunctions.fianltest_actual_vs_predicted(read_path, sub_dataset, window_size, max_life, finaltest_output, alpha_grid, alpha_low, alpha_high, _COLORS, batch_size)
#

