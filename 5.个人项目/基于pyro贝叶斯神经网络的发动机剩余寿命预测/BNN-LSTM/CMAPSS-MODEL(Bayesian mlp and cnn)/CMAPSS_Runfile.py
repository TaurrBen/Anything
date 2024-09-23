import pyro

import CMAPSS_Dataloader
import CMAPSS_CNN
import CMAPSS_TrainLoop
import CMAPSS_PlotFunctions
import CMAPSS_CONFIG as config

train_loader, test_loader, unshuffle_train_loader, finaltest_loader=CMAPSS_Dataloader.LoadCMAPSS(config.read_path,
                                                                                                 config.sub_dataset,
                                                                                                 config.window_size,
                                                                                                 config.max_life,
                                                                                                 config.scaler,
                                                                                                 config.scaler_range,
                                                                                                 config.shuffle,
                                                                                                 config.batch_size)

# train_loss_epoch, test_loss_epoch, train_output, test_output, finaltest_output = CMAPSS_CNN.train(train_loader, test_loader, unshuffle_train_loader, finaltest_loader, num_iter)
net = CMAPSS_CNN.LSTM
model = CMAPSS_CNN.BayesianCNN(net)
train_loss_epoch, test_loss_epoch, train_output, test_output, finaltest_output,(model,guide) = CMAPSS_CNN.train(model.model, model.guide, train_loader, test_loader, unshuffle_train_loader, finaltest_loader,config.num_iter)


# model, optimizer, loss_func = CMAPSS_CNN.Network()
#
# train_loss_epoch, test_loss_epoch, train_output, test_output, finaltest_output = CMAPSS_TrainLoop.Train(train_loader, test_loader,
#                                                                                        unshuffle_train_loader,
#                                                                                        finaltest_loader,
#                                                                                        model, optimizer,
#                                                                                        loss_func,
#                                                                                        num_epochs)

CMAPSS_PlotFunctions.train_actual_predicted(config.read_path, config.sub_dataset, config.window_size, config.max_life, train_output, config.alpha_grid, config.alpha_low, config.alpha_high, config._COLORS)
CMAPSS_PlotFunctions.test_actual_predicted(config.read_path, config.sub_dataset, config.window_size, config.max_life, test_output, config.alpha_grid, config.alpha_low, config.alpha_high, config._COLORS)
CMAPSS_PlotFunctions.loss_plot(config.sub_dataset, train_loss_epoch, test_loss_epoch, config.alpha_grid, config.alpha_low, config.alpha_high, config._COLORS)
CMAPSS_PlotFunctions.param_plot(model,guide,finaltest_loader)
socre = CMAPSS_PlotFunctions.fianltest_actual_vs_predicted(config.read_path, config.sub_dataset, config.window_size, config.max_life, finaltest_output, config.alpha_grid, config.alpha_low, config.alpha_high, config._COLORS)
#

