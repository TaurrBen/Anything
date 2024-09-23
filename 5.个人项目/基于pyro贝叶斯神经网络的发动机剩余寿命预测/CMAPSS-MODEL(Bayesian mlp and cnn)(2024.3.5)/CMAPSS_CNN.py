import math

import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule
import torch
import torch.nn.functional as F
from tqdm import tqdm
from pyro.poutine.trace_messenger import TraceMessenger
from torch.distributions import constraints


class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.zeropad = torch.nn.ZeroPad2d((0, 0, 0, 9))
        self.conv1 = torch.nn.Conv2d(1, 10, (10, 1), 1, 0, 1)
        # self.conv2 = torch.nn.Conv2d(10, 10, (10, 1), 1, 0, 1)
        # self.conv3 = torch.nn.Conv2d(10, 10, (10, 1), 1, 0, 1)
        self.conv4 = torch.nn.Conv2d(10, 10, (10, 1), 1, 0, 1)
        self.conv5 = torch.nn.Conv2d(10, 1, (3, 1), 1, (1, 0), 1)
        self.fc1 = torch.nn.Linear(420, 100)
        self.fc2 = torch.nn.Linear(100, 1)
        self.dropout = torch.nn.Dropout(p=0.5, inplace=False)
        self.activfunc = torch.nn.Tanh()

    def forward(self,input_):

        out = self.zeropad(input_)
        out = self.conv1(out)
        out = self.activfunc(out)

        # out = self.zeropad(out)
        # out = self.conv2(out)
        # out = self.activfunc(out)
        #
        # out = self.zeropad(out)
        # out = self.conv3(out)
        # out = self.activfunc(out)

        out = self.zeropad(out)
        out = self.conv4(out)
        out = self.activfunc(out)

        out = self.conv5(out)
        out = self.activfunc(out)

        out = out.view(out.size(0),-1)

        out = self.fc1(out)
        out = self.activfunc(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(30*14, 100)
        self.fc2 = torch.nn.Linear(100, 1)
        self.fc3 = torch.nn.Linear(100, 1)
        self.relu = torch.nn.ReLU()
        self.softplus = torch.nn.Softplus()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self,input_):

        out = input_.view(input_.size(0),input_.size(2)*input_.size(3))
        out = self.fc1(out)
        out = self.relu(out)
        mean = self.fc2(out)
        # mean = self.relu(mean)

        sigma = self.fc3(out)
        # sigma = 0.5 * (self.sigmoid(sigma) + 1) + 1e-05
        sigma = self.softplus(0.01 * sigma) + 1e-05
        return mean,sigma

class LSTM(torch.nn.Module):
    def __init__(self, input_size=14, hidden_layer_size=None, output_size=1, batch_size=256,bidirectional=False):
        super().__init__()
        self.num_directions = 1
        if hidden_layer_size is None:
            # hidden_layer_size = [10, 32, 10]
            hidden_layer_size = [64, 32, 16]
        if bidirectional:
            self.num_directions = 2
        self.relu = torch.nn.ReLU()
        self.softplus = torch.nn.Softplus()
        self.sigmoid = torch.nn.Sigmoid()
        self.lstm1 = torch.nn.LSTM(input_size=input_size,hidden_size=hidden_layer_size[0],num_layers=1,batch_first=True,dropout=0.1,bidirectional=bidirectional)
        self.lstm2 = torch.nn.LSTM(input_size=hidden_layer_size[0]*self.num_directions, hidden_size=hidden_layer_size[1], num_layers=1, batch_first=True,dropout=0.1,bidirectional=bidirectional)
        self.lstm3 = torch.nn.LSTM(input_size=hidden_layer_size[1]*self.num_directions, hidden_size=hidden_layer_size[2], num_layers=1, batch_first=True,dropout=0.1,bidirectional=bidirectional)

        self.fc1 = torch.nn.Linear(hidden_layer_size[2]*self.num_directions, output_size)
        self.fc2 = torch.nn.Linear(hidden_layer_size[2]*self.num_directions, output_size)

        # self.hidden_cell = (torch.zeros(1, batch_size, self.hidden_layer_size),
        #                     torch.zeros(1, batch_size, self.hidden_layer_size))  # (num_layers * num_directions, batch_size, hidden_size)

    def forward(self, input_):
        input = input_.view(input_.size(0),input_.size(2),input_.size(3))
        lstm_out, _ = self.lstm1(input)
        lstm_out, _ = self.lstm2(lstm_out)
        lstm_out, _ = self.lstm3(lstm_out)
        mean = self.fc1(lstm_out[:,-1,:])
        # mean = self.relu(mean)

        sigma = self.fc2(lstm_out[:,-1,:])
        # sigma = 0.5 * (self.sigmoid(sigma) + 1) + 1e-05
        sigma = self.softplus(0.01 * sigma) + 1e-05
        return mean,sigma

class BayesianCNN(PyroModule):
    def __init__(self,net=CNN,conf=dict()):
        super().__init__()
        self.net = net()
        self.num_iters = conf.get('num_iters',1)
        self.batch_size = conf.get('batch_size', 500)
        self.print_every = conf.get('print_every', 1)

        self.weight_std = conf.get('weight_std', 0.1)
        self.noise_level = conf.get('noise_level', 1)

        self.lr = conf.get('lr', 0.1)
        self.betas = conf.get('betas', (0.9, 0.999))
        self.eps = conf.get('betas', 1e-04)
        self.weight_decay = conf.get('weight_decay', 0)
        self.opt = {"lr": self.lr,"betas":self.betas,"eps":self.eps,"weight_decay":self.weight_decay}
        self.loss_func = conf.get('loss_func', RMSELoss())

    def model(self,X,y=None):
        priors = {}
        def make_normal_prior(w):
            return dist.Normal(torch.zeros_like(w), torch.ones_like(w))
        for name, p in self.net.named_parameters():
            # priors[name] = make_normal_prior(p)
            if name.startswith("lstm1.weight")or name.startswith("lstm2.weight")or name.startswith("lstm3.weight"):
                priors[name] = dist.Normal(torch.zeros_like(p), torch.ones_like(p)).to_event(2)
            elif name.startswith("lstm1.bias")or name.startswith("lstm2.bias")or name.startswith("lstm3.bias"):
                priors[name] = dist.Normal(torch.zeros_like(p), 0.001 * torch.ones_like(p)).to_event(1)
            elif name.endswith("weight"):
                if name.startswith("conv"):
                    priors[name] = dist.Normal(torch.zeros_like(p), torch.ones_like(p)).to_event(4)
                elif name.startswith("fc") or name.startswith("linear"):
                    priors[name] = dist.Normal(torch.zeros_like(p), torch.ones_like(p)).to_event(2)
            elif name.endswith("bias"):
                if name.startswith("conv"):
                    priors[name] = dist.Normal(torch.zeros_like(p), 0.01 * torch.ones_like(p)).to_event(1)
                elif name.startswith("fc") or name.startswith("linear"):
                    priors[name] = dist.Normal(torch.zeros_like(p), 0.001 * torch.ones_like(p)).to_event(1)
            pyro.sample(name, priors[name])
        lifted_module = pyro.random_module("module", self.net, priors)
        lifted_reg_model = lifted_module()
        # mu = lifted_reg_model(X)
        # sigma = (pyro.sample("obs_sigma", dist.Normal(y.std(), 0.1)))+1e-04
        # sigma = 1
        mu,sigma = lifted_reg_model(X)
        with pyro.plate("data", X.shape[0]):
            ###########正态？？噪声？？
            # obs_sigma = pyro.sample("obs_sigma", dist.Normal(sigma, 0.01*torch.ones_like(mu)).to_event(1))
            # pyro.sample("obs", dist.Normal(mu, obs_sigma * torch.ones_like(mu)).to_event(1), obs=y)

            pyro.sample("obs", dist.Normal(mu, sigma * torch.ones_like(mu)).to_event(1), obs=y)
            # pyro.sample("obs", dist.Normal(mu + sigma,0.01 * torch.ones_like(mu)).to_event(1), obs=y)
        return mu

    def guide(self,X,y=None):
        priors = {}
        def make_variational_params(name, w, act_fn=F.softplus):
            with torch.no_grad():
                loc = pyro.param(f'{name}_loc', torch.randn_like(w))
                scale = pyro.param(f'{name}_scale', torch.randn_like(w))
            return pyro.distributions.Normal(loc, act_fn(scale))
        for name, p in self.net.named_parameters():
            priors[name] = make_variational_params(name, p)
            if name.startswith("lstm1.weight")or name.startswith("lstm2.weight")or name.startswith("lstm3.weight"):
                priors[name] = priors[name].to_event(2)
            elif name.startswith("lstm1.bias")or name.startswith("lstm2.bias")or name.startswith("lstm3.bias"):
                priors[name] = priors[name].to_event(1)
            elif name.endswith("weight"):
                if name.startswith("conv"):
                    priors[name] = priors[name].to_event(4)
                elif name.startswith("fc")or name.startswith("linear"):
                    priors[name] = priors[name].to_event(2)
            elif name.endswith("bias"):
                priors[name] = priors[name].to_event(1)
            pyro.sample(name, priors[name])
        lifted_module = pyro.random_module("module", self.net, priors)
        return lifted_module()

    def fit(self,train_loader, test_loader,unshuffle_train_loader,finaltest_loader,num_iter):
        self.num_iters = num_iter
        train_loss_epoch = []
        test_loss_epoch = []
        train_output = []
        test_output = []
        finaltest_output = []
        optim = pyro.optim.Adam(self.opt)
        loss = pyro.infer.Trace_ELBO()
        svi = pyro.infer.SVI(self.model, self.guide, optim, loss=loss)
        pyro.clear_param_store()
        print("\tIter | \tTrain Loss | \tTest Loss")
        for iter in range(self.num_iters):
            running_loss_tr = 0
            running_loss_te = 0
            batch_counter_tr = 0
            batch_counter_te = 0

            for i, (data_tr, label_tr) in enumerate(tqdm(train_loader)):
                batch_counter_tr += 1
                output_tr,_ = self.predict(data_tr.float())
                loss_tr = svi.step(data_tr,label_tr)
                running_loss_tr += loss_tr
                if iter == self.num_iters - 1:
                    train_output += output_tr.flatten().tolist()
            epoch_loss_tr = running_loss_tr / batch_counter_tr
            train_loss_epoch.append(epoch_loss_tr)

            if iter == self.num_iters - 1:
                if unshuffle_train_loader is not None:
                    train_output = []
                    for i, (data_uns_tr, lable_un_tr) in enumerate(tqdm(unshuffle_train_loader)):
                        output_uns_tr,_ = self.predict(data_uns_tr.float())
                        train_output += output_uns_tr.flatten().tolist()
            self.eval()
            for i, (data_te, label_te) in enumerate(tqdm(test_loader)):
                batch_counter_te += 1
                output_te,_ = self.predict(data_te.float())
                loss_te = self.loss_func(output_te, label_te)
                running_loss_te += loss_te.item()
                if iter == self.num_iters - 1:
                    test_output += output_te.flatten().tolist()

            epoch_loss_te = running_loss_te / batch_counter_te
            test_loss_epoch.append(epoch_loss_te)
            print("\n\t{} \t{} \t{}".format(iter + 1, epoch_loss_tr, epoch_loss_te))
        self.eval()
        for i, (data_fte) in enumerate(tqdm(finaltest_loader)):
            output_fte,_ = self.predict(data_fte.float())
            finaltest_output += output_fte.flatten().tolist()

        return train_loss_epoch, test_loss_epoch, train_output, test_output, finaltest_output






    # def forward(self, X, y=None):
    #     return self.net.forward(X)

    def predict(self,X,num_sample=20):
        sampled_models = [self.guide(None, None) for _ in range(num_sample)]
        y = [model(X).data for model in sampled_models]
        y_pred = torch.mean(torch.stack(y),0)
        y_std = torch.std(torch.stack(y),0)

        # predictive = pyro.infer.Predictive(model=self.model, guide=self.guide, num_samples=num_sample)
        # preds = predictive(X,None)
        # y_pred = preds['obs'].T.detach().numpy().mean(axis=3)
        # y_std = preds['obs'].T.detach().numpy().std(axis=3)
        return y_pred, y_std

def weights_init(layer):
    if isinstance(layer, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(layer.weight, mode = 'fan_out')
        if layer.bias is not None:
            layer.bias.data.fill_(0.01)
    
    elif isinstance(layer, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(layer.weight, mode = 'fan_out')
        if layer.bias is not None:
            layer.bias.data.fill_(0.001)
            
    return None

class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.mse = torch.nn.MSELoss()
            
    def forward(self, pred, actual):
        return torch.sqrt(self.mse(pred,actual))



def train(model,guide,train_loader, test_loader,unshuffle_train_loader,finaltest_loader,num_iter,fusion=False,uncertainty=False):
    loss_func = RMSELoss()
    train_loss_epoch = []
    test_loss_epoch = []
    train_output = []
    test_output = []
    finaltest_output = []
    train_output_std = []
    test_output_std = []
    finaltest_output_std = []

    train_output_epistemic_uncertainty = []
    train_output_aleatoric_uncertainty = []
    test_output_epistemic_uncertainty = []
    test_output_aleatoric_uncertainty = []
    finaltest_epistemic_uncertainty = []
    finaltest_aleatoric_uncertainty = []
    optim = pyro.optim.Adam({"lr": 0.1,"betas":(0.9, 0.999),"eps":1e-04,"weight_decay": 0})
    loss = pyro.infer.Trace_ELBO()
    svi = pyro.infer.SVI(model, guide, optim, loss=loss)
    pyro.clear_param_store()
    print("\tIter | \tTrain Loss | \tTest Loss")
    for iter in range(num_iter):
        running_loss_tr = 0
        running_loss_te = 0
        batch_counter_tr = 0
        batch_counter_te = 0

        for i, (data_tr, label_tr) in enumerate(tqdm(train_loader)):
            # observation
            # with TraceMessenger() as tracer:
            #     guide(None,None)
            # trace = tracer.trace
            # for name, node in trace.nodes.items():
            #     if name.startswith('fc2'):
            #         print(name, node['fn'], node['value'], node['is_observed'])

            batch_counter_tr += 1
            output_tr = predict(guide,data_tr.float(),uncertainty)
            loss_tr = svi.step(data_tr,label_tr)
            running_loss_tr += loss_tr
            if iter == num_iter - 1:
                train_output += output_tr[0].flatten().tolist()
                train_output_std += output_tr[1].flatten().tolist()
                out1 = (train_output, train_output_std)
                if uncertainty:
                    train_output_epistemic_uncertainty += output_tr[2].flatten().tolist()
                    train_output_aleatoric_uncertainty += output_tr[3].flatten().tolist()
                    out1 = (train_output, train_output_std,train_output_epistemic_uncertainty,train_output_aleatoric_uncertainty)
        epoch_loss_tr = running_loss_tr / batch_counter_tr
        train_loss_epoch.append(epoch_loss_tr)

        if iter == num_iter - 1:
            if unshuffle_train_loader is not None:
                train_uns_output = []
                train_uns_output_std = []
                train_uns_output_epistemic_uncertainty = []
                train_uns_output_aleatoric_uncertainty = []
                for i, (data_uns_tr, lable_un_tr) in enumerate(tqdm(unshuffle_train_loader)):
                    # observation
                    # with TraceMessenger() as tracer:
                    #     guide(None, None)
                    # trace = tracer.trace
                    # for name, node in trace.nodes.items():
                    #     print(name, node['fn'], node['value'], node['is_observed'])

                    output_uns_tr = predict(guide,data_uns_tr.float(),uncertainty)
                    train_uns_output += output_uns_tr[0].flatten().tolist()
                    train_uns_output_std += output_uns_tr[1].flatten().tolist()
                    out1 = (train_uns_output, train_uns_output_std)
                    if uncertainty:
                        train_uns_output_epistemic_uncertainty += output_uns_tr[2].flatten().tolist()
                        train_uns_output_aleatoric_uncertainty += output_uns_tr[3].flatten().tolist()
                        out1 = (train_uns_output, train_uns_output_std, train_uns_output_epistemic_uncertainty,
                                train_uns_output_aleatoric_uncertainty)
        # model.eval()
        for i, (data_te, label_te) in enumerate(tqdm(test_loader)):
            batch_counter_te += 1
            output_te = predict(guide,data_te.float(),uncertainty)
            loss_te = loss_func(output_te[0], label_te)
            running_loss_te += loss_te.item()
            if iter == num_iter- 1:
                test_output += output_te[0].flatten().tolist()
                test_output_std += output_te[1].flatten().tolist()
                out2 = (test_output, test_output_std)
                if uncertainty:
                    test_output_epistemic_uncertainty += output_te[2].flatten().tolist()
                    test_output_aleatoric_uncertainty += output_te[3].flatten().tolist()
                    out2 = (test_output, test_output_std,test_output_epistemic_uncertainty,test_output_aleatoric_uncertainty)

        epoch_loss_te = running_loss_te / batch_counter_te
        test_loss_epoch.append(epoch_loss_te)
        print("\n\t{}/{} \t{} \t{}".format(iter + 1, num_iter,epoch_loss_tr, epoch_loss_te))
    # model.eval()
    def summary(samples):
        site_stats = {}

        for k, v in samples.items():
            site_stats[k] = {
                "mean": torch.mean(v, 0),
                "std": torch.std(v, 0),
                "5%": v.kthvalue(int(len(v) * 0.05), dim=0)[0],
                "95%": v.kthvalue(int(len(v) * 0.95), dim=0)[0],
            }
        return site_stats

    # predictive = pyro.infer.Predictive(model=model, guide=guide, num_samples=100,return_sites=["obs",'obs_sigma','_RETURN'])
    for i, (data_fte) in enumerate(tqdm(finaltest_loader)):
        # samples = predictive(data_fte.float())
        # pred_summary = summary(samples)
        output_fte = predict(guide,data_fte.float(),uncertainty)
        # finaltest_output += pred_summary['obs']['mean'].flatten().tolist()
        # finaltest_output_std += pred_summary['obs']['std'].flatten().tolist()
        finaltest_output += output_fte[0].flatten().tolist()
        finaltest_output_std += output_fte[1].flatten().tolist()#output_fte_std
        out3 = (finaltest_output, finaltest_output_std)
        if uncertainty:
            finaltest_epistemic_uncertainty += output_fte[2].flatten().tolist()#output_fte_epistemic_uncertainty
            finaltest_aleatoric_uncertainty += output_fte[3].flatten().tolist()#output_fte_aleatoric_uncertainty
            out3 = (finaltest_output,finaltest_output_std,finaltest_epistemic_uncertainty,finaltest_aleatoric_uncertainty)

    return train_loss_epoch, test_loss_epoch, out1, out2, out3

def predict(guide,X,uncertainty=False,num_sample1=10,num_sample2=10):
    def NormalSampling(mu,sigma,num_sample):
        pred_data = []
        for i in range(num_sample):
            temp = pyro.sample("pred", dist.Normal(mu, sigma * torch.ones_like(mu))).data
            pred_data += [temp]
        return pred_data
    sampled_models = [guide(None,None) for _ in range(num_sample1)]
    # y = [model(X).data for model in sampled_models]
    # y1 = [model(X) for model in sampled_models]
    y2 = [model(X)[0] for model in sampled_models]  ##条件期望？
    y3 = [model(X)[1] for model in sampled_models]  ##条件方差？
    #
    y = y2
    y_mean1 = torch.mean(torch.stack(y),0)  ####条件期望？
    y_std1 = torch.std(torch.stack(y),0)  ####条件期望的标准差？
    # y_mean1 = torch.mean(torch.exp(torch.stack(y)) - 1, 0)
    # y_std1 = torch.std(torch.exp(torch.stack(y)) - 1, 0)
    y_mean,y_std = y_mean1,y_std1
    out = y_mean, y_std
    if uncertainty:
        epistemic_uncertainty1 = torch.var(torch.stack(y),0)   ####条件期望的方差？
        aleatoric_uncertainty1 = torch.mean(torch.stack(y3),0)  ##条件方差的均值？
        y4 = torch.stack([torch.stack(NormalSampling(y2[i], y3[i], num_sample2)) for i in range(num_sample1)])
        # y4 = torch.exp(torch.stack([torch.stack(NormalSampling(y2[i], y3[i], num_sample2)) for i in range(num_sample1)]))-1
        y_mean2 = torch.mean(torch.mean(y4, 1), 0)
        y_var2 = torch.mean(torch.mean((y4 - y_mean2) ** 2, 1), 0)
        y_mean3 = torch.mean(y4, 1)
        y_var3 = torch.mean(torch.mean((y4.view(y4.size(1), y4.size(0), y4.size(2), y4.size(3)) - y_mean3) ** 2, 0), 0)
        aleatoric_uncertainty2 = y_var3
        epistemic_uncertainty2 = y_var2 - aleatoric_uncertainty2
        # epistemic_uncertainty, aleatoric_uncertainty = epistemic_uncertainty1, aleatoric_uncertainty1
        epistemic_uncertainty, aleatoric_uncertainty = epistemic_uncertainty2, aleatoric_uncertainty2
        y_mean, y_std = y_mean2, torch.sqrt(y_var2)
        out = y_mean, y_std , epistemic_uncertainty, aleatoric_uncertainty

    # # predictive = pyro.infer.Predictive(model=model, guide=guide, num_samples=num_sample)
    # # preds = predictive(X,None)
    # # y_pred = preds['obs'].T.detach().numpy().mean(axis=3)
    # # y_std = preds['obs'].T.detach().numpy().std(axis=3)
    return out

def Network():
    model = BayesianCNN()
    model.apply(weights_init)
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001, betas=(0.9, 0.999), eps=1e-04, weight_decay=0)
    loss_func = RMSELoss()
    return model, optimizer, loss_func
    
    
    
    
    
    