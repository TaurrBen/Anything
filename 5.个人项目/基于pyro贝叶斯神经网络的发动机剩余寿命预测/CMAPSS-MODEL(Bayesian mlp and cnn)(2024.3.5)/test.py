# import torch
# import pyro
# pyro.set_rng_seed(101)
#
# def weather():
#     cloudy = pyro.sample('cloudy', pyro.distributions.Bernoulli(0.3))
#     cloudy = 'cloudy' if cloudy.item() == 1.0 else 'sunny'
#     mean_temp = {'cloudy': 55.0, 'sunny': 75.0}[cloudy]
#     scale_temp = {'cloudy': 10.0, 'sunny': 15.0}[cloudy]
#     temp = pyro.sample('temp', pyro.distributions.Normal(mean_temp, scale_temp))
#     return cloudy, temp.item()
#
# import pyro.distributions as dist
# from pyro.poutine.trace_messenger import TraceMessenger
# mu = 8.5
# def scale_obs(mu):
#     weight = pyro.sample("weight", dist.Normal(mu, 1.))
#     return pyro.sample("measurement", dist.Normal(weight, 0.75), obs=torch.tensor(9.5))
#
# with TraceMessenger() as tracer:
#     scale_obs(mu)
#
# trace = tracer.trace
# logp = 0.
# for name, node in trace.nodes.items():
#     print(name, node['fn'], node['value'], node['is_observed'])
#     if node["type"] == "sample":
#         logp = logp + node["fn"].log_prob(node["value"]).sum()
#
# def scale(mu):
#     weight = pyro.sample("weight", dist.Normal(mu, 1.0))
#     return pyro.sample("measurement", dist.Normal(weight, 0.75))
#
# # Hint for the next section
# def scale_obs(mu):
#     weight = pyro.sample("weight", dist.Normal(mu, 1.))
#     return pyro.sample("measurement", dist.Normal(weight, 0.75), obs=9.5)
#
# mu = 8.5
# conditioned_scale = pyro.condition(scale, data={"measurement": 9.5})
# # Input of `pyro.condition`: a model and a dictionary of observations
# conditioned_scale(mu)
# # Always uses the given values at observed sample statements!
#
# import pyro.distributions as dist
# from pyro.poutine.trace_messenger import TraceMessenger
#
# cond_data = {"temp": torch.tensor(52)}
#
# with TraceMessenger() as tracer:
#     conditioned_scale(mu)
#
# trace = tracer.trace
# logp = 0.
# for name, node in trace.nodes.items():
#     print(name, node['fn'], node['value'], node['is_observed'])
#     if node["type"] == "sample":
#         logp = logp + node["fn"].log_prob(node["value"]).sum()
#
# def deferred_conditioned_scale(measurement, mu):
#     return pyro.condition(scale, data={"measurement": measurement})(mu)
#
# def scale_obs(mu):
#     weight = pyro.sample("weight", dist.Normal(mu, 1.))
#     return pyro.sample("measurement", dist.Normal(weight, 0.75), obs=9.5)

# #####################################2
# import matplotlib.pyplot as plt
# import numpy as np
# import torch
#
# import pyro
# import pyro.infer
# import pyro.optim
# import pyro.distributions as dist
# from pyro.poutine.trace_messenger import TraceMessenger
# pyro.set_rng_seed(101)
# mu = torch.tensor(8.5)
#
# def scale(mu):
#     weight = pyro.sample("weight", dist.Normal(mu, 1.0))
#     return pyro.sample("measurement", dist.Normal(weight, 0.75))
# conditioned_scale = pyro.condition(scale, data={"measurement": torch.tensor(9.5)})
#
# def scale_parametrized_guide(mu):
#     a = pyro.param("a", torch.tensor(mu))
#     b = pyro.param("b", torch.tensor(1.))
#     return pyro.sample("weight", dist.Normal(a, torch.abs(b)))
#
# pyro.clear_param_store()
# svi = pyro.infer.SVI(model=conditioned_scale,
#                      guide=scale_parametrized_guide,
#                      optim=pyro.optim.SGD({"lr": 0.001, "momentum":0.1}),
#                      loss=pyro.infer.Trace_ELBO())
#
# losses, a,b  = [], [], []
# num_steps = 2500
# for t in range(num_steps):
#     with TraceMessenger() as tracer:
#         # conditioned_scale(mu)
#         scale_parametrized_guide(mu)
#
#     trace = tracer.trace
#     logp = 0.
#     for name, node in trace.nodes.items():
#         pass
#         # print(name, node['fn'], node['value'], node['is_observed'])
#         # if node["type"] == "sample":
#         #     logp = logp + node["fn"].log_prob(node["value"]).sum()
#
#     losses.append(svi.step(mu))
#     a.append(pyro.param("a").item())
#     b.append(pyro.param("b").item())
#
# plt.plot(losses)
# plt.title("ELBO")
# plt.xlabel("step")
# plt.ylabel("loss")
# print('a = ',pyro.param("a").item())
# print('b = ', pyro.param("b").item())
#
# plt.subplot(1,2,1)
# plt.plot([0,num_steps],[9.14,9.14], 'k:')
# plt.plot(a)
# plt.ylabel('a')
#
# plt.subplot(1,2,2)
# plt.ylabel('b')
# plt.plot([0,num_steps],[0.6,0.6], 'k:')
# plt.plot(b)
# plt.tight_layout()

# #####################################3
# import matplotlib.pyplot as plt
# import numpy as np
# import torch
#
# import pyro
# import pyro.infer
# import pyro.optim
# import pyro.distributions as dist
# pyro.set_rng_seed(101)
# mu = 8.5
#
# def scale(mu):
#     weight = pyro.sample("weight", dist.Normal(mu, 1.0))
#     return pyro.sample("measurement", dist.Normal(weight, 0.75),obs=torch.tensor(9.5))
# # conditioned_scale = pyro.condition(scale, data={"measurement": torch.tensor(9.5)})
#
# def scale_parametrized_guide(mu):
#     a = pyro.param("a", torch.tensor(mu))
#     b = pyro.param("b", torch.tensor(1.))
#     return pyro.sample("weight", dist.Normal(a, torch.abs(b)))
#
# loss_fn = pyro.infer.Trace_ELBO().differentiable_loss
#
# pyro.clear_param_store()
# with pyro.poutine.trace(param_only=True) as param_capture: # 提取参数信息
#     loss = loss_fn(scale, scale_parametrized_guide, mu)
#     loss.backward()
# params = [site["value"].unconstrained() for site in param_capture.trace.nodes.values()]
#
# print("Before updated:", pyro.param('a'), pyro.param('b'))
#
# losses, a,b  = [], [], []
# lr = 0.001
# num_steps = 1000
# # 梯度下降参数更新
# def step(params):
#     for x in params:
#         x.data = x.data - lr * x.grad
#         x.grad.zero_()
#
# for t in range(num_steps):
#     with pyro.poutine.trace(param_only=True) as param_capture:
#         loss = loss_fn(scale, scale_parametrized_guide, mu)
#         loss.backward()
#         losses.append(loss.data)
#
#     params = [site["value"].unconstrained() for site in param_capture.trace.nodes.values()]
#     a.append(pyro.param("a").item())
#     b.append(pyro.param("b").item())
#     step(params)
# print("After updated:", pyro.param('a'), pyro.param('b'))
#
#
# plt.plot(losses)
# plt.title("ELBO")
# plt.xlabel("step")
# plt.ylabel("loss");
# print('a = ',pyro.param("a").item())
# print('b = ', pyro.param("b").item())

# #####################################4
# import math, os, torch, pyro
# import torch.distributions.constraints as constraints
# import pyro.distributions as dist
# from pyro.optim import Adam
# from pyro.infer import SVI, Trace_ELBO
#
# # assert pyro.__version__.startswith('1.3.0')
# pyro.enable_validation(True)
# pyro.clear_param_store()
#
# data = []
# data.extend([torch.tensor(1.0) for _ in range(6)])
# data.extend([torch.tensor(0.0) for _ in range(4)])
#
# def model(data): # 参数的先验分布是 Beta(10, 10)
#     alpha0, beta0 = torch.tensor(10.0), torch.tensor(10.0)
#     theta = pyro.sample("latent_fairness", dist.Beta(alpha0, beta0))
#     for i in range(len(data)):
#         pyro.sample("obs_{}".format(i), dist.Bernoulli(theta), obs=data[i])
# def guide(data): # 参数后验分布的 guide 是 Beta(p, q), p, q 初始值为 15.0， 15.0
#     alpha_q = pyro.param("alpha_q", torch.tensor(15.0), constraint=constraints.positive)
#     beta_q = pyro.param("beta_q", torch.tensor(15.0), constraint=constraints.positive)
#     pyro.sample("latent_fairness", dist.Beta(alpha_q, beta_q))
#
# adam_params = {"lr": 0.0005, "betas": (0.90, 0.999)}
# optimizer = Adam(adam_params)
# svi = SVI(model, guide, optimizer, loss=Trace_ELBO()) # 目标函数和优化方法
#
# n_steps = 2000
# for step in range(n_steps):
#     svi.step(data)
#     if step % 50 == 0:
#         print('.', end='')
#
# alpha_q = pyro.param("alpha_q").item()
# beta_q = pyro.param("beta_q").item()
# inferred_mean = alpha_q / (alpha_q + beta_q)
# factor = beta_q / (alpha_q * (1.0 + alpha_q + beta_q))
# inferred_std = inferred_mean * math.sqrt(factor)
# print("\nbased on the data and our prior belief, the fairness " +
#       "of the coin is %.3f +- %.3f" % (inferred_mean, inferred_std))


# #####################################5
import os, torch, pyro
import numpy as np
import torchvision.datasets as dset
import torch.nn as nn
import torchvision.transforms as transforms
import pyro.distributions as dist
import pyro.contrib.examples.util  # patches torchvision
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
pyro.enable_validation(True)
pyro.distributions.enable_validation(False)
pyro.set_rng_seed(0)

class Decoder(nn.Module): # 用于构建模型分布的 decoder
    def __init__(self, z_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(z_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, 784)
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        hidden = self.softplus(self.fc1(z))
        loc_img = self.sigmoid(self.fc21(hidden))
        return loc_img

class Encoder(nn.Module): # 用于构建指导分布的 encoder
    def __init__(self, z_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(784, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, z_dim)
        self.fc22 = nn.Linear(hidden_dim, z_dim)
        self.softplus = nn.Softplus()

    def forward(self, x):
        x = x.reshape(-1, 784)
        hidden = self.softplus(self.fc1(x))
        z_loc = self.fc21(hidden)
        z_scale = torch.exp(self.fc22(hidden))
        return z_loc, z_scale

class VAE(nn.Module):
    def __init__(self, z_dim=50, hidden_dim=400, use_cuda=False):
        super().__init__()
        self.encoder = Encoder(z_dim, hidden_dim)
        self.decoder = Decoder(z_dim, hidden_dim)
        if use_cuda:
            self.cuda()
        self.use_cuda = use_cuda
        self.z_dim = z_dim

    def model(self, x): # 模型分布  p(x|z)p(z)
        pyro.module("decoder", self.decoder)
        with pyro.plate("data", x.shape[0]):
            z_loc = x.new_zeros(torch.Size((x.shape[0], self.z_dim)))
            z_scale = x.new_ones(torch.Size((x.shape[0], self.z_dim)))
            z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
            loc_img = self.decoder.forward(z)
            pyro.sample("obs", dist.Bernoulli(loc_img).to_event(1), obs=x.reshape(-1, 784))

    def guide(self, x): # 指导分布 q(z|x)
        pyro.module("encoder", self.encoder)
        with pyro.plate("data", x.shape[0]):
            z_loc, z_scale = self.encoder.forward(x)
            pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))

    def reconstruct_img(self, x):
        z_loc, z_scale = self.encoder(x)
        z = dist.Normal(z_loc, z_scale).sample()
        loc_img = self.decoder(z) # 注意在图像空间中我们没有抽样
        return loc_img

def setup_data_loaders(batch_size=128, use_cuda=False):
    root = './data'
    download = True
    trans = transforms.ToTensor()
    train_set = dset.MNIST(root=root, train=True, transform=trans,
                           download=download)
    test_set = dset.MNIST(root=root, train=False, transform=trans)
    kwargs = {'num_workers': 1, 'pin_memory': use_cuda}
    train_loader = torch.utils.data.DataLoader(dataset=train_set,
        batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(dataset=test_set,
        batch_size=batch_size, shuffle=False, **kwargs)
    return train_loader, test_loader

def train(svi, train_loader, use_cuda=False):
    epoch_loss = 0.
    for x, _ in train_loader:
        if use_cuda:
            x = x.cuda()
        epoch_loss += svi.step(x)
    normalizer_train = len(train_loader.dataset)
    total_epoch_loss_train = epoch_loss / normalizer_train
    return total_epoch_loss_train

def evaluate(svi, test_loader, use_cuda=False):
    test_loss = 0.
    for x, _ in test_loader:
        if use_cuda:
            x = x.cuda()
        test_loss += svi.evaluate_loss(x)
    normalizer_test = len(test_loader.dataset)
    total_epoch_loss_test = test_loss / normalizer_test
    return total_epoch_loss_test

# 模型训练
LEARNING_RATE = 1.0e-3
USE_CUDA = False
NUM_EPOCHS = 5
TEST_FREQUENCY = 5
train_loader, test_loader = setup_data_loaders(batch_size=256, use_cuda=USE_CUDA)
pyro.clear_param_store()
vae = VAE(use_cuda=USE_CUDA)
adam_args = {"lr": LEARNING_RATE}
optimizer = Adam(adam_args)
svi = SVI(vae.model, vae.guide, optimizer, loss=Trace_ELBO())

train_elbo = []
test_elbo = []
for epoch in range(NUM_EPOCHS):
    total_epoch_loss_train = train(svi, train_loader, use_cuda=USE_CUDA)
    train_elbo.append(-total_epoch_loss_train)
    print("[epoch %03d]  average training loss: %.4f" % (epoch, total_epoch_loss_train))
    if epoch % TEST_FREQUENCY == 0:
        # report test diagnostics
        total_epoch_loss_test = evaluate(svi, test_loader, use_cuda=USE_CUDA)
        test_elbo.append(-total_epoch_loss_test)
        print("[epoch %03d] average test loss: %.4f" % (epoch, total_epoch_loss_test))