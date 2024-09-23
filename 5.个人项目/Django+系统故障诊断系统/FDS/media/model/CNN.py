import numpy as np
import pandas as pd
import torch
import torch.utils.data as data_utils
import torch.nn as nn
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from sklearn.impute import KNNImputer
from torch.autograd import Variable
import torch.nn.functional as F
def mean(data,no_elements):
    Y=np.zeros((data.shape[0],data.shape[1]))
    for i in range(data.shape[1]-no_elements+1):
        Y[:,i]=np.mean(data[:,i:i+no_elements],axis=1)
    return Y.astype(np.float64)
def sig_image(data,size):
    X=np.zeros((data.shape[0],size,size))
    for i in range(data.shape[0]):
        X[i]=(data[i,:].reshape(size,size))
    return X.astype(np.float64)
data_1 = pd.read_csv('./datashet/output.csv')
data_2 = pd.read_csv('./datashet/validate_1000.csv')
data_ = pd.concat([data_1, data_2]).reset_index(drop=True)
data_list = data_.values.tolist()
data_imputer = KNNImputer(n_neighbors=1)
df_result = data_imputer.fit_transform(data_list)
df = pd.DataFrame(df_result)
data=df.iloc[:,1:-1]
Labels=df.iloc[:,-1]
zreo_M = pd.DataFrame(np.zeros((11000, 14)))
data_concatenated = pd.concat([data, zreo_M], axis=1)

x=data_concatenated.to_numpy()
labels = Labels.to_numpy()
x_n=sig_image(x,11)
X = np.expand_dims(x_n, axis=1)
trainx, testx, trainlabel, testlabel = train_test_split(X, labels, test_size=0.2, random_state=2023)

sig_train, sig_test = trainx, testx
lab_train, lab_test = trainlabel, testlabel

sig_train = torch.from_numpy(sig_train)
sig_test = torch.from_numpy(sig_test)
lab_train= torch.from_numpy(lab_train)
lab_test = torch.from_numpy(lab_test)

batch_size = 128
train_tensor = data_utils.TensorDataset(sig_train, lab_train)
train_loader = data_utils.DataLoader(dataset = train_tensor, batch_size = batch_size, shuffle = True)

batch_size = 3000
test_tensor = data_utils.TensorDataset(sig_test, lab_test)
test_loader = data_utils.DataLoader(dataset = test_tensor, batch_size = batch_size, shuffle = False)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.mp1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(8, 32, kernel_size=2, stride=1)
        self.mp2 = nn.MaxPool2d(kernel_size=2, stride=1)
        self.fc1 = nn.Linear(288, 128)
        self.dp1 = nn.Dropout(p=0.1)
        self.fc2 = nn.Linear(128, 6)


    def forward(self, x):
        in_size = x.size(0)
        x = F.relu(self.mp1(self.conv1(x)))
        x = F.relu(self.mp2(self.conv2(x)))
        x = x.view(in_size, -1)
        x = F.relu(self.fc1(x))
        x = self.dp1(x)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)

cnn = CNN().double()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001)
num_epochs = 30

total_step = len(train_loader)
loss_list = []
acc_list = []
for epoch in range(num_epochs):
    for i, (signals, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        # Run the forward pass
        signals = signals
        labels = labels
        outputs = cnn(signals.double())
        loss = criterion(outputs, labels.long())

        loss_list.append(loss.item())

        # Backprop and perform Adam optimisation

        loss.backward()
        optimizer.step()
        # Track the accuracy
        total = labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels.long()).sum().item()
        acc_list.append(correct / total)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Train Accuracy: {:.2f}%'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                          (correct / total) * 100))

total_step = len(test_loader)
print(total_step)
loss_list_test = []
acc_list_test = []
with torch.no_grad():
    for i, (signals, labels) in enumerate(test_loader):
        # Run the forward pass
        signals=signals
        labels=labels
        outputs = cnn(signals.double())
        loss = criterion(outputs, labels.long())
        loss_list_test.append(loss.item())
        if epoch%10 ==0:
            print(loss)
        total = labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels.long()).sum().item()
        acc_list_test.append(correct / total)
        if (epoch) % 1 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                          (correct / total) * 100))

Predicted=predicted.numpy()
score = classification_report(testlabel, Predicted)
print(score)
lines = score.split('\n')
precision=[]
recall=[]
for i in range(2,8):
    precision.append(float(lines[i].split()[1]))
    recall.append(float(lines[i].split()[2]))
precision_average = sum(precision) / len(precision)
recall_average = sum(recall) / len(recall)
F1_score=200*(precision_average*recall_average)/(precision_average+recall_average)

print("F1_score:",F1_score)