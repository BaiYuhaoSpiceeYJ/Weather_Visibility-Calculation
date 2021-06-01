import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import seaborn as sns
from torch.autograd import Variable
import pandas as pd
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 50000)

df = pd.read_csv('2020.csv', encoding='utf-8')
df = df.dropna(how='any')
df = df[~df['PM2.5'].str.contains('—')]
df = df[~df['PM10'].str.contains('—')]
df = df[~df['AQI空气质量指数'].str.contains('—')]
df = df[['温度', '气压', '湿度', '平均风速', 'PM2.5', 'PM10', 'AQI空气质量指数', '能见度']]
df = pd.DataFrame(df, dtype='float32')
df['温度'] = df['温度'] / 30
df['气压'] = df['气压'] / 1000
df['湿度'] = df['湿度'] / 100
df['平均风速'] = df['平均风速'] / 3
df['PM2.5'] = df['PM2.5'] / 100
df['PM10'] = df['PM10'] / 100
df['AQI空气质量指数'] = df['AQI空气质量指数'] / 100
df['能见度'] = df['能见度'] / 10

df = np.array(df, dtype=np.float32)
np.random.shuffle(df)
train_split = df[:int(df.shape[0]*0.8)]
test_split = df[int(df.shape[0]*0.8):]

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        hidden = 64
        self.fc1 = nn.Linear(7, hidden)
        self.activate1 = torch.nn.LeakyReLU()
        self.fc2 = nn.Linear(hidden, hidden)
        self.activate2 = torch.nn.LeakyReLU()
        self.fc3 = nn.Linear(hidden, hidden)
        self.activate3 = torch.nn.LeakyReLU()
        self.fc4 = nn.Linear(hidden, 1)
        self.activate4 = torch.nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.activate1(x)
        x = self.fc2(x)
        x = self.activate2(x)
        x = self.fc3(x)
        x = self.activate3(x)
        x = self.fc4(x)
        x = self.activate4(x)
        return x

class Dataset(torch.utils.data.dataset.Dataset):
    def __init__(self, df):
        super(Dataset, self).__init__()
        self.df = df
        self.len = df.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        seed = np.random.randint(0, self.len)
        train = self.df[seed, :-1]
        label = self.df[seed, -1]
        return torch.FloatTensor(train), torch.FloatTensor([label])

dset = Dataset(train_split)
dloader = DataLoader(dset, batch_size=128, shuffle=True, num_workers=0)
iter_dloader = dloader

net = Net()
net = net.cuda()
print("Total number of parameters is {}  ".format(sum(x.numel() for x in net.parameters())))
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
criterion = torch.nn.MSELoss(reduction='mean')

if False:
    for num_iter in range(10000000):
        net.train()
        df, label = iter(iter_dloader).next()
        label = label
        df = Variable(df.cuda())
        label = Variable(label.cuda())
        optimizer.zero_grad()
        predict_label = net(df)
        loss = criterion(predict_label, label)
        loss.backward()
        optimizer.step()
        print('Iter:{}    Loss:{}'.format(num_iter, loss.item()))
        if (num_iter + 1) / 100000 == 0:
            torch.save(net, str(num_iter+1) + '.ckpt')

net = torch.load('100000.ckpt')
net.eval()
out = []
label = []
sub1 = []
sub2 = []
train_split = train_split[np.argsort(train_split[:, -1])]
for i in range(train_split.shape[0]):
    label.append(train_split[i, -1])
    df = torch.FloatTensor(train_split[i, :-1]).cuda()
    out_ = float(net(df)[0].cpu())
    out.append(out_)
    sub1.append(out_ - train_split[i, -1])
    sub2.append((out_ - train_split[i, -1]) / train_split[i, -1])

plt.subplot(311)
plt.plot(range(len(label)), sorted(label), c='k', label='real')
plt.subplot(312)
plt.plot(range(len(out)), out, c='r', label='predict')
plt.subplot(313)
sns.distplot(sub1)
plt.show()

print(np.array(np.abs(sub1)).mean()*10, 'km')
print(np.array(np.abs(sub2)).mean()*100, '%')

out = []
label = []
sub1 = []
sub2 = []

test_split = test_split[np.argsort(test_split[:, -1])]
for i in range(test_split.shape[0]):
    label.append(test_split[i, -1])
    df = torch.FloatTensor(test_split[i, :-1]).cuda()
    out_ = float(net(df)[0].cpu())
    out.append(out_)
    sub1.append(out_ - test_split[i, -1])
    sub2.append((out_ - test_split[i, -1]) / test_split[i, -1])

plt.subplot(311)
plt.plot(range(len(label)), sorted(label), c='k', label='real')
plt.subplot(312)
plt.plot(range(len(out)), out, c='r', label='predict')
plt.subplot(313)
sns.distplot(sub1)
plt.show()

print(np.array(np.abs(sub1)).mean()*10, 'km')
print(np.array(np.abs(sub2)).mean()*100, '%')
