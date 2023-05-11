# -*- coding: utf-8 -*-
# @Time    : 2023/4/6 14:15
# @Author  : LeslieChiu
# @Site    : 
# @File    : Neural Networks.py
# @Software: PyCharm

import torch
from torch import nn
import warnings
warnings.filterwarnings('ignore')
torch.manual_seed(1234)
'''Neural Networks Category'''
'''Simple MLP'''
class MLP(nn.Module):
    def __init__(self,input_size):
        super(MLP,self).__init__()
        self.hidden_size = 64
        hidden_size = self.hidden_size
        self.input_fc = nn.Linear(input_size, hidden_size)
        self.layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for i in range(5)])
        self.output_fc = nn.Linear(hidden_size, 1)
        self.activation = nn.ReLU()

    def forward(self,x):
        x = self.input_fc(x)
        for idx in range(len(self.layers)):
            x = self.layers[idx](x)
            if idx < 5:
                x = self.activation(x)
        out = self.output_fc(x)
        return out


import torch
from torch import nn
'''Panel NN'''
class PanelNN:
    def __init__(self):
        self.model = None
        self.factor_names = None
        self.data = None
    def fit(self,data,factor_names):
        self.data = data
        self.factor_names = factor_names
    def train(self,sample_use,EPOCH,model_type):
        data = self.data.copy()
        factor_names = self.factor_names
        model,_ = TrainPanelNN(model_type,data,EPOCH,factor_names,sample_use)
        self.model = model

    def predict(self,data):
        net = self.model
        data = data.copy()
        x = data[self.factor_names].values
        x = torch.from_numpy(x).to(torch.float32)
        y_pred = net(x)
        return y_pred.detach().numpy()


'''import nn category from NeuralNetworks'''
'''DataLoader'''
def DataLoader(df,factor_names,sample_use):
    df_list = [group.reset_index(drop=True) for _,group in df.groupby('trade_date')]
    data_list = df_list[:sample_use]
    for data in data_list:
        x,y = data[factor_names].values,data['return_t+1'].values
        x,y = torch.from_numpy(x).to(torch.float32),torch.from_numpy(y).to(torch.float32)
        yield x,y

'''train function'''
def TrainPanelNN(model_type,data,EPOCH,factor_names,sample_use):
    dictionary = {
        'MLP': MLP,
    }
    net = dictionary[model_type](input_size=len(factor_names))
    loss = nn.MSELoss(reduction='none')
    opt = torch.optim.Adam(net.parameters())
    print(data['trade_date'].unique()[-1])
    for epoch in range(EPOCH):
        dataloader = DataLoader(data,factor_names,sample_use)
        train_loss = list()
        opt.zero_grad()
        l = 0.0
        for x,y in dataloader:
            if model_type in ['MLP']:
                y = y.unsqueeze(1)
            l += loss(net(x),y).mean()
        l.backward()
        opt.step()
        train_loss.append(l.item()/sample_use)
        if epoch%100 == 0:
            print(f'epoch:{epoch+1},loss:{train_loss[-1]}')
    return net,train_loss

