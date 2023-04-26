# -*- coding: utf-8 -*-
# @Time    : 2023/4/26 19:04
# @Author  : LeslieChiu
# @Site    : 
# @File    : PortfolioOptimize.py
# @Software: PyCharm
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from pylab import plt, mpl
plt.style.use( "seaborn" )
mpl.rcParams[ 'font.family' ] = 'serif'
import torch

'''define some function'''
def p_mean(r, w):
    p_m = np.sum(r * w) * 252
    return p_m


def p_std(cov, w):
    p_s = np.sqrt(np.dot(w.T, np.dot(cov, w))) * np.sqrt(252)
    return p_s


def p_var(w, r, cov):
    return p_mean(r, w), np.power(p_std(cov, w), 2)

'''PortfolioOptimize tools class'''
class PortfolioOptimize:
    def __init__(self, *args):
        self.data = None
        self.stock_list = None
        self.LogReturn = None
        self.FixedVariables = {
            "mean_daily_return": None,
            'cov_matrix': None
        }

    def set(self, data, stock_list):
        col = [i for i in data.columns]
        '''the Date and stock_list must in the columns'''
        if 'Date' not in data.columns:
            assert IndexError('Date is not in the columns of dataframe')
        for ts in stock_list:
            if ts not in data.columns:
                assert IndexError(f'the ts:{ts} is not in the columns of dataframe')
        self.data = data.copy()
        self.stock_list = stock_list
        stock = stock_list
        '''get log return dataframe'''
        data.index = pd.to_datetime(data['Date'])
        data = data[stock]
        data = np.log(data)
        data = data - data.shift(1)
        data = data.iloc[1:, :]
        self.LogReturn = data.copy()
        '''calculate some variables'''
        self.FixedVariables['mean_daily_return'] = data.mean()
        self.FixedVariables['cov_matrix'] = data.cov()

    '''get the max sharpe,R and min var with random stimulation'''

    def analysis(self, density=5000, download=False):
        data = self.LogReturn.copy()
        stock = self.stock_list.copy()
        results_size = ((3 + len(stock), density))
        results = np.zeros(results_size)
        r, cov = self.FixedVariables["mean_daily_return"], self.FixedVariables["cov_matrix"]
        '''get simulation result'''
        for i in range(density):
            weights = np.random.random(len(stock))
            weights /= np.sum(weights)
            portfolio_return = p_mean(r, weights)
            portfolio_std_dev = p_std(cov, weights)
            results[0, i] = portfolio_return
            results[1, i] = portfolio_std_dev
            results[2, i] = results[0, i] / results[1, i]
            for j in range(len(stock)):
                results[3 + j, i] = weights[j]
        '''map into dataframe'''
        columns = ['R', 'Std', 'Sharpe'] + stock
        df = pd.DataFrame(data=results.T, columns=columns)
        print('-----------------------------Simulation Result-----------------------------')
        df0 = df.loc[df['Sharpe'] == df['Sharpe'].max()]
        df1 = df.loc[df['R'] == df['R'].max()]
        df2 = df.loc[df['Std'] == df['Std'].min()]
        simulation_table = pd.concat([df0, df1, df2], axis=0)
        simulation_table.index = ['Max Sharpe', 'Max Return', 'Min Volatility']
        print(simulation_table)
        if download:
            simulation_table.to_csv('simulation_table.csv')
        print('-----------------------------Simulation Plot-----------------------------')
        '''plot'''
        fig = plt.figure(figsize=(20, 9))
        plt.scatter(x=df['Std'], y=df['R'], c=df['Sharpe'], cmap='RdYlBu')
        plt.xlabel('Std')
        plt.ylabel('R')
        plt.show()

    '''we optimize portfolio with pytorch'''

    def optimize(self, target, EPOCH, LR):
        data = self.LogReturn.copy()
        stock = self.stock_list.copy()
        r, cov = self.FixedVariables["mean_daily_return"].values, self.FixedVariables["cov_matrix"].values
        r, cov = torch.from_numpy(r).to(torch.float32), torch.from_numpy(cov).to(torch.float32)
        r = r.resize(r.size(0), 1)
        '''defince '''

        def Negetive_R(r, w, cov):
            return -torch.sum(w.T @ r) * 252

        def Std(r, w, cov):
            return torch.sqrt(w.T @ (cov @ w)) * torch.sqrt(torch.tensor([252])).item()

        def Negetive_Sharpe(r, w, cov):
            return Negetive_R(r, w, cov) / Std(r, w, cov)

        '''target must in ['Sharpe','Return','Std']'''
        loss_dictionary = {
            'Sharpe': Negetive_Sharpe,
            'Return': Negetive_R,
            'Std': Std
        }
        '''we need translate target into loss fucntion in torch'''
        loss = loss_dictionary[target]
        '''init weights'''
        weights = torch.randn(len(stock), requires_grad=True)
        train_loss = list()
        for epoch in range(EPOCH):
            w = torch.softmax(weights, dim=0)
            w = w.resize(w.size(0), 1)
            l = loss(r, w, cov)
            l.backward()
            weights.data -= weights.grad * LR
            weights.grad.zero_()
            train_loss.append(l.item())
        print('-----------------------------Optimization Result-----------------------------')
        print(f'target:{target}, loss:', l.item())
        plt.plot(train_loss)
        plt.title('Loss Figure')
        plt.xlabel('EPOCH')
        plt.ylabel('LOSS')
        plt.show()
        weights_values = torch.softmax(weights, dim=0).detach().numpy().reshape(-1, len(stock))
        weights_table = pd.DataFrame(weights_values, columns=stock, index=['weights'])
        print(weights_table)


'''test example using this tool'''
if __name__=='__main__':
    stock = ['AAPL', 'AMZN', 'MSFT', 'YHOO']
    data = pd.read_excel("data.xlsx")
    PortOpt = PortfolioOptimize()
    PortOpt.set(data=data, stock_list=stock)
    PortOpt.analysis()
    PortOpt.optimize(target='Sharpe', EPOCH=1001, LR=0.1)