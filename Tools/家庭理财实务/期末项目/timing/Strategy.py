# -*- coding: utf-8 -*-
# @Time    : 2023/5/11 21:38
# @Author  : LeslieChiu
# @Site    : 
# @File    : Strategy.py
# @Software: PyCharm
'''code from IndexPrediction Project'''

'''define strategy and backtrade'''
import numpy as np
import pandas as pd
import random
from pylab import plt

'''Strategy which is link to the ModelUniverse api'''
class Strategy:
    def __init__(self):
        self.pred = None
    def set(self,pred):
        self.pred = pred
    def generate(self,x):#用于生成信号
        if self.pred is not None:
                return self.pred
        else:
            return [random.choice([1,-1]) for i in range(x.shape[0])]

'''BackTrade is to put strategy in history market'''
class BackTrade:
    def __init__(self):
        self.data = None
        self.strategy = None
        self.percent = None
        self.limit = True
        self.factor_names = None
        self.account = pd.DataFrame(columns=['trade_date','cash','position','value','cost'])
    def fit(self,data,factor_names):
        data['price'] = data['close']/data.iloc[0,:]['close']
        self.data = data
        self.factor_names = factor_names
    def set_strategy(self,strategy):
        self.strategy = strategy
    def run(self,limit,percent):
        factor_names = self.factor_names
        limit = limit
        percent = percent
        account = self.account.copy()
        data = self.data.copy()
        pred = self.strategy.generate(data[factor_names])
        data = data.iloc[12:,:]
        '''根据全预测值大于1的输出信号为1 否则为-1'''
        data['signal'] = [1 if i>1 else -1 for i in pred]
        account.loc[0] = ['day0',1000,0,1000,0]
        for idx in range(data.shape[0]):
            account_info = account.iloc[idx,:]
            cash, position, cost = account_info['cash'],\
                                   account_info['position'],account_info['cost']
            info = data.iloc[idx,:]
            if info['signal'] == 0: #close 用来控制止盈损和爆仓
                if position !=0:
                    cash += position*info['price']
                    position,cost =0,0
                    # print(info['trade_date'],\
                    #       'close the position with{}'.format(info['price']))
            elif info['signal'] == 1:#long
                if position<0:
                    cash += position * info['price']
                    position,cost =0,0
                    # print(info['trade_date'], \
                    #       'close the position with{}'.format(info['price']))
                elif position == 0:
                    position = cash*percent//info['price']
                    cash -= position*info['price']
                    cost = info['price']
                    # print(info['trade_date'], \
                    #       'open long position with cost{}'.format(info['price']))
            elif info['signal'] == -1:
                if position>0:
                    cash += position* info['price']
                    position,cost =0,0
                    # print(info['trade_date'], \
                    #       'close the position with{}'.format(info['price']))
                elif not limit:#主动开空仓有限制
                    if position == 0:
                        position = -1*cash * percent // info['price']
                        cash -= position * info['price']
                        cost = info['price']
                        # print(info['trade_date'], \
                        #       'open short position with cost{}'.format(info['price']))

            value = cash+position*info['price']
            account.loc[idx+1] = [info['trade_date'],cash,position,value,cost]#
        account['return'] = account['value']/account.shift(1)['value']
        account.dropna(how='any',inplace=True)
        '''strategy performance plot'''
        fig = plt.figure()
        plt.title('Strategy Performance')
        plt.plot(account['trade_date'], account['return'].cumprod(), label='strategy_value')
        plt.plot(data['trade_date'], data['return'].cumprod(), label='Index Value')
        plt.legend()
        plt.show()
        '''summary table'''
        summary = pd.DataFrame(columns=['年化收益','月均收益','月收益波动率',\
                                        '年化夏普比率','最大回撤'])
        account0 = account.copy()
        account0.index = account0['trade_date']
        summary['年化收益'] = account0['return'].resample('Y').prod()
        summary['月均收益'] = account0['return'].resample('Y').mean()
        summary['月收益波动率'] = account0['return'].resample('Y').std()
        summary['年化夏普比率'] = (summary['月均收益']-1)/summary['月收益波动率']* np.power(12, 1 / 2)
        data_list = [group for _,group in account0.resample('Y')]
        down = None
        for data in data_list:
            cum_return = data['return'].cumprod()
            cum_return = cum_return/cum_return[0]
            res = cum_return.cummax()-cum_return
            down = pd.concat([down,res])
        summary['最大回撤'] = down.resample('Y').max()
        func = lambda x: x.apply(lambda x0:'{:.2%}'.format(x0-1))
        summary[['年化收益','月均收益']]=\
            summary[['年化收益','月均收益']].apply(func)
        summary['最大回撤'] = summary['最大回撤'].apply(lambda x0:'{:.2%}'.format(x0))
        summary.reset_index(inplace=True)
        summary['trade_date'] = summary['trade_date'].apply(lambda x:x.strftime("%Y"))
        func = lambda x: x.apply(lambda x0: '{:.3}'.format(x0))
        summary[['月收益波动率','年化夏普比率']] = \
            summary[['月收益波动率','年化夏普比率']].apply(func)
        return fig,summary



