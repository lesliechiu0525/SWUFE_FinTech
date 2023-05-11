'''FamaMacbeth Rolling Model'''
import numpy as np

class FamaMacbeth():
    def __init__(self,method='equal'):
        self.data=None
        self.coef=None
        self.intercept=None
        self.method=method

    def set_data(self,data):
        self.data=data

    def fit(self,feature_name,sample_use=24):
        method=self.method
        coef=np.zeros(shape=(sample_use,len(feature_name)))
        intercept=np.zeros(shape=(sample_use,1))
        data=self.data.copy()
        date_range=[i for i in data["trade_date"].unique()]
        date_choice=date_range[:sample_use]
        for i in range(len(date_choice)):
            df_month=data.loc[data["trade_date"]==date_range[i],:]
            df_month['intercept']=1
            x=df_month[['intercept']+feature_name].values
            y=df_month["return_t+1"].values
            beta=np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)
            coef[i,:]=beta[1:]
            intercept[i,:]=beta[0]
        '''equal weights for any cross-section in rolling window'''
        if method=="equal":
            final_coef=coef.mean(axis=0)
            final_intercept=intercept.mean(axis=0)
            self.coef=final_coef
            self.intercept=final_intercept
            '''weights decay exp'''
        elif method=='near_softmax_weight':
            weight=np.array(range(1,sample_use+1))
            #softmax
            exp_weight = np.exp(weight)
            sum_exp_weight = np.sum(exp_weight)
            softmax_weight=exp_weight / sum_exp_weight
            final_coef=np.average(coef,axis=0,weights=softmax_weight)
            final_intercept=np.average(intercept,axis=0,weights=softmax_weight)
            self.coef=final_coef
            self.intercept=final_intercept
            '''weights decay linear'''
        elif method=='near_linear_weight':
            weight=np.array(range(1,sample_use+1))
            linear_weight=weight/np.sum(weight)
            final_coef=np.average(coef,axis=0,weights=linear_weight)
            final_intercept=np.average(intercept,axis=0,weights=linear_weight)
            self.coef=final_coef
            self.intercept=final_intercept
        return self.coef,self.intercept
    '''use final FM coef to predict'''
    def predict(self,x):
        return x.dot(self.coef)+self.intercept