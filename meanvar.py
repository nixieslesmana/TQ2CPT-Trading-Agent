import pandas as pd
import numpy as np

from tqdm import tqdm
from preprocessing.preprocessors import data_split
from pypfopt  import EfficientFrontier
from pypfopt.risk_models import CovarianceShrinkage, risk_matrix
from pypfopt.expected_returns import mean_historical_return

df=pd.read_csv('data/dow_30_2009_2020.csv')

idx_list = pd.to_datetime(df.datadate.unique().astype(str), format='%Y%m%d')
tic_list = df.tic.unique()
tic_list.sort()

rebalance_window = 63
validation_window = 63
unique_trade_date = df[(df.datadate > 20151001)&(df.datadate <= 20200707)].datadate.unique()

df_trade_date = pd.DataFrame({'datadate':unique_trade_date})
dir_ = 'sALL_cpt_ent1_k0_t5_preprocTa/'

def get_account_value(model_name, rebalance_window, validation_window, unique_trade_date, df_trade_date):
    df_account_value=pd.DataFrame()
    for i in range(rebalance_window+validation_window, len(unique_trade_date)+1,rebalance_window):
        temp = pd.read_csv('results/'+dir_+'account_value_trade_{}_{}.csv'.format(model_name,i))
        df_account_value = df_account_value.append(temp,ignore_index=True)
    df_account_value = pd.DataFrame({'account_value':df_account_value['0']})
    sharpe=(252**0.5)*df_account_value.account_value.pct_change(1).mean()/df_account_value.account_value.pct_change(1).std()
    print(sharpe)
    df_account_value=df_account_value.join(df_trade_date[63:].reset_index(drop=True))
    return df_account_value

def annualized_sharpe(returns_df: pd.DataFrame, risk_free_rate: float) -> pd.Series:
    """
    Compute the annualized Sharpe ratio of a DataFrame of daily returns
    :param returns_df: DataFrame of daily returns of different strategies (n_samples, n_strategies)
    :param risk_free_rate: risk-free rate
    :return: annualized sharpe ratio of each strategy (n_strategies)
    """
    daily_returns = returns_df.mean()
    daily_volatility = returns_df.std()
    sharpe_ratio = (daily_returns - risk_free_rate) / daily_volatility
    annualized_sharpe = np.sqrt(252) * sharpe_ratio
    return annualized_sharpe

###############################################################################

def compute_meanvar():

    mean_var_pf = pd.DataFrame(index=idx_list, columns=tic_list)

    for i in tqdm(range(rebalance_window + validation_window, len(unique_trade_date), rebalance_window)):
        train = data_split(df, start=20090000, end=unique_trade_date[i - rebalance_window]) # - validation_window]) #])
        
        '''
        unique_train_date = df[(df.datadate > 20081231)&(df.datadate <= 20200707)].datadate.unique()
        
        if i - rebalance_window - validation_window == 0:
            initial = True
        else:
            initial = False
        
        if initial: # i - 63*2 - rebalance_window - validation_window < 0
            start_date = 20090101
            start_id = 0
        else:
            #start_date += 63 #unique_trade_date[i - 63 - rebalance_window - validation_window]
            start_id += 63
            start_date = unique_train_date[start_id]
    
        train = data_split(df, start = start_date, end = unique_trade_date[i - rebalance_window]) #])
        '''
        
        ## validation env
        trade = data_split(df, start=unique_trade_date[i - rebalance_window], # - validation_window], #],
                                end=unique_trade_date[i]) # - rebalance_window]) #])
        
        #print(unique_trade_date[i])
        #raise ValueError()
        train['adjcp'] = train['prccd'] / train['ajexdi']
        trade['adjcp'] = trade['prccd'] / trade['ajexdi']
        train = train.pivot_table(values='adjcp', columns='tic', index='datadate')
        trade = trade.pivot_table(values='adjcp', columns='tic', index='datadate')
        
        train.index = pd.to_datetime(train.index.astype(str), format='%Y%m%d')
        trade.index = pd.to_datetime(trade.index.astype(str), format='%Y%m%d')
    
        mu = mean_historical_return(train)
        S = CovarianceShrinkage(train).ledoit_wolf()
        ef = EfficientFrontier(mu, S)
        weights = ef.max_sharpe()
        cleaned_weights = ef.clean_weights()
        row = list(cleaned_weights.values())
        for d in trade.index.astype(str):
            mean_var_pf.loc[d , :] = row
    
    mean_var_pf = mean_var_pf.dropna()
    
    mean_var_pf_data = df.copy()
    mean_var_pf_data['adjcp'] = mean_var_pf_data['prccd'] / mean_var_pf_data['ajexdi']
    mean_var_pf_data = mean_var_pf_data.pivot_table(values='adjcp', columns='tic', index='datadate')
    mean_var_pf_data.index = pd.to_datetime(mean_var_pf_data.index.astype(str), format='%Y%m%d')
    
    mean_var_pf_prices = mean_var_pf_data.loc[mean_var_pf.index]
    mean_var_pf_prices_rets_tmp = mean_var_pf_prices.pct_change()
    mean_var_pf_turnover = abs(mean_var_pf_prices.multiply(mean_var_pf) - mean_var_pf_prices.shift(1).multiply(mean_var_pf).multiply(mean_var_pf_prices_rets_tmp+1).shift(-1)) *0.0001
    mean_var_pf_turnover = mean_var_pf_turnover.sum(axis=1)
    
    mean_var_pf_data_rets = mean_var_pf_data.pct_change()
    
    mean_var_pf_data_rets = mean_var_pf_data_rets.loc[mean_var_pf.index, :]
    mean_var_pf_data_rets = mean_var_pf_data_rets.multiply(mean_var_pf)
    mean_var_pf_data_rets = mean_var_pf_data_rets.sum(axis=1) - mean_var_pf_turnover
    
    mean_var_pf_data_rets[0] = 0.
    #print(mean_var_pf_data_rets)
    
    mean_var_cum = mean_var_pf_data_rets +1
    mean_var_cum = mean_var_cum.cumprod()
    #mean_var_cum.plot()
    
    #print(mean_var_cum)
    
    returns_df = pd.DataFrame() #tqc_cpt_actor.index)
    returns_df = returns_df.ffill()
    returns_df = returns_df.pct_change(1)
    returns_df['mean_var'] = mean_var_pf_data_rets
    
    #print(returns_df)
    #print(annualized_sharpe(returns_df, risk_free_rate=0))
    
    compare_df = returns_df.copy()
    compare_df+=1
    compare_df = compare_df.cumprod()
    compare_df *= 1e6
    #print(compare_df) # account_val
    # plt.plot(compare_df)
    
    return mean_var_pf_data_rets, compare_df

#meanvar_rets, meanvar_acct_val = compute_meanvar()
#annualized_sharpe(meanvar_rets, risk_free_rate=0)

'''
############
returns_df = returns_df.dropna()
print(returns_df)
print(annualized_sharpe(returns_df, risk_free_rate=0))'''
