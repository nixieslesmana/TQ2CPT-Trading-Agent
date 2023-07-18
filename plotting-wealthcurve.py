import pandas as pd
import numpy as np
import torch as th
import quantstats as qs
#import riskfolio as rp
import empyrical as ep
from meanvar import compute_meanvar

import matplotlib
import matplotlib.pyplot as plt 
matplotlib.use('Agg')
#%matplotlib inline

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio


import os
os.environ['KMP_DUPLICATE_LIB_OK']='True' # silence error lbmp5md.dll

'''
##########CPT PLOTS

alpha = 88, lambda = 2.25
alpha = 95, lambda = 1.5

[WHICH IS MORE LOSS AVERSE?]

###########BOX PLOT

'''

dir_ = '00 MAIN DATA/' #'sALL_ensemble_t5_preprocTa/' #'sALL_mean_t5_preprocTa/'
def get_daily_return(df):
    df['daily_return']=df.account_value.pct_change(1)    #shift=1
    #Compute daily return: exactly same as SP500['daily_return'] = (SP500['sp500']/ SP500['sp500'].shift(1)) -1
    #df=df.dropna()
    print('Sharpe: ',(252**0.5)*df['daily_return'].mean()/ df['daily_return'].std())
    return df

def get_account_value(model_name, rebalance_window, validation_window, 
                      unique_trade_date, df_trade_date, dir_ = dir_):
    df_account_value=pd.DataFrame()
    for i in range(rebalance_window+validation_window, len(unique_trade_date)+1,rebalance_window):
        temp = pd.read_csv('results/'+dir_+'account_value_trade_{}_{}.csv'.format(model_name,i))
        df_account_value = df_account_value.append(temp,ignore_index=True)
    df_account_value = pd.DataFrame({'account_value':df_account_value['0']})
    sharpe=(252**0.5)*df_account_value.account_value.pct_change(1).mean()/df_account_value.account_value.pct_change(1).std()
    #print(sharpe)
    df_account_value=df_account_value.join(df_trade_date[63:].reset_index(drop=True))
    
    #print(df_account_value.shape)
    return df_account_value

def mean_var_return(daily_return_df):
    plt.style.use('ggplot') #Change/Remove This If you Want

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(daily_return_df.mean(axis=1), alpha=0.5, color='red', label='mean cumulative return', linewidth = 1.0)
    print(daily_return_df.index.values)
    ax.fill_between(daily_return_df.index.values, daily_return_df.mean(axis=1) - daily_return_df.std(axis=1), daily_return_df.mean(axis=1) + daily_return_df.std(axis=1), color='#888888', alpha=0.4)
    ax.fill_between(daily_return_df.index.values, daily_return_df.mean(axis=1) - 2*daily_return_df.std(axis=1), daily_return_df.mean(axis=1) + 2*daily_return_df.std(axis=1), color='#888888', alpha=0.2)
    ax.legend(loc='best')
    #ax.set_ylim([-0.04,0.04])
    ax.set_ylabel("Cumulative Returns")
    ax.set_xlabel("Time")

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

######################################## START ################################

df=pd.read_csv('data/dow_30_2009_2020.csv')

rebalance_window = 63
validation_window = 63
unique_trade_date = df[(df.datadate > 20151001)&(df.datadate <= 20200707)].datadate.unique()
df_trade_date = pd.DataFrame({'datadate':unique_trade_date})
quantiles = 3  

returns_df = None

'''
algo = 'cpt' #'ensemble' # 'cpt'
actor_mode, cpt_type = 'cpt', 88 #None, None #'cpt', 88 | 'cpt', 95
ent_coef = 1 #"auto"
k = 0 #0 #1
'''

timesteps = 5000

dji_acct_val = None
meanvar_acct_val = None
dji_rets = None
meanvar_rets = None

color = {'cpt88': 'red', 'cpt95': 'orange', 'cptNone': 'green', 
         'ensemble': 'blue', #'ensembleFalse': 'magenta',
         'mean-var': 'tab:olive', 'dji': 'black'}
linestyle = ['-', '--', '-.', ':']

minSharpe = .45
maxSharpe = 1.65

for seed in [99172, 71108, 18660, 1310, 1701, 29171, 13102, 79665, 61282, 58992]:
    turb_var = .9 #.9 #.95

    plt.figure(figsize=(25, 7))
    
    for algo in ['cpt']: 
        #['cpt', 'ensemble', 'mean-var', 'dji']:
        switch = 1    
    
        if algo == 'cpt':
            
            preproc = 'Ta'
            n_quantiles = 25
            n_critics = 3 # 5
            
            for actor_mode, cpt_type in [(None, None)]: 
                #[('cpt', 88), ('cpt', 95), (None, None)]:
                
                styleID = 0
                
                for ent_coef, k in [("auto", 1)]: 
                    #[(1., 0), ("auto", 1)]:
                    try:
                        print(seed, turb_var, algo, ent_coef, k, actor_mode, cpt_type)
    
                        if ent_coef == "auto":
                            entID = 0
                        else:
                            entID = int(ent_coef)
                        
                        name = f'tqcA{actor_mode}E{entID}C{n_critics}K{k}T{timesteps//1000}S{seed}' #f'tqc_actor_{quantiles}c_{seed}'
                        
                        dir_ = '00 MAIN DATA/'
                        dir_ += 'E{}-K{}/{}/'.format(entID, k, cpt_type)
                        
                        if turb_var == .9:
                            turbID = int(100 * turb_var)
                            name += f'Tu{turbID}'
                        
                        tqc_cpt_actor = get_account_value(name, rebalance_window, validation_window, 
                                                          unique_trade_date, df_trade_date, dir_ = dir_)
                        tqc_cpt_actor.columns = ['account_value','Date']
                        tqc_cpt_actor['Date'] = pd.to_datetime(tqc_cpt_actor['Date'].astype(str), format='%Y%m%d')
                        tqc_cpt_actor.set_index('Date', inplace=True)
                        
                        if actor_mode == None:
                            algoname = 'TQC'
                        else:
                            algoname = 'TQ2CPT-' + str(cpt_type)
                            
                        algo_rets = tqc_cpt_actor['account_value']
                        algo_rets = algo_rets.ffill()
                        algo_rets = algo_rets.pct_change(1)
                        sharpe = annualized_sharpe(algo_rets, 0)
                        
                        sharpe_norm = (sharpe - minSharpe) / (maxSharpe - minSharpe)
                        
                        print(sharpe, sharpe_norm)
                        
                        plt.plot(tqc_cpt_actor, color = color[algo+str(cpt_type)], alpha = sharpe_norm, linestyle = linestyle[styleID],
                                 label = algoname +'(ent_coef={}, k={}), Sharpe={}'.format(ent_coef, k, np.round(sharpe, 2)))
                        
                        styleID += 1
                        
                    except:
                        
                        switch = 0
                        styleID += 1
                        
                        continue
            
        elif algo == 'ensemble':
            
            dir_ = '00 MAIN DATA/'
            
            styleID = 0
            
            for preproc in ['Ta', False]:
                #['Ta', False]: #, False]:
                
                print(seed, turb_var, algo, preproc)

                name = f'ensembleT{timesteps//1000}S{seed}P{preproc}'
                
                if turb_var == .9:
                    turbID = int(100 * turb_var)
                    name += f'Tu{turbID}'
                #f'ensemble_{seed}'
                ensemble_account_value = get_account_value(name, rebalance_window, validation_window, 
                                                           unique_trade_date, df_trade_date, dir_ = dir_)
                ensemble_account_value.columns = ['account_value','Date']
                ensemble_account_value['Date'] = pd.to_datetime(ensemble_account_value['Date'].astype(str), format='%Y%m%d')
                ensemble_account_value.set_index('Date', inplace=True)
            
                tqc_cpt_actor = ensemble_account_value
                
                if preproc == 'Ta':
                    algoname = 'Ensemble-S'
                else:
                    algoname = 'Ensemble'
                
                algo_rets = tqc_cpt_actor['account_value']
                algo_rets = algo_rets.ffill()
                algo_rets = algo_rets.pct_change(1)
                sharpe = annualized_sharpe(algo_rets, 0)
                sharpe_norm = (sharpe - minSharpe) / (maxSharpe - minSharpe)

                print(sharpe, sharpe_norm)
                
                plt.plot(tqc_cpt_actor, color = color[algo], alpha = sharpe_norm, linestyle = linestyle[styleID],
                         label = algoname + ', Sharpe={}'.format(np.round(sharpe, 2)))
                
                styleID += 1

        
        elif algo == 'mean-var':
            if meanvar_acct_val is None:
                meanvar_rets, meanvar_acct_val = compute_meanvar()
            
            algo_rets = meanvar_rets
            sharpe = annualized_sharpe(algo_rets, 0)
            sharpe_norm = (sharpe - minSharpe) / (maxSharpe - minSharpe)

            print(sharpe, sharpe_norm)
            plt.plot(meanvar_acct_val, color = color[algo], alpha = sharpe_norm,
                     label = 'Mean-Var, Sharpe={}'.format(np.round(sharpe, 2)))
                
            #raise NotImplementedError()
            
        elif algo == 'dji':
            
            if dji_acct_val is None:

                dji_df = pd.read_csv("data/^DJI.csv")
                dji_df['Date'] = pd.to_datetime(dji_df['Date'], format='%d/%m/%Y')
                dji_df.set_index('Date', inplace=True)
                
                dji_rets = dji_df['Adj Close'].pct_change(1)
                dji_rets = dji_rets.ffill()
                #dji_rets = dji_rets.pct_change(1)
                #dji_rets[0] = 0
                #dji_rets += 1
                #dji_rets.index # from 2009
                #tqc_cpt_actor.index # from 2016

                dji_rets = dji_rets[tqc_cpt_actor.index]
                dji_rets[0] = 0
                print(annualized_sharpe(dji_rets, risk_free_rate=0))

                dji_acct_val = dji_rets + 1
                dji_acct_val = dji_acct_val.cumprod()
                dji_acct_val *= 1e6
                
                # dji_acct_val = compare_df
                
            algo_rets = dji_rets
            sharpe = annualized_sharpe(algo_rets, 0)
            sharpe_norm = (sharpe - minSharpe) / (maxSharpe - minSharpe)

            print(sharpe, sharpe_norm)
            
            plt.plot(dji_acct_val, color = color[algo], alpha = sharpe_norm,
                     label = 'DJI, Sharpe={}'.format(np.round(sharpe, 2)))
            
            #raise NotImplementedError()
        
        else:
            
            raise NotImplementedError()
        
        if returns_df is None:
            returns_df = pd.DataFrame(index=tqc_cpt_actor.index)
        
        if switch == 1:
            returns_df['tqc_cpt_{}'.format(seed)] = tqc_cpt_actor['account_value']
        
    plt.legend()
    if turb_var == .9:
        plt.ylim([.9*1e6, 2.*1e6])
    else:
        plt.ylim([.9*1e6, 2.*1e6])
    
    plt.xlim([tqc_cpt_actor.index[0], tqc_cpt_actor.index[-1]])
    plt.savefig('./output/wealth_curve_{}_{}.png'.format(seed, turb_var), dpi=200)
    plt.close()
    
dji_df = pd.read_csv("data/^DJI.csv")
dji_df['Date'] = pd.to_datetime(dji_df['Date'], format='%d/%m/%Y')
dji_df.set_index('Date', inplace=True)

'''
#########

dji_rets = dji_df['Adj Close'].pct_change(1)
dji_rets = dji_rets.ffill()
#dji_rets = dji_rets.pct_change(1)
#dji_rets[0] = 0
#dji_rets += 1
#dji_rets.index # from 2009
#tqc_cpt_actor.index # from 2016

dji_rets = dji_rets[tqc_cpt_actor.index]
dji_rets[0] = 0
print(annualized_sharpe(dji_rets, risk_free_rate=0))

dji_acct_val = dji_rets + 1
dji_acct_val = dji_acct_val.cumprod()
dji_acct_val *= 1e6

# plt.plot(dji_acct_val)

######

dji_df['dji_returns'] = dji_df['Adj Close'].pct_change(1) #Compute daily returns of DJIA
dji_df = dji_df.dropna()
dji_df.tail()
#returns_df['dji'] = dji_df['Adj Close']
'''

returns_df = returns_df.ffill()
returns_df = returns_df.pct_change(1)

#annualized_sharpe(returns_df, risk_free_rate=0)

'''
##################

alpha_ = 0.95
lambda_ = 1.5
rho1, rho2 = 0.5, 0.5
b_ = 0

def compute_CPT(tensor, sort = True, B=b_, alpha_=alpha_, lambda_ = lambda_,
                rho1 = rho1, rho2 = rho2):
   
    #print('inside compute_CPT, params:', alpha_, rho1, lambda_, B)
   
    if sort:
        tensor, _ = th.sort(tensor)

    quantiles = tensor
    
    utilities = th.where(quantiles >= B, ((quantiles-B).abs())**alpha_, -lambda_ * ((B-quantiles).abs())**alpha_)
    
    batchSize = tensor.shape[0] #1 # default, by implementation
    supportSize = quantiles.shape[-1] # len(quantiles)
    supportTorch = th.linspace(0.0, 1.0, 1 + supportSize)
    tausPos1 = (1 - supportTorch[:-1]).repeat(batchSize, 1).view(batchSize, 1, -1) # dim = ???
    tausPos2 = (1 - supportTorch[1:]).repeat(batchSize, 1).view(batchSize, 1, -1)
    tausNeg1 = supportTorch[1:].repeat(batchSize, 1).view(batchSize, 1, -1)
    tausNeg2 = supportTorch[:-1].repeat(batchSize, 1).view(batchSize, 1, -1)
    
    weightedProbs = th.where(quantiles >= B, 
                             tausPos1**rho1 / ((tausPos1**rho1 + (1-tausPos1)**rho1)**(1/rho1)) - tausPos2**rho1 / ((tausPos2**rho1 + (1-tausPos2)**rho1)**(1/rho1)), 
                             tausNeg1**rho2 / ((tausNeg1**rho2 + (1-tausNeg1)**rho2)**(1/rho2))  - tausNeg2**rho2 / ((tausNeg2**rho2 + (1-tausNeg2)**rho2)**(1/rho2)))

    CPT_val = (utilities * weightedProbs).sum(-1) #.sum(2)

    return CPT_val # dim: (batchSize, 1)

##################
'''

def assess_pf(df):
    idx = list(df.columns)
    res_df = pd.DataFrame(index=idx, columns=['Sharpe', 'Cum Rets', 'CAGR', 'Ann Vol', 'Max DD'])
    #, 'Alpha', 'Beta'])
    
    #res_df = pd.DataFrame(index=idx, columns=['CPT95', 'CPT88', 'Sharpe', 'Cum Rets', 'CAGR', 'Ann Vol', 'Max DD'])
    
    for col in df.columns:
        row = []
        
        #print(df)
        #print(th.tensor(df[col]*252*100)) # (1, 25)
        
        '''
        cpt88= compute_CPT(th.tensor(df[col][1:]*252*100).view(1, -1), alpha_=.88, lambda_=2.25, 
                         rho1=.65, rho2=.65)[0][0].item()
        cpt95= compute_CPT(th.tensor(df[col][1:]*252*100).view(1, -1), alpha_=.95, lambda_=1.5, 
                         rho1=.5, rho2=.5)[0][0].item()
        
        print(cpt88)
        '''
        sharpe = qs.stats.sharpe(df[col])
        cum_rets = qs.stats.comp(df[col]) 
        cagr = qs.stats.cagr(df[col])   
        vol = qs.stats.volatility(df[col])   
        max_drawdown = qs.stats.max_drawdown(df[col])
        #greeks = qs.stats.greeks(df[col], df['dji']) 
        #alpha, beta = greeks['alpha'], greeks['beta']
        
        row = sharpe, cum_rets, cagr, vol, max_drawdown#, alpha, beta
        #row = cpt88, cpt95, sharpe, cum_rets, cagr, vol, max_drawdown#, alpha, beta
        res_df.loc[col, :] = row
        
    return res_df.astype(float)

res_df = assess_pf(returns_df)

res_df[["Cum Rets","CAGR","Ann Vol","Max DD"]] *=100
#res_df = res_df.round(2)

print(res_df.round(2))
print('\n')
print(res_df.describe().round(2))
'''
print('MEAN:', res_df.iloc[:-1, :].mean())
print('MIN:', res_df.iloc[:-1, :].std())
print('MAX:', res_df.iloc[:-1, :].std())
print('STDEV:', res_df.iloc[:-1, :].std())
'''
