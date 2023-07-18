import pandas as pd
import numpy as np
import torch as th
import time
from datetime import datetime

import quantstats as qs
#import riskfolio as rp
import empyrical as ep

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

import gym
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from sb3_contrib_cpt import TQC_CPT

# customized env
from env.EnvMultipleStock_train import StockEnvTrain
from env.EnvMultipleStock_validation import StockEnvValidation
from env.EnvMultipleStock_trade import StockEnvTrade

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True' # silence error lbmp5md.dll

from preprocessing.preprocessors import data_split
from tqdm import tqdm
from utils import *

# RL models from stable-baselines
from stable_baselines3 import PPO, A2C, DDPG
from preprocessing.preprocessors import *
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise

def get_daily_return(df):
    df['daily_return']=df.account_value.pct_change(1)    #shift=1
    #Compute daily return: exactly same as SP500['daily_return'] = (SP500['sp500']/ SP500['sp500'].shift(1)) -1
    #df=df.dropna()
    print('Sharpe: ',(252**0.5)*df['daily_return'].mean()/ df['daily_return'].std())
    return df

def get_account_value(model_name, rebalance_window, validation_window, unique_trade_date, df_trade_date):
    df_account_value=pd.DataFrame()
    for i in range(rebalance_window+validation_window, len(unique_trade_date)+1,rebalance_window):
        temp = pd.read_csv('results/account_value_trade_{}_{}.csv'.format(model_name,i))
        df_account_value = df_account_value.append(temp,ignore_index=True)
    df_account_value = pd.DataFrame({'account_value':df_account_value['0']})
    sharpe=(252**0.5)*df_account_value.account_value.pct_change(1).mean()/df_account_value.account_value.pct_change(1).std()
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


def DRL_prediction(df,
                   model,
                   name,
                   last_state,
                   iter_num,
                   unique_trade_date,
                   rebalance_window,
                   turbulence_threshold,
                   initial):
    ### make a prediction based on trained model###

    ## trading env
    trade_data = data_split(df, start=unique_trade_date[iter_num - rebalance_window], end=unique_trade_date[iter_num])
    env_trade = DummyVecEnv([lambda: StockEnvTrade(trade_data,
                                                   turbulence_threshold=turbulence_threshold,
                                                   initial=initial,
                                                   previous_state=last_state,
                                                   model_name=name,
                                                   iteration=iter_num)])
    obs_trade = env_trade.reset()

    for i in range(len(trade_data.index.unique())):
        action, _states = model.predict(obs_trade)

        obs_trade, rewards, dones, info = env_trade.step(action)
        if i == (len(trade_data.index.unique()) - 2):
            # print(env_test.render())
            last_state = env_trade.render()
    
    df_last_state = pd.DataFrame({'last_state': [last_state]})
    df_last_state.to_csv('results/last_state_{}_{}.csv'.format(name, i), index=False)
    
    return last_state

def DRL_validation(model, test_data, test_env, test_obs) -> None:
    ###validation process###
    for i in range(len(test_data.index.unique())):
        action, _states = model.predict(test_obs)
        test_obs, rewards, dones, info = test_env.step(action)

def get_validation_sharpe(iteration):
    ###Calculate Sharpe ratio based on validation results###
    df_total_value = pd.read_csv('results/account_value_validation_{}.csv'.format(iteration), index_col=0)
    df_total_value.columns = ['account_value_train']
    df_total_value['daily_return'] = df_total_value.pct_change(1)
#     sharpe = (4 ** 0.5) * df_total_value['daily_return'].mean() / \
#              df_total_value['daily_return'].std()
    sharpe = (252 ** 0.5) * df_total_value['daily_return'].mean() / \
             df_total_value['daily_return'].std()
    return sharpe

########################### MODEL DEFINITIONS #################################

def train_A2C(env_train, model_name, preproc_var = True, timesteps=20000000, seed=12345):
    """A2C model"""
    
    global AVG_FEATURE_a2c, STD_FEATURE_a2c
    
    start = time.time()
    
    print('PREPROCESSING..')
    if PREPROC_GLOB in ['Ta', 'Tb']:
        if preproc_var: 
            preproc = A2C('MlpPolicy', env_train, verbose=0, seed=seed,n_steps=256, ent_coef=0.005, learning_rate=0.0002)
            AVG_FEATURE_a2c = th.zeros(181)
            STD_FEATURE_a2c = th.ones(181)
            preproc.policy._update_preprocess(AVG_FEATURE_a2c, STD_FEATURE_a2c)
            
            preproc.learn(total_timesteps=256) # 256 or 5..
            
            print(preproc.policy.addition[:10])
            print(preproc.policy.multiplier[:10])
            
            for rollout_data in preproc.rollout_buffer.get(batch_size=None):
                obs_ = rollout_data.observations
                print(obs_.shape)
                AVG_FEATURE_a2c = th.tensor([obs_[:, d].mean().item() for d in range(181)]).to(th.int32).to(th.float64)
                STD_FEATURE_a2c = th.tensor([obs_[:, d].std().item() for d in range(181)]).to(th.int32)
                STD_FEATURE_a2c = 1/(1 + STD_FEATURE_a2c).to(th.float64)
                
                continue
            
    else:
        AVG_FEATURE_a2c = th.zeros(181)
        STD_FEATURE_a2c = th.ones(181)
    
    print('PREPROC_GLOB:', PREPROC_GLOB)
    print('preproc_var:', preproc_var)
    print('mean:', AVG_FEATURE_a2c[:10])
    print('std:', STD_FEATURE_a2c[:10])
    
    #raise ValueError()

    print('TRAINING..')
    model = A2C('MlpPolicy', env_train, verbose=0, seed=seed,n_steps=5, ent_coef=0.005, learning_rate=0.0002)
    '''
    print('PRE')
    print('mean:', model.policy.addition[:10])
    print('std:', model.policy.multiplier[:10])
    '''
    model.policy._update_preprocess(AVG_FEATURE_a2c, STD_FEATURE_a2c)
    '''
    print('POST')
    print('mean:', model.policy.addition[:10])
    print('std:', model.policy.multiplier[:10])
    '''
    
    model.learn(total_timesteps=timesteps)

    end = time.time()

    return model

def train_PPO(env_train, model_name, preproc_var = True, timesteps=50000, seed=12345):
    """PPO model"""

    global AVG_FEATURE_ppo, STD_FEATURE_ppo
    
    start = time.time()
    
    print('PREPROCESSING..')
    if PREPROC_GLOB in ['Ta', 'Tb']:
        if preproc_var: 
            preproc = PPO('MlpPolicy', env_train, ent_coef = 0.005, seed=seed,n_steps=256, learning_rate=0.0001, batch_size=128)
            AVG_FEATURE_ppo = th.zeros(181)
            STD_FEATURE_ppo = th.ones(181)
            preproc.policy._update_preprocess(AVG_FEATURE_ppo, STD_FEATURE_ppo)
            
            preproc.learn(total_timesteps = 256)
            
            for rollout_data in preproc.rollout_buffer.get(batch_size=None):
                obs_ = rollout_data.observations
                #print(obs_.shape)
                AVG_FEATURE_ppo = th.tensor([obs_[:, d].mean().item() for d in range(181)]).to(th.int32).to(th.float64)
                STD_FEATURE_ppo = th.tensor([obs_[:, d].std().item() for d in range(181)]).to(th.int32)
                
                STD_FEATURE_ppo = 1/(1 + STD_FEATURE_ppo).to(th.float64)
                
                continue
            
    else:
        AVG_FEATURE_ppo = th.zeros(181)
        STD_FEATURE_ppo = th.ones(181)
        
    print('PREPROC_GLOB:', PREPROC_GLOB)
    print('preproc_var:', preproc_var)
    print('mean:', AVG_FEATURE_ppo[:10])
    print('std:', STD_FEATURE_ppo[:10])
    
    #######
    print('TRAINING..')
    model = PPO('MlpPolicy', env_train, ent_coef = 0.005, seed=seed,n_steps=2048, learning_rate=0.0001, batch_size=128)
    
    '''
    print('PRE')
    print('mean:', model.policy.addition[:10])
    print('std:', model.policy.multiplier[:10])
    '''  
    model.policy._update_preprocess(AVG_FEATURE_ppo, STD_FEATURE_ppo)
    '''
    print('POST')
    print('mean:', model.policy.addition[:10])
    print('std:', model.policy.multiplier[:10])
    '''
    model.learn(total_timesteps=timesteps)
    end = time.time()

    model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
    
    return model

def train_DDPG(env_train, model_name, preproc_var = True, timesteps=10000, seed=12345):
    """DDPG model"""
    
    global AVG_FEATURE_ddpg, STD_FEATURE_ddpg

    # add the noise objects for DDPG
    n_actions = env_train.action_space.shape[-1]
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))
    
    
    start = time.time()
    
    ##################
    
    print('PREPROCESSING..')
    if PREPROC_GLOB in ['Ta', 'Tb']:
        if preproc_var: 
            preproc = DDPG('MlpPolicy', env_train, action_noise=action_noise, seed=seed, batch_size=128, buffer_size=256, learning_rate=0.001)
            preproc.gradient_steps = 1
           
            AVG_FEATURE_ddpg = th.zeros(181)
            STD_FEATURE_ddpg = th.ones(181)
            preproc.actor._update_preprocess(AVG_FEATURE_ddpg, STD_FEATURE_ddpg)
            preproc.actor_target._update_preprocess(AVG_FEATURE_ddpg, STD_FEATURE_ddpg)
            preproc.critic._update_preprocess(AVG_FEATURE_ddpg, STD_FEATURE_ddpg)
            preproc.critic_target._update_preprocess(AVG_FEATURE_ddpg, STD_FEATURE_ddpg)
            
            preproc.learn(total_timesteps=1)
            
            print(preproc.replay_buffer.size())
            
            # set to NONE 
            # modify preproc to ones, zeros -> check errors
            
            replay_data = preproc.replay_buffer.sample(256, env = preproc._vec_normalize_env)
            obs_ = replay_data.observations
            print(obs_.shape)
            AVG_FEATURE_ddpg = th.tensor([obs_[:, d].mean().item() for d in range(181)]).to(th.int32).to(th.float64)
            STD_FEATURE_ddpg = th.tensor([obs_[:, d].std().item() for d in range(181)]).to(th.int32)
            STD_FEATURE_ddpg = 1/(1 + STD_FEATURE_ddpg).to(th.float64)
            
    else:
        AVG_FEATURE_ddpg = th.zeros(181)
        STD_FEATURE_ddpg = th.ones(181)
    
    print('PREPROC_GLOB:', PREPROC_GLOB)
    print('preproc_var:', preproc_var)
    print('mean:', AVG_FEATURE_ddpg[:10])
    print('std:', STD_FEATURE_ddpg[:10])
    
    ################
    
    print('TRAINING..')
    model = DDPG('MlpPolicy', env_train, action_noise=action_noise, seed=seed,batch_size=128, buffer_size=50000, learning_rate=0.001)
    
    # Update preprocessor of obs
    model.actor._update_preprocess(AVG_FEATURE_ddpg, STD_FEATURE_ddpg)
    model.actor_target._update_preprocess(AVG_FEATURE_ddpg, STD_FEATURE_ddpg)
    model.critic._update_preprocess(AVG_FEATURE_ddpg, STD_FEATURE_ddpg)
    model.critic_target._update_preprocess(AVG_FEATURE_ddpg, STD_FEATURE_ddpg)
    #model.policy._update_preprocess
    
    model.learn(total_timesteps=timesteps)
    end = time.time()

    model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
    return model

def run_ensemble_strategy(df, unique_trade_date, rebalance_window, validation_window, 
                          timesteps=100, seed=12345, name='ensemble',
                          turb_var = .95) -> None:
    """Ensemble Strategy that combines PPO, A2C and DDPG"""
    print("============Start Ensemble Strategy============")
    
    unique_train_date = df[(df.datadate > 20081231)&(df.datadate <= 20200707)].datadate.unique()
    
    # for ensemble model, it's necessary to feed the last state
    # of the previous model to the current model as the initial state
    last_state_ensemble = []
    ppo_sharpe_list = []
    ddpg_sharpe_list = []
    a2c_sharpe_list = []    #a2c_org_sharpe_list = []
    model_use = []

    insample_turbulence = df[(df.datadate<20151000) & (df.datadate>=20090000)]
    insample_turbulence = insample_turbulence.drop_duplicates(subset=['datadate'])
    insample_turbulence_threshold = np.quantile(insample_turbulence.turbulence.values, turb_var)

    start = time.time()
    for i in range(rebalance_window + validation_window, len(unique_trade_date), rebalance_window):
        print("============================================")
        ## initial state is empty
        if i - rebalance_window - validation_window == 0:
            initial = True
        else:
            initial = False

        # Tuning trubulence index based on historical data
        # Turbulence lookback window is one quarter
        end_date_index = df.index[df["datadate"] == unique_trade_date[i - rebalance_window - validation_window]].to_list()[-1]
        start_date_index = end_date_index - 30*validation_window + 1
        
        historical_turbulence = df.iloc[start_date_index:(end_date_index + 1), :]
        historical_turbulence = historical_turbulence.drop_duplicates(subset=['datadate'])
        historical_turbulence_mean = np.mean(historical_turbulence.turbulence.values)

        if historical_turbulence_mean > insample_turbulence_threshold:
            # if the mean of the historical data is greater than the 95% quantile of insample turbulence data
            # then we assume that the current market is volatile,
            # therefore we set the 95% quantile of insample turbulence data as the turbulence threshold
            # meaning the current turbulence can't exceed the 95% quantile of insample turbulence data
            turbulence_threshold = insample_turbulence_threshold
        else:
            # if the mean of the historical data is less than the 90% quantile of insample turbulence data
            # then we tune up the turbulence_threshold, meaning we lower the risk
            turbulence_threshold = np.quantile(insample_turbulence.turbulence.values, 1)
        print("turbulence_threshold: ", turbulence_threshold)

        ############## Environment Setup starts ##############
        ## training env
        
        ################################# NEW #################################
        # IMPLEMENT: RELOAD W NEW DATA | REINIT INCLUDE PREV WINDOW | REINIT KEEP BUFFER FROM PREVIOUS WINDOW
        # GOAL: FOR EACH WINDOW, NEEDS TO USE NEW DATA
        if initial: # i - 63*2 - rebalance_window - validation_window < 0
            start_date = 20090101
            start_id = 0
        else:
            #start_date += 63 #unique_trade_date[i - 63 - rebalance_window - validation_window]
            start_id += 63
            start_date = unique_train_date[start_id]
        
        train = data_split(df, start = start_date, end = unique_trade_date[i - rebalance_window - validation_window]) #])
        ### OLD: train = data_split(df, start=20090101, end=unique_trade_date[i - rebalance_window - validation_window])
        env_train = DummyVecEnv([lambda: StockEnvTrain(train)])
        
        ## validation env
        validation = data_split(df, start=unique_trade_date[i - rebalance_window - validation_window],
                                end=unique_trade_date[i - rebalance_window])
        env_val = DummyVecEnv([lambda: StockEnvValidation(validation,
                                                          turbulence_threshold=turbulence_threshold,
                                                          iteration=i)])
        obs_val = env_val.reset()
        ############## Environment Setup ends ##############

        ############## Training and Validation starts ##############
        print("======Model training from: ", 20090000, "to ",
              unique_trade_date[i - rebalance_window - validation_window])
        # print("training: ",len(data_split(df, start=20090000, end=test.datadate.unique()[i-rebalance_window]) ))
        # print("==============Model Training===========")
        
        print('A2C, tsteps=', timesteps)
        if PREPROC_GLOB == 'Tb':
            model_a2c = train_A2C(env_train, model_name="A2C_30k_dow_{}".format(i), 
                                  preproc_var = initial,
                                  timesteps=timesteps, seed=seed)
        else: # 'Ta', False
            model_a2c = train_A2C(env_train, model_name="A2C_30k_dow_{}".format(i), timesteps=timesteps, seed=seed)
        
        DRL_validation(model=model_a2c, test_data=validation, test_env=env_val, test_obs=obs_val)
        sharpe_a2c = get_validation_sharpe(i)
        
        print('PPO') #### TOT STEPS??
        if PREPROC_GLOB == 'Tb':
            model_ppo = train_PPO(env_train, model_name="PPO_100k_dow_{}".format(i),
                                  preproc_var = initial,
                                  timesteps=timesteps, seed=seed)
        else:
            model_ppo = train_PPO(env_train, model_name="PPO_100k_dow_{}".format(i), timesteps=timesteps, seed=seed)

        DRL_validation(model=model_ppo, test_data=validation, test_env=env_val, test_obs=obs_val)
        sharpe_ppo = get_validation_sharpe(i)
        
        print('DDPG') ### MAY CHANGE PREPROC TO FALSE (not much affected!)
        if PREPROC_GLOB == 'Tb':
            model_ddpg = train_DDPG(env_train, model_name="DDPG_10k_dow_{}".format(i),
                                    preproc_var = initial,
                                    timesteps=timesteps, seed=seed)
        else:
            model_ddpg = train_DDPG(env_train, model_name="DDPG_10k_dow_{}".format(i), timesteps=timesteps, seed=seed)
        
        DRL_validation(model=model_ddpg, test_data=validation, test_env=env_val, test_obs=obs_val)
        sharpe_ddpg = get_validation_sharpe(i)

        ppo_sharpe_list.append(sharpe_ppo)
        a2c_sharpe_list.append(sharpe_a2c)
        ddpg_sharpe_list.append(sharpe_ddpg)

        # Model Selection based on sharpe ratio
        if (sharpe_ppo >= sharpe_a2c) & (sharpe_ppo >= sharpe_ddpg):
            model_ensemble = model_ppo
            model_use.append('PPO')
        elif (sharpe_a2c > sharpe_ppo) & (sharpe_a2c > sharpe_ddpg):
            model_ensemble = model_a2c
            model_use.append('A2C')
        else:
            model_ensemble = model_ddpg
            model_use.append('DDPG')
        
        ############ Training and Validation ends, Trading starts #############
        
        print("======Trading from: ", unique_trade_date[i - rebalance_window], "to ", unique_trade_date[i])
        print("Used Model: ", model_ensemble)
        
        last_state_ensemble = DRL_prediction(df=df, model=model_ensemble, name=name,
                                            last_state=last_state_ensemble, iter_num=i,
                                            unique_trade_date=unique_trade_date,
                                            rebalance_window=rebalance_window,
                                            turbulence_threshold=turbulence_threshold,
                                            initial=initial)
        print("============Trading Done============")
        
        ############## Trading ends ##############

    end = time.time()
    print("Ensemble Strategy took: ", (end - start) / 60, " minutes")

def run_once(seed, algo, turb_var, PREPROC_GLOB):
    
    global PREPROC_GLOB

    preprocessed_path = "done_data.csv"
    data = pd.read_csv(preprocessed_path, index_col=0)
    unique_trade_date = data[(data.datadate > 20151001)&(data.datadate <= 20200707)].datadate.unique()
    rebalance_window = 63
    validation_window = 63
    
    timesteps = 5000 #2000 #5000 #1000, self.terminal of envTrain hit when self.day >= 1761
    
    if algo == 'cpt':
        
        print('For cpt, run main_CPT.py instead')
        raise NotImplementedError()
        
        '''
        learning_starts = 256 #256 #200
        # CPT = 88, 95, ..
        n_quantiles = 25
        n_critics = 3 # 5
        ent_coef = 1. #"auto" # 1.#"auto"
        k = 0 #0
        
        actor_mode = 'cpt' #None #'maxent', #None, #'cpt',
   
        if ent_coef == "auto":
            entID = 0
        else:
            entID = int(ent_coef)
           
        run_tqc_cpt_strategy(data, unique_trade_date, rebalance_window, validation_window,
                             timesteps=timesteps, seed=seed, learning_starts=learning_starts,
                             n_quantiles=n_quantiles, n_critics=n_critics, k = k,
                             name= f'tqcA{actor_mode}E{entID}C{n_critics}K{k}T{timesteps//1000}S{seed}',
                             actor_mode=actor_mode, critic_mode=None, ent_coef = ent_coef)'''
        
    elif algo == 'ensemble':
        
        #preprocess = 'Ta' # F, T, Ta (True, adaptive)
        #PREPROC_GLOB = False #False #False # Tb,Ta
        #turb_var = .95
        
        name = f'ensembleT{timesteps//1000}S{seed}P{PREPROC_GLOB}'
        
        if turb_var == .9:
            turbID = int(100 * turb_var)
            name += f'Tu{turbID}'

        run_ensemble_strategy(data, unique_trade_date, rebalance_window, validation_window,
                              timesteps = timesteps, seed = seed, turb_var = turb_var, 
                              name = name)
        
    else:
        raise NotImplementedError()

if __name__ == "__main__":
    for seed in [69152, 10672, 40931,  6280, 17913, 51104, 43136, 35269, 85721, 75927]:
        # [99172, 71108, 1701, 18660, 1310, 29171, 13102, 79665, 61282, 58992]:
        # III: [69152, 10672, 40931,  6280, 17913, 51104, 43136, 35269, 85721, 75927]
        # IV: [28994,  1266, 58514, 79698, 90945, 88887, 79147, 95650, 40488, 34506]
        for turb_var in [.9]:
            for PREPROC_GLOB in ['Ta', False]:
                PREPROC_GLOB = False # for PREPROC_GLOB in ['Ta', False] -- seed SET 1!
                 
                AVG_FEAT_a2c = None
                STD_FEAT_a2c = None
                AVG_FEAT_ppo = None
                STD_FEAT_ppo = None
                AVG_FEAT_ddpg = None
                STD_FEAT_ddpg = None
                
                run_once(seed, 'ensemble', turb_var, PREPROC_GLOB) #<-- 'cpt' not working!
