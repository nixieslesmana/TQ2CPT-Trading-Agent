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

############################## MODEL DEFINITIONS ##############################

AVG_FEAT = None
STD_FEAT = None

def train_TQC_CPT(env, model_name, preproc_var = True, timesteps=10000, seed=12345, n_quantiles=25, n_critics=2, 
                  learning_starts=100, a=1.0, actor_mode='cpt', critic_mode='cpt', k=2,
                  ent_coef = "auto"):
    
    global AVG_FEAT, STD_FEAT
    
    """TQC_CPT model, please make sure to set timesteps > learning_starts"""

    start = time.time()
    policy_kwargs = dict(n_critics=n_critics, n_quantiles=n_quantiles, critic_mode=critic_mode)

    runID = datetime.now().strftime("%d%m%y%H%M%S")
    
    '''
    ############ new: for reuse of previous window's trained_model ############   
    print('i:', model_name)
    
    if model_name > 2 * 63:
        prev_name = "TQC-k{}c{}q{}s{}_{}_{}_{}".format(k, n_critics, n_quantiles, seed, actor_mode, critic_mode, model_name - 63)
        model = TQC_CPT.load(f"output/{prev_name}", env = env)
        print('RE-LOADED params:', sorted(list(model.actor.parameters_to_vector()))[:3])
        
    else:
        model = TQC_CPT("MlpPolicy", env, top_quantiles_to_drop_per_net=k, verbose=1, policy_kwargs=policy_kwargs, seed=seed, learning_starts=learning_starts, a=a, actor_mode=actor_mode)
    
    model_name = "TQC-k{}c{}q{}s{}_{}_{}_{}".format(k, n_critics, n_quantiles, seed, actor_mode, critic_mode, model_name)
    
    model.name = model_name + '_reload_' + runID
    
    # cek reoloaded hyperparams -- if wrong, input into .load()
    print('check hyperparams:', model.seed, model.top_quantiles_to_drop_per_net, model.actor_mode, model.learning_starts)
    print('critic in policy kwargs:', model.policy_kwargs)
    
    model._current_progress_remaining, model.num_timesteps, model._total_timesteps = 1, 0, 0 # do not matter for constant lr
    print('check lr: ', model._current_progress_remaining, model.lr_schedule(model._current_progress_remaining))
    print('progress inputs:', model.num_timesteps, model._total_timesteps)
    
    print('ent_coef: ', model.ent_coef, model.log_ent_coef.detach())#, th.exp(model.log_ent_coef.detach()))
    ###########################################################################
    '''
    
    if preproc_var: ## then make STD_FEAT, AVG_FEAT global
    
        ####################### preproc model #############################
        print('PREPROCESSING..')
        preproc = TQC_CPT("MlpPolicy", env, top_quantiles_to_drop_per_net=k, verbose=1, policy_kwargs=policy_kwargs, 
                        seed=seed, learning_starts=learning_starts, a=a, actor_mode=actor_mode, ent_coef = ent_coef)
        preproc.name = 'preproc'
        print('grad_steps:', preproc.gradient_steps)
        preproc.gradient_steps = 0
        preproc.learn(total_timesteps = 256+1, log_interval=4)
        
        replay_data = preproc.replay_buffer.sample(256, env=preproc._vec_normalize_env)
        obs_ = replay_data.observations
        AVG_FEAT = th.Tensor([obs_[:, d].mean() for d in range(obs_.shape[1])]).to(th.int32).to(th.float64)
        STD_FEAT = th.Tensor([obs_[:, d].std() for d in range(obs_.shape[1])]).to(th.int32)
        STD_FEAT = 1/ (1 + STD_FEAT).to(th.float64)
        
        print(obs_.shape) #replay_data.observations.shape)
    
    print('preproc_var:', preproc_var)
    print('\nmean:', AVG_FEAT[:10]) # different cz when n_updates = 1, buffer size = 257
    print('std:', STD_FEAT[:10])
    
    ####################### old: re-init model ################################
    model = TQC_CPT("MlpPolicy", env, top_quantiles_to_drop_per_net=k, verbose=1, policy_kwargs=policy_kwargs, 
                    seed=seed, learning_starts=learning_starts, a=a, actor_mode=actor_mode, ent_coef = ent_coef)
    print('\nActor mode:', actor_mode, ' Critic mode:', critic_mode)
    
    model_name = "TQC{}-{}-{}_{}_{}_{}".format(k, n_critics, n_quantiles, actor_mode, critic_mode, model_name)
    model.name = model_name + '_reinit_' + runID
    '''
    print('PRE')
    print('mean:', model.actor.addition[:10])
    print('std:', model.actor.multiplier[:10])
    #print('mean:', model.critic.addition[:10])
    #print('std:', model.critic.multiplier[:10])
    #print('mean:', model.critic_target.addition[:10])
    #print('std:', model.critic_target.multiplier[:10])
    '''
    model.actor._update_preprocess(AVG_FEAT, STD_FEAT)
    model.critic._update_preprocess(AVG_FEAT, STD_FEAT)
    model.critic_target._update_preprocess(AVG_FEAT, STD_FEAT)
    '''
    print('POST')
    print('mean:', model.actor.addition[:10])
    print('std:', model.actor.multiplier[:10])
    #print('mean:', model.critic.addition[:10])
    #print('std:', model.critic.multiplier[:10])
    #print('mean:', model.critic_target.addition[:10])
    #print('std:', model.critic_target.multiplier[:10])
    '''
    print('TRAINING...')
    print('PRE-TRAIN/RE-LOADED params:', sorted(list(model.actor.parameters_to_vector()))[:5])
    ############
    
    # timesteps = 5000 # now: 1000
    model.learn(total_timesteps=timesteps, log_interval=4) # seed, model_name
    end = time.time()

    #print('POST-TRAIN buffer size:', model.replay_buffer.size())
    #print('params:', sorted(list(model.actor.parameters_to_vector()))[:3])
    
    # print LOSSES OF THIS WINDOW: from off-polcy.py
    
    model.save(f"output/{model.name}")

    return model

def run_tqc_cpt_strategy(df, unique_trade_date, rebalance_window, validation_window, 
                         timesteps=10000, seed=12345, n_quantiles=25, n_critics=2, 
                         learning_starts=100, a=1.0, name='default', 
                         actor_mode='cpt', critic_mode='cpt', k=2, ent_coef = "auto",
                         turb_var = .95) -> None:

    #print('Actor_mode:', actor_mode, ', Critic_mode:', critic_mode)
    unique_train_date = df[(df.datadate > 20081231)&(df.datadate <= 20200707)].datadate.unique()
    
    last_state= []
    sharpe_list = []
    model_use = []

    insample_turbulence = df[(df.datadate<20151000) & (df.datadate>=20090000)]
    insample_turbulence = insample_turbulence.drop_duplicates(subset=['datadate'])
    insample_turbulence_threshold = np.quantile(insample_turbulence.turbulence.values, turb_var)

    start = time.time()
    for i in tqdm(range(rebalance_window + validation_window, len(unique_trade_date), rebalance_window)):
        print("============================================")
        if i - rebalance_window - validation_window == 0:
            initial = True
        else:
            initial = False

        end_date_index = df.index[df["datadate"] == unique_trade_date[i - rebalance_window - validation_window]].to_list()[-1]
        start_date_index = end_date_index - 30*validation_window + 1
        
        historical_turbulence = df.iloc[start_date_index:(end_date_index + 1), :]
        print('check start_date:', historical_turbulence.shape)
        
        historical_turbulence = historical_turbulence.drop_duplicates(subset=['datadate'])
        historical_turbulence_mean = np.mean(historical_turbulence.turbulence.values)

        if historical_turbulence_mean > insample_turbulence_threshold:
            turbulence_threshold = insample_turbulence_threshold
        else:
            turbulence_threshold = np.quantile(insample_turbulence.turbulence.values, 1)
        print("Turbulence threshold: ", turbulence_threshold)

        ############## Environment Setup starts ##############
        ## training env        
        
        ################################# NEW #################################
        # IMPLEMENT: RELOAD W NEW DATA | REINIT INCLUDE PREV WINDOW | REINIT KEEP BUFFER FROM PREVIOUS WINDOW
        # GOAL: FOR EACH WINDOW, NEEDS TO USE NEW DATA
        if initial: # i - 63*2 - rebalance_window - validation_window < 0
            start_date = 20090101
            start_id = 0
            #preproc = True
        else:
            #start_date += 63 #unique_trade_date[i - 63 - rebalance_window - validation_window]
            start_id += 63
            start_date = unique_train_date[start_id]
            #preproc = False
        
        train = data_split(df, start = start_date, end = unique_trade_date[i - rebalance_window - validation_window]) #])
            
        '''
        ################################# OLD #################################
        # train = data_split(df, start = 20090101, end = unique_trade_date[i - rebalance_window])
        '''
        
        env_train = DummyVecEnv([lambda: StockEnvTrain(train)])
        
        ## validation env
        # @main: validation = data_split(df, start=20090000, end=unique_trade_date[i - rebalance_window])
        # @.ipynb:
        validation = data_split(df, start=unique_trade_date[i - rebalance_window - validation_window],
                                end=unique_trade_date[i - rebalance_window])
        env_val = DummyVecEnv([lambda: StockEnvValidation(validation,
                                                          turbulence_threshold=turbulence_threshold,
                                                          iteration=i)])
        obs_val = env_val.reset()

        ############## Training and Validation starts ##############
        print(f"======{name} Training========")
        #print('initial:', initial)
        #raise ValueError()
        
        if PREPROC_GLOB == 'Tb':
        
            model = train_TQC_CPT(env_train, model_name = i, #"TQC_Prelec_org_30k_dow_{}".format(i), 
                                  preproc_var = initial, 
                                  timesteps=timesteps, seed=seed, n_quantiles=n_quantiles, n_critics=n_critics, 
                                  learning_starts=learning_starts, a=a, actor_mode=actor_mode, critic_mode=critic_mode, k=k,
                                  ent_coef = ent_coef)
        
        elif PREPROC_GLOB == 'Ta':
            model = train_TQC_CPT(env_train, model_name = i, #"TQC_Prelec_org_30k_dow_{}".format(i), 
                                  timesteps=timesteps, seed=seed, n_quantiles=n_quantiles, n_critics=n_critics, 
                                  learning_starts=learning_starts, a=a, actor_mode=actor_mode, critic_mode=critic_mode, k=k,
                                  ent_coef = ent_coef)
        else:
            print('PREPROC_GLOB received invalid specs')
            
            raise NotImplementedError()
        
        print(f"======{name} Validation from: ", unique_trade_date[i - rebalance_window - validation_window],
              "to ", unique_trade_date[i - rebalance_window])

        ## validation
        DRL_validation(model=model, test_data=validation, test_env=env_val, test_obs=obs_val)
        sharpe = get_validation_sharpe(i)
        print(f"{name} Validation Sharpe Ratio: ", sharpe)

        sharpe_list.append(sharpe)

        model_use.append(name)

        ############## Trading starts ##############
        print("======Trading from: ", unique_trade_date[i - rebalance_window], "to ", unique_trade_date[i])
        
        last_state = DRL_prediction(df=df, model=model, name=name,
                                    last_state=last_state, iter_num=i,
                                    unique_trade_date=unique_trade_date,
                                    rebalance_window=rebalance_window,
                                    turbulence_threshold=turbulence_threshold,
                                    initial=initial)

        print("============Trading Done============")
        ############## Trading ends ##############

    end = time.time()
    print(f"{name} Strategy took: ", (end - start) / 60, " minutes")


def run_once(seed, actor_mode, turb_var, k, ent_coef):
    
    global PREPROC_GLOB

    preprocessed_path = "done_data.csv"
    data = pd.read_csv(preprocessed_path, index_col=0)
    unique_trade_date = data[(data.datadate > 20151001)&(data.datadate <= 20200707)].datadate.unique()
    rebalance_window = 63
    validation_window = 63
    
    learning_starts = 256 #256 #200
    n_quantiles = 25
    n_critics = 3 # 1, 5, 3
    #ent_coef = "auto" # (desktop) "auto" # 1.#"auto"
    #k = 1 #0, 2, (desktop) 1
    timesteps = 5000 #2000 #5000 #1000, self.terminal of envTrain hit when self.day >= 1761
    
    #actor_mode = 'cpt' #'cpt' # None (TQC-ori), #'cpt' (CANNOT ATTACH 88/95! better stats?), #'maxent' 
    PREPROC_GLOB = 'Ta' # 'Tb'
    #turb_var = .9 # NEXT!!!!!!!!!!!!!!!!
    
    if ent_coef == "auto":
        entID = 0
    elif ent_coef == 1.:
        entID = int(ent_coef)
    else:
        entID = ent_coef
        
    name = f'tqcA{actor_mode}E{entID}C{n_critics}K{k}T{timesteps//1000}S{seed}'
    
    if PREPROC_GLOB == 'Tb':
        name += f'P{PREPROC_GLOB}'
        
    if turb_var == .9:
        turbID = int(100 * turb_var)
        name += f'Tu{turbID}'
        
    run_tqc_cpt_strategy(data, unique_trade_date, rebalance_window, validation_window, 
                         timesteps=timesteps, seed=seed, learning_starts=learning_starts,
                         n_quantiles=n_quantiles, n_critics=n_critics, k = k,
                         name= name, actor_mode=actor_mode, critic_mode=None, 
                         ent_coef = ent_coef, turb_var=turb_var)

if __name__ == "__main__":
    for turb_var in [.9]: # (prev) [.9, .95]:
        for seed in [69152, 10672, 40931,  6280]:
            # I: [99172, 71108, 18660, 1310, 1701, 29171, 13102, 79665, 61282, 58992]:
            # III: [69152, 10672, 40931,  6280, 17913, 51104, 43136, 35269, 85721, 75927]
            # IV: [28994,  1266, 58514, 79698, 90945, 88887, 79147, 95650, 40488, 34506]
            for actor_mode in ['cpt']: # (desktop: cpt88-619)
                for k, ent_coef in [(0, 1.), (1, "auto")]:
                    #[(0, .5), (1, .5)]: # (prev) [(0, 1.), (1, "auto")]:
                    PREPROC_GLOB = None
                    run_once(seed, actor_mode, turb_var, k, ent_coef)