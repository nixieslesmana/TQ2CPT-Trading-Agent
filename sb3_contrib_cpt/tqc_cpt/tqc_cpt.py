# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 13:41:14 2023

@author: Nixie S Lesmana
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union

import gym
import numpy as np
from math import ceil
import matplotlib.pyplot as plt

import torch as th
import csv
from datetime import datetime

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from sb3_contrib_cpt.tqc_cpt.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback
from stable_baselines3.common.utils import get_parameters_by_name, polyak_update

from sb3_contrib.common.utils import quantile_huber_loss
from sb3_contrib_cpt.tqc_cpt.policies import CnnPolicy, MlpPolicy, MultiInputPolicy, TQCPolicy


## cpt
alpha_ = 0.95
lambda_ = 1.5
rho1, rho2 = 0.5, 0.5
a_ = 0.3
b_ = 0
'''

## cpt88
alpha_ = 0.88
lambda_ = 2.25
rho1, rho2 = .65, .65 #.61, .69 #0.65, 0.65 # .61 (pos), .69 (neg)
a_ = 0.3
b_ = 0
'''
print('CPT PARAMS:', alpha_, rho1, rho2, lambda_, b_)

SelfTQC_CPT = TypeVar("SelfTQC_CPT", bound="TQC_CPT")

'''
def lbub_monotonic(quantiles):
    # expect list of len K
    out_ub = [0]*len(quantiles)
    out_ub[0] = quantiles[0]
    out_lb = [0]*len(quantiles)
    out_lb[0] = quantiles[0]
    for k in range(1, len(quantiles)):
        prev = out_ub[k-1]
        cur = quantiles[k]
        if cur < prev:
            out_ub[k] = prev #(cur + prev) / 2
            #out[k-1] = (cur + prev) / 2
            out_lb[k-1] = cur
        else:
            out_ub[k] = cur
            out_lb[k] = cur
    # if quantiles alr monotonone, auto return quantiles
    return (np.array(out_lb) + np.array(out_ub))/2
'''

def prelec_weighting(taus, a=a_):
    """
    :param quantiles: 1D tensor of sorted quantiles
    :param a: the parameter a of the prelec weighting function
    :return: the prelec-weighted quantiles
    """

    weights = np.exp(-(-np.log(taus))**a)
    # weights = np.exp(-(-np.log(taus)))/taus
    return weights

def compute_prelec(tensor, total_quantiles, sort=True, B=b_):
    
    if sort:
        tensor, _ = th.sort(tensor)

    XList = tensor.detach().numpy()
    #q = len(XList)
    #XList = lbub_monotonic(XList)

    vectorised_utility = np.vectorize(utility, excluded = ['pos', 'alpha' , 'lmbd', 'B'])

    CPT_val_pos = 0
    CPT_val_neg = 0
    pos_mask = XList >= B
    neg_mask = XList < B

    try:
        CPT_val_pos = np.nansum(vectorised_utility(XList[np.array(pos_mask)], True, alpha_, lambda_, B)/total_quantiles)
    except:
        CPT_val_pos = 0
    try:
        CPT_val_neg = np.nansum(vectorised_utility(XList[np.array(neg_mask)], False, alpha_, lambda_, B)/total_quantiles)
    except:
        CPT_val_neg = 0
    CPT_val = th.tensor(CPT_val_pos + CPT_val_neg)

    return CPT_val

#################################### CPT compute ###############################

def cpt_weighting(F, pos, rho1 = rho1, rho2 = rho2): 
    
    if pos == True:
        return F**rho1 / ((F**rho1 + (1-F)**rho1)**(1/rho1))
        # return (rho1 * (F**(rho1 - 1)) * ((F**rho1 + (1-F)**rho1)**(1/rho1)) - F**rho1 * (rho1 * (F**(rho1 - 1)) + (1-F)**(rho1 - 1) * -1)) / ((F**rho1 + (1-F)**rho1)**(1/rho1))**2

    else:
        return F**rho2 / ((F**rho2 + (1-F)**rho2)**(1/rho2))
        # return (rho2 * F**(rho2-1) * (F**rho2 + (1-F)**rho2)**(1/rho2) - F**rho2 * (1-F)**(rho2-1) * (F**rho2 + (1-F)**rho2)**(1/rho2)**(-rho2)) / (F**rho2 + (1-F)**rho2)**(1/rho2)**2

def utility(x, pos, alpha = alpha_, lmbd = lambda_, B = b_):
    
    if pos == True:
        return (x-B)**alpha
    else:
        return -lmbd * (B-x)**alpha 

'''
import torch as th
alpha_ = 0.95
lambda_ = 1.5
rho1, rho2 = 0.5, 0.5
a_ = 0.3
b_ = 0
tensor = th.Tensor([[[-10897.9577,  -8713.0451,  -7000.1268,  -5662.4362,  -5793.1418,
         -4086.0113,  -4100.1552,  -4013.2058,  -2581.0837,  -1020.1089,
         -1735.7339,   -981.8352,  -1398.7791,   -736.9388,   -445.5684,
          -138.9197,     76.1080,    -12.7236,    732.0802,    658.5605,
          1772.3086,   4903.7620,   3116.6291,   2216.4681,   4099.7732]], 
        [[-10897,  -8713,  -7000,  -5662,  -5793,
                 -4086,  -4100,  -4013,  -2581.0837,  -1020.1089,
                 -1735.7339,   -981.8352,  -1398.7791,   -736.9388,   -445.5684,
                  -138.9197,     0,   0,    732.0802,    658.5605,
                  1772.3086,   4903.7620,   3116.6291,   2216.4681,   4099.7732]]])
sort = False
'''

def compute_CPT_vectorize(tensor, sort = True, B=b_):
   
    #print('inside compute_CPT, params:', alpha_, rho1, lambda_, B)
   
    if sort:
        tensor, _ = th.sort(tensor)

    ## CHECK CPT VALUE = 0
    ## Implement grad_fn backward
    # NON-MONOTONIC ISSUE EXISTS!!!!
    
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

########################## OLD-IMPLEMENT ##################################
def compute_CPT_old(tensor, sort = True, B=b_):
    
    #XList = tensor.detach().numpy()
    XList = tensor
    q = len(XList)
    
    '''
    XList = lbub_monotonic(XList)
    '''
    
    vectorised_utility = np.vectorize(utility, excluded = ['pos', 'alpha' , 'lmbd', 'B'])

    CPT_val_pos = 0
    CPT_val_neg = 0
    pos_mask = XList >= B
    neg_mask = XList < B
    

    '''
    try:
        CPT_val_pos = np.nansum(vectorised_utility(XList[np.array(pos_mask)], True, alpha_, lambda_, B)/q)
    except:
        CPT_val_pos = 0
    try:
        CPT_val_neg = np.nansum(vectorised_utility(XList[np.array(neg_mask)], False, alpha_, lambda_, B)/q)
    except:
        CPT_val_neg = 0
    '''

    try:
        dwF_pos = cpt_weighting(1 - np.linspace(0, 1, q+1)[:-1], True) - cpt_weighting(1 - np.linspace(0, 1, q+1)[1:], True) # to replace 1/q
        #u_pos = vectorised_utility(XList[np.array(pos_mask)], True, alpha_, lambda_, B)
        #CPT_val_pos = np.dot(u_pos, dwF_pos[np.array(pos_mask)])
        u_pos = vectorised_utility(XList[pos_mask], True, alpha_, lambda_, B)
        CPT_val_pos = np.dot(u_pos, dwF_pos[pos_mask])
    except:
        CPT_val_pos = 0
    
    try:
        dwF_neg = cpt_weighting(np.linspace(0, 1, q+1)[1:], False) - cpt_weighting(np.linspace(0, 1, q+1)[:-1], False)
        #u_neg = vectorised_utility(XList[np.array(neg_mask)], False, alpha_, lambda_, B)
        #CPT_val_neg = np.dot(u_neg, dwF_neg[np.array(neg_mask)])
        u_neg = vectorised_utility(XList[neg_mask], False, alpha_, lambda_, B)
        CPT_val_neg = np.dot(u_neg, dwF_neg[neg_mask])
    except:
        CPT_val_neg = 0
    
    CPT_val = th.tensor(CPT_val_pos + CPT_val_neg)
    
    return CPT_val

###############################################################################

def prelec_quantile_huber_loss(current_quantiles, target_quantiles, cum_prob=None, a=1.0, sum_over_quantiles=True, n_target_quantiles=25):
    """
    Prelec weighting function.
    :param current_quantiles: current estimate of quantiles, must be either (batch_size, n_quantiles) or (batch_size, n_critics, n_quantiles)
    :param target_quantiles: target of quantiles, must be either (batch_size, n_target_quantiles), (batch_size, 1, n_target_quantiles), or (batch_size, n_critics, n_target_quantiles)
    :param cum_prob: cumulative probabilities to calculate quantiles (also called midpoints in QR-DQN paper), must be either (batch_size, n_quantiles), (batch_size, 1, n_quantiles), or (batch_size, n_critics, n_quantiles). (if None, calculating unit quantiles)
    :param a: the parameter a of the prelec weighting function
    :param sum_over_quantiles: if summing over the quantile dimension or not
    :return: the loss
    """
    if cum_prob is None:
        n_quantiles = current_quantiles.shape[-1]
        # Cumulative probabilities to calculate quantiles.
        cum_prob = (th.arange(n_quantiles, device=current_quantiles.device, dtype=th.float) + 0.5) / n_quantiles

    current_quantiles, _ = th.sort(current_quantiles)
    target_quantiles, _ = th.sort(target_quantiles)
    current_quantiles = prelec_weighting(current_quantiles, a, n_target_quantiles)
    target_quantiles = prelec_weighting(target_quantiles, a, n_target_quantiles)

    pairwise_delta = target_quantiles - current_quantiles
    abs_pairwise_delta = th.abs(pairwise_delta)
    huber_loss = th.where(abs_pairwise_delta > 1, abs_pairwise_delta - 0.5, pairwise_delta**2 * 0.5)
    loss = th.abs(cum_prob - (pairwise_delta.detach() < 0).float()) * huber_loss
    if sum_over_quantiles:
        loss = loss.sum(dim=-2).mean()
    else:
        loss = loss.mean()

    return loss
    

class TQC_CPT(OffPolicyAlgorithm):
    """

    Controlling Overestimation Bias with Truncated Mixture of Continuous Distributional Quantile Critics.
    Paper: https://arxiv.org/abs/2005.04269
    This implementation uses SB3 SAC implementation as base.

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: learning rate for adam optimizer,
        the same learning rate will be used for all networks (Q-Values, Actor and Value function)
        it can be a function of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1)
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
        like ``(5, "step")`` or ``(2, "episode")``.
    :param gradient_steps: How many gradient update after each step
    :param action_noise: the action noise type (None by default), this can help
        for hard exploration problem. Cf common.noise for the different action noise type.
    :param replay_buffer_class: Replay buffer class to use (for instance ``HerReplayBuffer``).
        If ``None``, it will be automatically selected.
    :param replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation.
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    :param ent_coef: Entropy regularization coefficient. (Equivalent to
        inverse of reward scale in the original SAC paper.)  Controlling exploration/exploitation trade-off.
        Set it to 'auto' to learn it automatically (and 'auto_0.1' for using 0.1 as initial value)
    :param target_update_interval: update the target network every ``target_network_update_freq``
        gradient steps.
    :param target_entropy: target entropy when learning ``ent_coef`` (``ent_coef = 'auto'``)
    :param top_quantiles_to_drop_per_net: Number of quantiles to drop per network
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param use_sde_at_warmup: Whether to use gSDE instead of uniform sampling
        during the warm up phase (before learning starts)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    policy_aliases: Dict[str, Type[BasePolicy]] = {
        "MlpPolicy": MlpPolicy,
        "CnnPolicy": CnnPolicy,
        "MultiInputPolicy": MultiInputPolicy,
    }

    def __init__(
        self,
        policy: Union[str, Type[TQCPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Callable] = 3e-4,
        buffer_size: int = 1000000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: int = 1,
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[ReplayBuffer] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        ent_coef: Union[str, float] = "auto",
        target_update_interval: int = 1,
        target_entropy: Union[str, float] = "auto",
        top_quantiles_to_drop_per_net: int = 2,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        a=1.0, 
        actor_mode="prelec"
    ):

        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise=action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
            optimize_memory_usage=optimize_memory_usage,
            supported_action_spaces=(gym.spaces.Box),
            support_multi_env=True,
        )

        self.a = a
        self.target_entropy = target_entropy
        self.log_ent_coef = None  # type: Optional[th.Tensor]
        # Entropy coefficient / Entropy temperature
        # Inverse of the reward scale
        self.ent_coef = ent_coef
        self.target_update_interval = target_update_interval
        self.ent_coef_optimizer = None
        self.top_quantiles_to_drop_per_net = top_quantiles_to_drop_per_net
        self.actor_mode = actor_mode
        
        self.name = None

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()
        self._create_aliases()
        # Running mean and running var
        self.batch_norm_stats = get_parameters_by_name(self.critic, ["running_"])
        self.batch_norm_stats_target = get_parameters_by_name(self.critic_target, ["running_"])

        # Target entropy is used when learning the entropy coefficient
        if self.target_entropy == "auto":
            # automatically set target entropy if needed
            self.target_entropy = -np.prod(self.env.action_space.shape).astype(np.float32)
        else:
            # Force conversion
            # this will also throw an error for unexpected string
            self.target_entropy = float(self.target_entropy)

        # The entropy coefficient or entropy can be learned automatically
        # see Automating Entropy Adjustment for Maximum Entropy RL section
        # of https://arxiv.org/abs/1812.05905
        if isinstance(self.ent_coef, str) and self.ent_coef.startswith("auto"):
            # Default initial value of ent_coef when learned
            init_value = 1.0
            if "_" in self.ent_coef:
                init_value = float(self.ent_coef.split("_")[1])
                assert init_value > 0.0, "The initial value of ent_coef must be greater than 0"

            # Note: we optimize the log of the entropy coeff which is slightly different from the paper
            # as discussed in https://github.com/rail-berkeley/softlearning/issues/37
            self.log_ent_coef = th.log(th.ones(1, device=self.device) * init_value).requires_grad_(True)
            self.ent_coef_optimizer = th.optim.Adam([self.log_ent_coef], lr=self.lr_schedule(1))
        else:
            # Force conversion to float
            # this will throw an error if a malformed string (different from 'auto')
            # is passed
            self.ent_coef_tensor = th.tensor(float(self.ent_coef), device=self.device)

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor # self.policy: class TQCPolicy(...)
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target
        
        #print('@tqc-cpt, l425:', self.policy)

    def train(self, gradient_steps: int, batch_size: int = 64, logger_dict = None) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizers learning rate
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []
        actor_losses1, actor_losses2 = [], []
        crash_iters = []
        
        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            
            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()

            # Action by the current actor for the sampled state
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations) # K_MAX = 5.
            log_prob = log_prob.reshape(-1, 1)
            
            ############################# ENT COEF ############################
            ent_coef_loss = None
            if self.ent_coef_optimizer is not None:
                # Important: detach the variable from the graph
                # so we don't change it with other losses
                # see https://github.com/rail-berkeley/softlearning/issues/60
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
    
                ent_coef_losses.append(ent_coef_loss.item())
                
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())
            self.replay_buffer.ent_coef = ent_coef.item()
            
            ####################### CRASH ITERS ############################
            # CPT95, E, maxEnt suggests pi(.|obs) = pi(.|obs') after some learn steps (between ._n_updates 701 to 1401)
            mean_acts, std = self.actor.action_dist.log_prob_u(only_dist_params = True)
            
            #print('@tqc-cpt, self actor action dist:', self.actor.action_dist)
            #raise ValueError()
            
            acts_as_set = th.unique(mean_acts, dim = 0)
            #print('crash dyn, how fast to 1?', acts_as_set.shape[0])
            if acts_as_set.shape[0] <= 100: #(mean_acts[0] != mean_acts[-1]).flatten().sum() == 0:
                crash_iters += [(self._n_updates, acts_as_set.shape[0])]
            
            crash_print = False
            if len(logger_dict.keys()) > 0:
                if len(logger_dict['crash_iters']) % 700 == 0 and len(crash_iters) > 0:
                       crash_print = True
            elif len(crash_iters) > 0:
                crash_print = True
            
            if self._n_updates % 700 in [1] or crash_print == True or self._n_updates in []: #[1108, 1112, 1115, 1114, 1164, 1214, 1274, 1332, 1335, 1338]: 
                #len(crash_iters) % 700 == 1: 
                print('@main, n_updates:', self._n_updates)
                '''
                print('---')
                print('replay_data', self.replay_buffer.size())
                print('obs:', replay_data.observations[:, 0].tolist(), replay_data.observations[:, 0].mean())
                batch_inds = range(0, self.replay_buffer.size())
                #print(self.replay_buffer._get_samples(batch_inds, env=self._vec_normalize_env))
                all_data = self.replay_buffer._get_samples(batch_inds, env=self._vec_normalize_env)
                print(all_data.observations[:, 0].tolist())
                print(all_data.actions[-2:, :])
                #print(all_data.flatten().tolist())
                
                # SOMETHING IS WRONG DURING FIRST UPDATE -- IS IT DATA OR ARCHITECTURE?
                print('---')
                '''
                
                '''
                runID = datetime.now().strftime("%d%m%y%H%M%S")
                gaussActs, log_prob_u, presum_logProb, mean_acts, std = self.actor.action_dist.log_prob_u()
                
                #log_prob_a = log_prob_u - th.sum(th.log(1 - actions_pi**2 + self.actor.action_dist.epsilon), dim=1)
                
                to_append = []
                
                # header!
                to_add1 = ['logPdf_a'] + ['']
                to_add1 += ['logPdf_u'] + ['']
                to_add1 += ['logPdf_u per stock'] + ['' for _ in range(30)]
                to_add1 += ['dist(.|obs) mean'] + ['' for _ in range(30)]
                to_add1 += ['dist(.|obs) logstd'] + ['' for _ in range(30)]
                to_add1 += ['a ~ tanh(u)'] + ['' for _ in range(30)]
                to_add1 += ['u ~ dist(.|obs)'] + ['' for _ in range(30)]
                to_add1 += ['obs']
                to_append += [to_add1]
                
                for batchID in range(256):
                    
                    to_add = log_prob.tolist()[batchID] + [''] # = log_prob_a, negative when crashIter, #to_add += [log_prob_a[batchID].item()] + ['']
                    to_add += [log_prob_u[batchID].item()] + ['']
                    to_add += presum_logProb[batchID].tolist() + [''] # negative unbounded
                    
                    to_add += mean_acts[batchID].tolist() + ['']
                    to_add += std[batchID].tolist() + ['']
                    
                    to_add += actions_pi[batchID].tolist() + ['']
                    to_add += gaussActs[batchID].tolist() + ['']
                    
                    to_add += replay_data.observations[batchID].tolist()
                    
                    to_append += [to_add]
                 
                try:
                    if ent_coef_loss.item() > 6:  
                        filename = './output/entLossJump_s{}_n{}_r{}_.csv'.format(self.seed, self._n_updates, runID)
                    else:
                        filename = './output/entLossJump_s{}_n{}_r{}.csv'.format(self.seed, self._n_updates, runID)
                except:
                    filename = './output/entLossJump_s{}_n{}_r{}.csv'.format(self.seed, self._n_updates, runID)
                    
                f = open(filename, 'a', newline = '')
                writer = csv.writer(f)
                writer.writerows(to_append)
                f.close()
                '''
            ###################################################################
            
            # Optimize entropy coefficient, also called entropy temperature or alpha in the paper
            if ent_coef_loss is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()
            
            ############################# CRITIC ##############################

            with th.no_grad():
                # Select action according to policy
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                # Compute and cut quantiles at the next state
                # batch x nets x quantiles
                next_quantiles = self.critic_target(replay_data.next_observations, next_actions)
                
                '''
                #next_actions = next_actions / 10
                print('@tqc l580, nx_action:', next_actions[0]) 
                print('nx_action mean:', [next_actions[:, d].mean().item() for d in range(30)])#.shape)
                print('std:', [next_actions[:, d].std().item() for d in range(30)])#.shape)
                
                raise ValueError()
                '''
                
                '''
                print('buffer size cur:', self.replay_buffer.size())
                how_many_pos = (next_quantiles.view(batch_size, -1) > 0).sum(1) # from 75 elems
                print('batchID has pos qtile values:', sum(how_many_pos > 0).item(), 'out of ', batch_size,
                      ', total:', sum(how_many_pos).item())
                pos_idx = th.Tensor(range(0, batch_size))[how_many_pos > 0].to(th.int32)
                
                #all_data = self.replay_buffer._get_samples(np.array(range(0, self.replay_buffer.size())), 
                #                                           env=self._vec_normalize_env)
                print('rewards @buffer, minmax:', min(replay_data.rewards.flatten().tolist()), 
                      max(replay_data.rewards.flatten().tolist()), 'avg: ', np.average(replay_data.rewards.flatten().tolist()))
                print('sum dones=True:', sum(replay_data.dones.flatten().tolist()), 'out of ', batch_size) #self.replay_buffer.size())
                '''
                # Sort and drop top k quantiles to control overestimation.
                next_quantiles, _ = th.sort(next_quantiles.reshape(batch_size, -1))
                
                n_target_quantiles = self.critic.quantiles_total - self.top_quantiles_to_drop_per_net * self.critic.n_critics
                next_quantiles = next_quantiles[:, :n_target_quantiles]
                 
                # td error + entropy term: Distributional Soft Bellman operator
                target_quantiles_ = next_quantiles - ent_coef * next_log_prob.reshape(-1, 1)
                target_quantiles = replay_data.rewards + (1 - replay_data.dones) * self.gamma * target_quantiles_
                target_quantiles.unsqueeze_(dim=1)
                
            # Get current Quantile estimates using action from the replay buffer
            current_quantiles = self.critic(replay_data.observations, replay_data.actions)
                
            # Compute critic loss, not summing over the quantile dimension as in the paper.
            critic_loss = quantile_huber_loss(current_quantiles, target_quantiles, sum_over_quantiles=False)
            critic_losses.append(critic_loss.item())
            
            # Optimize the critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()
            
            ###################################################################
            ''' ### CRITIC LOSS INCREASE - Z PRED NEGATIVE STILL?
            # tried: (i) KMAX=10, act [explodes] (ii) preprocess_act (iii) KMAX=1***
            # seems tht ACTOR LOSS INCREASE DOMINATES CRITIC LOSS DECREASE
            # cld be cause CRITIC UPDATE NOT FINISHED when ACTOR UPDATES
            
            # (iii) critic increase less, still dominated by actor
            # TIME TO PRINT!!!!!
            
            ### MEAN ACTS INCREASE AGAIN... NO CRASH (DIFFERENT ACROSS OBS).. BUT LOG_STD CRASH..
            ### WHEN DID MEAN ACTS CROSS?
            if self._n_updates % 700 == 1 or crash_print == True or self._n_updates in []: 
                #[1108, 1112, 1115, 1114, 1164, 1214, 1274, 1332, 1335, 1338]: 
                # make .csv() print criticLoss, dim = (256, 3*25)
                
                to_append = []
                
                # header!
                to_add1 = ['obs']
                to_append += [to_add1]
                
                for batchID in range(256):
                    
                    to_add = log_prob.tolist()[batchID] + [''] # = log_prob_a, negative when crashIter, #to_add += [log_prob_a[batchID].item()] + ['']
                    to_add += [log_prob_u[batchID].item()] + ['']
                    to_add += presum_logProb[batchID].tolist() + [''] # negative unbounded
                    
                    to_add += mean_acts[batchID].tolist() + ['']
                    to_add += std[batchID].tolist() + ['']
                    
                    to_add += actions_pi[batchID].tolist() + ['']
                    to_add += gaussActs[batchID].tolist() + ['']
                    
                    to_add += replay_data.observations[batchID].tolist()
                    
                    to_append += [to_add]
            '''
            
            ############################## ACTOR ##############################
            # Compute actor loss
            actor_quantiles = self.critic(replay_data.observations, actions_pi).to(th.float64) # dim: (batch=256, N, M)
            if self.actor_mode == 'cpt': # temp method - apply mean() on N axis > cpt() on M axis, mean() on batch axis
                qf_pi = compute_CPT_vectorize(actor_quantiles.mean(1, keepdim=True), sort = False) #.view(-1, 1)
                
            elif self.actor_mode == 'prelec':
                qf_pi = compute_prelec(actor_quantiles.flatten(), total_quantiles= self.critic.quantiles_total)
            elif self.actor_mode == 'maxent':
                qf_pi = th.zeros([batch_size, 1]) #0
            else:
                qf_pi = actor_quantiles.mean(dim=2).mean(dim=1, keepdim=True)
            
            actor_loss = (ent_coef * log_prob - qf_pi).mean()
            
            '''
            print('===')
            # after critic update, MISMATCH! Numeric issues..
            print('CRITIC LOSS', critic_losses)
            print('qf_pi:', qf_pi.flatten().tolist()[2]) # , qf_pi.shape)
            print('actor_qtiles:', actor_quantiles[2].tolist()) #### CAUSING ERRORS!
            #print('obs:', replay_data.observations[2].tolist())
            #print('actions_pi:', actions_pi[2].tolist())#, actions_pi.shape)
            print('===')
            '''
            
            # CRITIC OBS PREPROCESS DOES NOT AFFECT OUTPUT?, BUT qf_pi w or w/o backward AFFECT!
            actor_losses.append(actor_loss.item())
            
            actor_loss1 = (ent_coef * log_prob).mean()
            actor_loss2 = qf_pi.mean()
            actor_losses1.append(actor_loss1.item())
            actor_losses2.append(actor_loss2.item())
            
            ######################## CHECK CRASH ITERS ########################
            if self._n_updates in []: #[1108, 1112, 1115, 1114, 1164, 1214, 1274, 1332, 1335, 1338]:
                
                to_write = ['n_updates:' + str(self._n_updates) + '\n']
                
                to_write += ['PRE-UPDATE actor_params']
                
                for p in self.actor.parameters():
                    to_write += ['p.shape: ' + str(p.shape)]
                    to_write += ['p:' + str(p.detach().numpy())]
                    to_write += ['\n']
                # + str([p.detach().numpy() for p in self.actor.parameters()]) + '\n']
                #print('p.shape:', [p.size() for p in self.actor.parameters()])
                # [torch.Size([256, 181]), torch.Size([256]), 
                # torch.Size([256, 256]), torch.Size([256]), 
                # torch.Size([30, 256]), torch.Size([30]), torch.Size([30, 256]), torch.Size([30])]
                
                '''################# IS OBS UNSCALED THE ISSUE ?? #################
                features = self.actor.extract_features(obs) # = obs by current definition (NO BATCHNORM?)
                latent_pi = self.actor.latent_pi(features)
                mean_actions = self.mu(latent_pi)
                log_std = self.log_std(latent_pi)
                
                # Original Implementation to cap the standard deviation
                log_std = th.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX) # LOG_STD_MAX = 2, LOG_STD_MIN = -20
                
                # check which line causes crash..
                th.Tensor([-20, 2]).exp()
                Out[43]: tensor([2.0612e-09, 7.3891e+00])
                # std moves from one clamp to another
                # log_std TOO LARGE OR TOO SMALL
                
                from stable_baselines.common.input import observation_input
                observation_input(ob_space, batch_size, scale = scale)
                
                sb3 only has image_scale: print('actor check scale image:', self.actor.normalize_images)
                '''
            
            ####################### Optimize the actor ########################
            self.actor.optimizer.zero_grad()
            #with th.autograd.detect_anomaly():
            actor_loss.backward()
            self.actor.optimizer.step()
            
            ######################## CHECK CRASH ITERS ########################
            to_write = []
            if self._n_updates in []: #[1108, 1112, 1115, 1114, 1164, 1214, 1274, 1332, 1335, 1338]:
                to_write += ['POST-UPDATE actor_params']
             
                for p in self.actor.parameters():
                    to_write += ['p.shape: ' + str(p.shape)]
                    to_write += ['p.grad:' + str(p.grad.tolist())] #numpy())]
                    to_write += ['p:' + str(p.detach().tolist())] #numpy())]
                    to_write += ['\n']
            
                to_write += ['===' + '\n']
                
                filename = './output/crashes_s{}_n{}.txt'.format(self.seed, self._n_updates)
                f = open(filename, 'a', newline = '')    
                
                for h in to_write:
                    f.write('{}\n'.format(h))
                f.close()
            
            ###################################################################
            
            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                # Copy running stats, see https://github.com/DLR-RM/stable-baselines3/issues/996
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

        #print('@tqc-cpt, ACTOR LOSS:', actor_losses, actor_losses1, actor_losses2) # qf_pi DIFFERENT!!!

        if logger_dict is not None:
            logger_dict['n_updates'] = self._n_updates
            
            try:
                logger_dict['ent_coefs'] += ent_coefs
                logger_dict['ent_coef_losses'] += ent_coef_losses
                logger_dict['actor_loss'] += actor_losses
                logger_dict['crash_iters'] += crash_iters
                logger_dict['actor_loss1'] += actor_losses1
                logger_dict['actor_loss2'] += actor_losses2
                logger_dict['critic_loss'] += critic_losses
            except:
                logger_dict['ent_coefs'] = ent_coefs
                logger_dict['ent_coef_losses'] = ent_coef_losses
                logger_dict['actor_loss'] = actor_losses
                logger_dict['crash_iters'] = crash_iters
                logger_dict['actor_loss1'] = actor_losses1
                logger_dict['actor_loss2'] = actor_losses2
                logger_dict['critic_loss'] = critic_losses
                    
            return logger_dict

    def get_quantile(self):
        """ For Debugging"""
        replay_data = self.replay_buffer.sample(self.batch_size, env=self._vec_normalize_env)
        current_quantiles = self.critic(replay_data.observations, replay_data.actions)
        return current_quantiles

    def get_critic(self):
        replay_data = self.replay_buffer.sample(self.batch_size, env=self._vec_normalize_env)
        actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations) 
        return self.critic(replay_data.observations, actions_pi)

    def learn(
        self: SelfTQC_CPT,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "TQC_CPT",
        reset_num_timesteps: bool = True,
        # progress_bar: bool = False,
    ) -> SelfTQC_CPT:

        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            # progress_bar=progress_bar,
        )

    def _excluded_save_params(self) -> List[str]:
        # Exclude aliases
        return super()._excluded_save_params() + ["actor", "critic", "critic_target"]

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "actor.optimizer", "critic.optimizer"]
        if self.ent_coef_optimizer is not None:
            saved_pytorch_variables = ["log_ent_coef"]
            state_dicts.append("ent_coef_optimizer")
        else:
            saved_pytorch_variables = ["ent_coef_tensor"]
        return state_dicts, saved_pytorch_variables
