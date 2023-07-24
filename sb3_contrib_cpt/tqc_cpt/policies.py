import warnings
from typing import Any, Dict, List, Optional, Tuple, Type, Union
from math import ceil
import gym
import torch as th

from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution, StateDependentNoiseDistribution
#from sb3_contrib_cpt.tqc_cpt.distributions import SquashedDiagGaussianDistribution, StateDependentNoiseDistribution
from stable_baselines3.common.policies import BaseModel, BasePolicy
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    NatureCNN,
    create_mlp,
    get_actor_critic_arch,
)
from stable_baselines3.common.type_aliases import Schedule
from torch import nn as nn
import numpy as np

# CAP the standard deviation of the actor
LOG_STD_MAX = 2
LOG_STD_MIN = -20

################################## CPT PARAMS #################################
'''alpha_ = 0.95
lambda_ = 1.5
rho1, rho2 = 0.5, 0.5
a_ = 0.3
b_ = 0

def compute_cdf(k, q, pos):
    # compute cdf of k-th q quantile: 1 <= k <= q;
    
    if pos == True:
        return 1 - k/q
    else:
        return k/q
    
def prelec_weighting(taus, a=a_):
    """
    :param quantiles: 1D tensor of sorted quantiles
    :param a: the parameter a of the prelec weighting function
    :return: the prelec-weighted quantiles
    """

    weights = np.exp(-(-np.log(taus))**a)
    # weights = th.exp(-(-th.log(taus)))/taus
    return weights

def cpt_weighting(F, pos, rho1 = rho1, rho2 = rho2): 
    # @Prash: rho1 = .61, rho2 = .69 \in[0.3, 1]; assert w(F) monotonic in F;
    # @Barberis: rho1 = rho2 = .5 
    if pos == True:
        return F**rho1 / ((F**rho1 + (1-F)**rho1)**(1/rho1))
        # return (rho1 * (F**(rho1 - 1)) * ((F**rho1 + (1-F)**rho1)**(1/rho1)) - F**rho1 * (rho1 * (F**(rho1 - 1)) + (1-F)**(rho1 - 1) * -1)) / ((F**rho1 + (1-F)**rho1)**(1/rho1))**2

    else:
        return F**rho2 / ((F**rho2 + (1-F)**rho2)**(1/rho2))
        # return (rho2 * F**(rho2-1) * (F**rho2 + (1-F)**rho2)**(1/rho2) - F**rho2 * (1-F)**(rho2-1) * (F**rho2 + (1-F)**rho2)**(1/rho2)**(-rho2)) / (F**rho2 + (1-F)**rho2)**(1/rho2)**2
'''

################ PREPROCESS OBS (.multiplier, .addition) ######################
'''
AVG_FEATURE = th.tensor([[433643,  20,     25,     44,     40,     19,     69,     52,     23,
                          136,     24,    107,     16,     57,     34,     24,     57,     62,
                          29,      22,     13,     15,     53,     53,     43,     26,     16,
                          30,      30,     50,     70,    147,    176,   1336,    218,    153,
                            427,   1076,    351,    267,    110,    375,    477,    867,    183,
                            536,    253,    416,    283,    230,    266,    346,    492,    366,
                            560,    748,    819,    371,    347,   1046,    344,      0,      0,
                              0,      0,      0,      0,      0,      0,      2,      0,      1,
                              0,      0,      0,      0,      0,      0,      0,      0,      0,
                              0,      0,      0,      0,      0,      0,      0,      0,      0,
                              0,     60,     53,     50,     46,     54,     49,     50,     50,
                             55,     51,     55,     52,     48,     49,     52,     46,     52,
                             48,     55,     50,     46,     45,     49,     48,     48,     55,
                             46,     56,     45,     45,     77,     37,     20,     21,     35,
                              4,     25,     32,     50,      2,     51,     37,     26,     18,
                             32,      0,     48,     30,     56,     24,     19,     18,     32,
                              1,     24,     57,    -12,     26,      4,    -13,     27,     32,
                             28,     32,     20,     23,     30,     29,     24,     21,     21,
                             24,     27,     25,     27,     22,     25,     20,     28,     24,
                             27,     28,     28,     26,     27,     27,     24,     26,     21,
                             23]]).to(th.float64)
STDEV_FEATURE = 1/ (1 + th.tensor([[277424,      5,      9,      6,     11,      3,      4,     20,      4,
                                 35,      2,     13,      2,      4,      8,      2,      2,     11,
                                  3,      4,      1,      1,      4,      8,      4,      3,      2,
                                  1,      4,      2,      3,    120,    213,    666,    141,    114,
                                213,    601,    211,    219,    104,    451,    455,    468,    131,
                                414,    222,    253,    297,    226,    230,    329,    264,    280,
                                550,    257,    570,    277,    314,    408,    362,      0,      0,
                                  1,      1,      0,      1,      2,      0,      3,      0,      1,
                                  0,      0,      0,      0,      0,      1,      0,      0,      0,
                                  0,      0,      1,      0,      0,      0,      0,      0,      0,
                                  0,      8,     10,     11,     17,      8,     10,     14,     11,
                                 10,      9,      8,      9,     14,     11,     11,     13,     12,
                                 14,     10,     10,     12,     16,     13,     10,     11,      8,
                                 11,      9,     13,      9,     92,    109,    115,    108,     92,
                                 95,    112,    104,    104,    111,     97,     97,    101,    107,
                                107,    107,    110,     93,     95,    105,     97,    109,    103,
                                105,    118,     91,    102,    106,     93,     95,     18,     19,
                                 18,     26,     15,     21,     19,     17,     16,     15,     15,
                                 18,     25,     22,     21,     22,     17,     18,     17,     17,
                                 19,     20,     19,     21,     20,     17,     21,     18,     20,
                                 19]]).to(th.float64))
'''
STDEV_FEATURE = th.ones(181)
AVG_FEATURE = th.zeros(181)
K_MAX = 1.

#print('AVG_FEATURE:', AVG_FEATURE)
#print('STDEV_FEAT:', STDEV_FEATURE)


class Actor(BasePolicy):
    """
    Actor network (policy) for TQC.

    :param observation_space: Obervation space
    :param action_space: Action space
    :param net_arch: Network architecture
    :param features_extractor: Network to extract features
        (a CNN when using images, a nn.Flatten() layer otherwise)
    :param features_dim: Number of features
    :param activation_fn: Activation function
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE.
    :param sde_net_arch: Network architecture for extracting features
        when using gSDE. If None, the latent features from the policy will be used.
        Pass an empty list to use the states as features.
    :param use_expln: Use ``expln()`` function instead of ``exp()`` when using gSDE to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param clip_mean: Clip the mean output when using gSDE to avoid numerical instability.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        net_arch: List[int],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        full_std: bool = True,
        sde_net_arch: Optional[List[int]] = None,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        normalize_images: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
            squash_output=True,
        )

        # Save arguments to re-create object at loading
        self.use_sde = use_sde
        self.sde_features_extractor = None
        self.net_arch = net_arch
        self.features_dim = features_dim
        self.activation_fn = activation_fn
        self.log_std_init = log_std_init
        self.sde_net_arch = sde_net_arch
        self.use_expln = use_expln
        self.full_std = full_std
        self.clip_mean = clip_mean
        
        action_dim = get_action_dim(self.action_space)
        
        self.multiplier = STDEV_FEATURE #th.tensor([[10**-4] + [1]*(self.features_dim - 1)])
        self.addition = AVG_FEATURE #th.tensor([[.5*10**4] + [0]*(self.features_dim - 1)])
        
        '''
        self.batchNormLayer = nn.BatchNorm1d(features_dim, affine=False)
        
        ### batchNorm predict has (batch_size = 1..) ###
        print('features:', features_.shape)
        features = self.batchNormLayer(features_)
        print('features:', features.shape)
        
        raise ValueError
        #################
        '''
        
        latent_pi_net = create_mlp(features_dim, -1, net_arch, activation_fn)
        self.latent_pi = nn.Sequential(*latent_pi_net)
        
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else features_dim

        if sde_net_arch is not None:
            warnings.warn("sde_net_arch is deprecated and will be removed in SB3 v2.4.0.", DeprecationWarning)

        if self.use_sde:
            self.action_dist = StateDependentNoiseDistribution(
                action_dim, full_std=full_std, use_expln=use_expln, learn_features=True, squash_output=True
            )
            self.mu, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=last_layer_dim, latent_sde_dim=last_layer_dim, log_std_init=log_std_init
            )
            # Avoid numerical issues by limiting the mean of the Gaussian
            # to be in [-clip_mean, clip_mean]
            if clip_mean > 0.0:
                self.mu = nn.Sequential(self.mu, nn.Hardtanh(min_val=-clip_mean, max_val=clip_mean))
        else:
            '''print('@policies-sb3contrib l146')
            
            print('actiondim:', action_dim)
            print('latent_pi_net:', latent_pi_net)
            print('net_arch:', net_arch)
            print('last_layer_dim:', last_layer_dim) # OUT_layer: mu, std; before this last layer: latent_pi (sequential, ReLu)
            '''
            self.action_dist = SquashedDiagGaussianDistribution(action_dim)
            self.mu = nn.Linear(last_layer_dim, action_dim)
            self.log_std = nn.Linear(last_layer_dim, action_dim)

    def _update_preprocess(self, avg_feat: th.Tensor, std_feat: th.Tensor) -> th.Tensor:
        
        self.addition = avg_feat
        self.multiplier = std_feat
        
    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                features_dim=self.features_dim,
                activation_fn=self.activation_fn,
                use_sde=self.use_sde,
                log_std_init=self.log_std_init,
                full_std=self.full_std,
                use_expln=self.use_expln,
                features_extractor=self.features_extractor,
                clip_mean=self.clip_mean,
            )
        )
        return data

    def get_std(self) -> th.Tensor:
        """
        Retrieve the standard deviation of the action distribution.
        Only useful when using gSDE.
        It corresponds to ``th.exp(log_std)`` in the normal case,
        but is slightly different when using ``expln`` function
        (cf StateDependentNoiseDistribution doc).

        :return:
        """
        msg = "get_std() is only available when using gSDE"
        assert isinstance(self.action_dist, StateDependentNoiseDistribution), msg
        return self.action_dist.get_std(self.log_std)

    def reset_noise(self, batch_size: int = 1) -> None:
        """
        Sample new weights for the exploration matrix, when using gSDE.

        :param batch_size:
        """
        msg = "reset_noise() is only available when using gSDE"
        assert isinstance(self.action_dist, StateDependentNoiseDistribution), msg
        self.action_dist.sample_weights(self.log_std, batch_size=batch_size)

    def get_action_dist_params(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor, Dict[str, th.Tensor]]:
        """
        Get the parameters for the action distribution.

        :param obs:
        :return:
            Mean, standard deviation and optional keyword arguments.
        """
        
        obs = (obs - self.addition) * self.multiplier
        features = self.extract_features(obs)#, self.features_extractor_)
        latent_pi = self.latent_pi(features) # start NN
        mean_actions = self.mu(latent_pi)
        
        if self.use_sde:
            return mean_actions, self.log_std, dict(latent_sde=latent_pi)
        # Unstructured exploration (Original implementation)
        log_std = self.log_std(latent_pi)
        # Original Implementation to cap the standard deviation
        log_std = th.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        
        return mean_actions, log_std, {}

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> th.Tensor:
        mean_actions, log_std, kwargs = self.get_action_dist_params(obs)
        #print('@tqc-pol Actor forward')
        #print('mean_acts:', mean_actions.tolist()[0])
        #print('log_std:', log_std.tolist()[0])
        
        return self.action_dist.actions_from_params(mean_actions, log_std, deterministic=deterministic, **kwargs)

    def action_log_prob(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        
        #print('obs @tqc-pol action_log_prob', obs.shape)
        
        mean_actions, log_std, kwargs = self.get_action_dist_params(obs)
        # return action and associated log prob
        
        if K_MAX > 1.:
            actions_pi, log_prob = self.action_dist.log_prob_from_params(mean_actions, log_std, **kwargs)    
            return K_MAX * actions_pi, log_prob
        
        return self.action_dist.log_prob_from_params(mean_actions, log_std, **kwargs)

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        return self(observation, deterministic)


class Critic(BaseModel):
    """
    Critic network (q-value function) for TQC.

    :param observation_space: Obervation space
    :param action_space: Action space
    :param net_arch: Network architecture
    :param features_extractor: Network to extract features
        (a CNN when using images, a nn.Flatten() layer otherwise)
    :param features_dim: Number of features
    :param activation_fn: Activation function
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param share_features_extractor: Whether the features extractor is shared or not
        between the actor and the critic (this saves computation time)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        net_arch: List[int],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        n_quantiles: int = 25,
        n_critics: int = 2,
        share_features_extractor: bool = False,
        critic_mode = None
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        action_dim = get_action_dim(self.action_space)
        self.multiplier = STDEV_FEATURE #th.tensor([[10**-4] + [1]*(features_dim - 1)])
        self.addition = AVG_FEATURE #th.tensor([[.5*10**4] + [0]*(features_dim - 1)])
        
        self.share_features_extractor = share_features_extractor
        self.q_networks = []
        self.n_quantiles = n_quantiles
        self.n_critics = n_critics
        self.quantiles_total = n_quantiles * n_critics
        self.critic_mode = critic_mode
        
        for i in range(n_critics):
            qf_net = create_mlp(features_dim + action_dim, n_quantiles, net_arch, activation_fn)
            qf_net = nn.Sequential(*qf_net)
            self.add_module(f"qf{i}", qf_net)
            self.q_networks.append(qf_net)
    
    def _update_preprocess(self, avg_feat: th.Tensor, std_feat: th.Tensor) -> th.Tensor:
        
        self.addition = avg_feat
        self.multiplier = std_feat
        
    def forward(self, obs: th.Tensor, action: th.Tensor, cpt=True, prelec=False) -> List[th.Tensor]:
        # Learn the features extractor using the policy loss only
        # when the features_extractor is shared with the actor
        
        obs = (obs - self.addition) * self.multiplier
        action = action / K_MAX
        
        with th.set_grad_enabled(not self.share_features_extractor):
            features = self.extract_features(obs)#, self.features_extractor_)
            
        '''print('@tqc-policies, Critic, features:', self.share_features_extractor)
        print('obs.shape:', obs.shape, obs[0])
        print('feat.shape:', features.shape, features[0])
        raise ValueError()
        '''
        qvalue_input = th.cat([features, action], dim=1)
        # print(qvalue_input)
        # print('@tqc-pol:',qvalue_input[2].tolist())
        quantiles = th.stack(tuple(qf(qvalue_input) for qf in self.q_networks), dim=1)
        
        if not self.critic_mode:
            return quantiles
        else:
            raise NotImplementedError()
        
        '''
        # prelec weighting
        original_shape = quantiles.shape
        flattened = quantiles.view(-1)
        current_quantiles, indices  = th.sort(flattened)
        _, reverse_indices = th.sort(indices)
        q = flattened.shape[0]

        if self.critic_mode == 'prelec':    
            k = ceil(q/self.quantiles_total)
            midpoint = 1/self.quantiles_total
            midpoint /= 2
            taus = th.linspace(start = midpoint, end = 1-midpoint, steps = self.quantiles_total)
            taus = taus.repeat_interleave(k)
            weights = prelec_weighting(taus)
        else:
            XList = flattened.detach().numpy()
            m = len(XList)
            pos_mask = XList >= b_
            neg_mask = XList < b_
            vectorised_prob_func = np.vectorize(cpt_weighting, excluded=['pos'])

            k = ceil(q/self.quantiles_total)
            midpoint = 1/self.quantiles_total
            midpoint /= 2
            taus = th.linspace(start = midpoint, end = 1-midpoint, steps = self.quantiles_total)
            taus = taus.repeat_interleave(k)
            taus = taus.detach().numpy()

            # Directly comptute cdf from taus
            pos_proba = vectorised_prob_func(1-taus, True)
            neg_proba = vectorised_prob_func(taus, False)
            taus[pos_mask] = pos_proba[pos_mask]
            taus[neg_mask] = neg_proba[neg_mask]
            weights = th.tensor(taus)

            # i = np.arange(1, m+1)
            # pos_proba = vectorised_prob_func(1-i/m, True)
            # neg_proba = vectorised_prob_func(i/m, False)
            # i[pos_mask] = pos_proba[pos_mask]
            # i[neg_mask] = neg_proba[neg_mask]
            # weights = th.tensor(i)

        weighted_quantiles = current_quantiles * weights
        reshaped = weighted_quantiles.gather(dim=-1, index=reverse_indices)
        final_quantile = reshaped.view(original_shape)
        return final_quantile
        '''


class TQCPolicy(BasePolicy):
    """
    Policy class (with both actor and critic) for TQC.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param sde_net_arch: Network architecture for extracting features
        when using gSDE. If None, the latent features from the policy will be used.
        Pass an empty list to use the states as features.
    :param use_expln: Use ``expln()`` function instead of ``exp()`` when using gSDE to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param clip_mean: Clip the mean output when using gSDE to avoid numerical instability.
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the feature extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param n_quantiles: Number of quantiles for the critic.
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether to share or not the features extractor
        between the actor and the critic (this saves computation time)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        sde_net_arch: Optional[List[int]] = None,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_quantiles: int = 25,
        n_critics: int = 2,
        share_features_extractor: bool = False,
        critic_mode = None
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=True,
        )

        if net_arch is None:
            net_arch = [256, 256]

        actor_arch, critic_arch = get_actor_critic_arch(net_arch)

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.net_args = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "net_arch": actor_arch,
            "activation_fn": self.activation_fn,
            "normalize_images": normalize_images,
        }
        self.actor_kwargs = self.net_args.copy()

        if sde_net_arch is not None:
            warnings.warn("sde_net_arch is deprecated and will be removed in SB3 v2.4.0.", DeprecationWarning)

        sde_kwargs = {
            "use_sde": use_sde,
            "log_std_init": log_std_init,
            "use_expln": use_expln,
            "clip_mean": clip_mean,
        }
        self.actor_kwargs.update(sde_kwargs)
        self.critic_kwargs = self.net_args.copy()
        tqc_kwargs = {
            "n_quantiles": n_quantiles,
            "n_critics": n_critics,
            "net_arch": critic_arch,
            "share_features_extractor": share_features_extractor,
            'critic_mode': critic_mode
        }
        self.critic_kwargs.update(tqc_kwargs)
        self.actor, self.actor_target = None, None
        self.critic, self.critic_target = None, None
        self.share_features_extractor = share_features_extractor

        self._build(lr_schedule)

    def _build(self, lr_schedule: Schedule) -> None:
        self.actor = self.make_actor()
        self.actor.optimizer = self.optimizer_class(self.actor.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

        if self.share_features_extractor:
            self.critic = self.make_critic(features_extractor=self.actor.features_extractor)
            # Do not optimize the shared features extractor with the critic loss
            # otherwise, there are gradient computation issues
            critic_parameters = [param for name, param in self.critic.named_parameters() if "features_extractor" not in name]
        else:
            # Create a separate features extractor for the critic
            # this requires more memory and computation
            self.critic = self.make_critic(features_extractor=None)
            critic_parameters = self.critic.parameters()

        # Critic target should not share the feature extactor with critic
        self.critic_target = self.make_critic(features_extractor=None)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Target networks should always be in eval mode
        self.critic_target.set_training_mode(False)

        self.critic.optimizer = self.optimizer_class(critic_parameters, lr=lr_schedule(1), **self.optimizer_kwargs)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                activation_fn=self.net_args["activation_fn"],
                use_sde=self.actor_kwargs["use_sde"],
                log_std_init=self.actor_kwargs["log_std_init"],
                use_expln=self.actor_kwargs["use_expln"],
                clip_mean=self.actor_kwargs["clip_mean"],
                lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
                n_quantiles=self.critic_kwargs["n_quantiles"],
                n_critics=self.critic_kwargs["n_critics"],
            )
        )
        return data

    def reset_noise(self, batch_size: int = 1) -> None:
        """
        Sample new weights for the exploration matrix, when using gSDE.

        :param batch_size:
        """
        self.actor.reset_noise(batch_size=batch_size)

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> Actor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        
        #print('@tqc-cpt policy l580, actor_kwargs pre Actor init:', self.actor_kwargs)
        
        return Actor(**actor_kwargs).to(self.device)

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> Critic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return Critic(**critic_kwargs).to(self.device)

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> th.Tensor:
        return self._predict(obs, deterministic=deterministic)

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        #print('@tqc-pol, TQCPolicy')
        #print('self.actor:', self.actor)
        #print('last_obs, inside:', observation.tolist()[0])
        #print('vs OUTSIDE???')
        return self.actor(observation, deterministic)

    def set_training_mode(self, mode: bool) -> None:
        """
        Put the policy in either training or evaluation mode.
        This affects certain modules, such as batch normalisation and dropout.
        :param mode: if true, set to training mode, else set to evaluation mode
        """
        self.actor.set_training_mode(mode)
        self.critic.set_training_mode(mode)
        self.training = mode


MlpPolicy = TQCPolicy


class CnnPolicy(TQCPolicy):
    """
    Policy class (with both actor and critic) for TQC.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param sde_net_arch: Network architecture for extracting features
        when using gSDE. If None, the latent features from the policy will be used.
        Pass an empty list to use the states as features.
    :param use_expln: Use ``expln()`` function instead of ``exp()`` when using gSDE to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param clip_mean: Clip the mean output when using gSDE to avoid numerical instability.
    :param features_extractor_class: Features extractor to use.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param n_quantiles: Number of quantiles for the critic.
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether to share or not the features extractor
        between the actor and the critic (this saves computation time)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        sde_net_arch: Optional[List[int]] = None,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        features_extractor_class: Type[BaseFeaturesExtractor] = NatureCNN,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_quantiles: int = 25,
        n_critics: int = 2,
        share_features_extractor: bool = False,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            use_sde,
            log_std_init,
            sde_net_arch,
            use_expln,
            clip_mean,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            n_quantiles,
            n_critics,
            share_features_extractor,
        )


class MultiInputPolicy(TQCPolicy):
    """
    Policy class (with both actor and critic) for TQC.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param sde_net_arch: Network architecture for extracting features
        when using gSDE. If None, the latent features from the policy will be used.
        Pass an empty list to use the states as features.
    :param use_expln: Use ``expln()`` function instead of ``exp()`` when using gSDE to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param clip_mean: Clip the mean output when using gSDE to avoid numerical instability.
    :param features_extractor_class: Features extractor to use.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param n_quantiles: Number of quantiles for the critic.
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether to share or not the features extractor
        between the actor and the critic (this saves computation time)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        sde_net_arch: Optional[List[int]] = None,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        features_extractor_class: Type[BaseFeaturesExtractor] = CombinedExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_quantiles: int = 25,
        n_critics: int = 2,
        share_features_extractor: bool = False,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            use_sde,
            log_std_init,
            sde_net_arch,
            use_expln,
            clip_mean,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            n_quantiles,
            n_critics,
            share_features_extractor,
        )