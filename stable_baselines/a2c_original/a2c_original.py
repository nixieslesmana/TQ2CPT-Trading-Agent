import time
import csv
import gym
import numpy as np
import tensorflow as tf

from stable_baselines import logger
from stable_baselines.common import explained_variance, tf_util, ActorCriticRLModel, SetVerbosity, TensorboardWriter
from stable_baselines.common.policies import ActorCriticPolicy, RecurrentActorCriticPolicy
from stable_baselines.common.runners import AbstractEnvRunner
from stable_baselines.common.schedules import Scheduler
from stable_baselines.common.tf_util import mse, total_episode_reward_logger
from stable_baselines.common.math_util import safe_mean


def discount_with_dones(rewards, dones, gamma):    ## Same function with LIRPG/baselines/a2c/utils.py
    """
    Apply the discount value to the reward, where the environment is not done

    :param rewards: ([float]) The rewards
    :param dones: ([bool]) Whether an environment is done or not
    :param gamma: (float) The discount value
    :return: ([float]) The discounted rewards
    """
    discounted = []
    ret = 0  # Return: discounted reward
    for reward, done in zip(rewards[::-1], dones[::-1]):
        ret = reward + gamma * ret * (1. - done)  # fixed off by one bug
        discounted.append(ret)
    return discounted[::-1]


class A2C_Original(ActorCriticRLModel):
    """
    The A2C (Advantage Actor Critic) model class, https://arxiv.org/abs/1602.01783

    :param policy: (ActorCriticPolicy or str) The policy model to use (MlpPolicy, CnnPolicy, CnnLstmPolicy, ...)
    :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
    :param gamma: (float) Discount factor
    :param n_steps: (int) The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    :param vf_coef: (float) Value function coefficient for the loss calculation
    :param ent_coef: (float) Entropy coefficient for the loss calculation
    :param max_grad_norm: (float) The maximum value for the gradient clipping
    :param learning_rate: (float) The learning rate

    :param alpha: (float)  RMSProp decay parameter (default: 0.99)
    :param momentum: (float) RMSProp momentum parameter (default: 0.0)
    :param epsilon: (float) RMSProp epsilon (stabilizes square root computation in denominator of RMSProp update)
        (default: 1e-5)

    :param lr_schedule: (str) The type of scheduler for the learning rate update ('linear', 'constant',
                              'double_linear_con', 'middle_drop' or 'double_middle_drop')
    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    :param tensorboard_log: (str) the log location for tensorboard (if None, no logging)
    :param _init_setup_model: (bool) Whether or not to build the network at the creation of the instance
                              (used only for loading)
    :param policy_kwargs: (dict) additional arguments to be passed to the policy on creation
    :param full_tensorboard_log: (bool) enable additional logging when using tensorboard
        WARNING: this logging can take a lot of space quickly
    :param seed: (int) Seed for the pseudo-random generators (python, numpy, tensorflow).
        If None (default), use random seed. Note that if you want completely deterministic
        results, you must set `n_cpu_tf_sess` to 1.
    :param n_cpu_tf_sess: (int) The number of threads for TensorFlow operations
        If None, the number of cpu of the current machine will be used.
    """

    def __init__(self, policy, env, gamma=0.99, n_steps=5, vf_coef=0.25, ent_coef=0.01, max_grad_norm=0.5,
                 learning_rate=7e-4, alpha=0.99, momentum=0.0, epsilon=1e-5, lr_schedule='constant',
                 verbose=0, tensorboard_log=None, _init_setup_model=True, policy_kwargs=None,
                 full_tensorboard_log=False, seed=None, n_cpu_tf_sess=None):
#use lr_alpha and lr_beta to replace learning_rate
        self.n_steps = n_steps
        self.gamma = gamma
        self.vf_coef = vf_coef
        #self.v_mix_coef = v_mix_coef
        #self.v_ex_coef = v_ex_coef
        #self.r_ex_coef = r_ex_coef
        #self.r_in_coef = r_in_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.alpha = alpha
        self.momentum = momentum
        self.epsilon = epsilon
        self.lr_schedule = lr_schedule
        self.learning_rate = learning_rate
        #self.lr_alpha = lr_alpha
        #self.lr_beta = lr_beta
        self.tensorboard_log = tensorboard_log
        self.full_tensorboard_log = full_tensorboard_log

        self.learning_rate_ph = None
        self.n_batch = None
        self.actions_ph = None
        #A = tf.compat.v1.placeholder(tf.int32, [nbatch], 'A')
        self.advs_ph = None
        #ADV_EX = tf.compat.v1.placeholder(tf.float32, [nbatch], 'ADV_EX')
        self.rewards_ph = None
        #R_EX = tf.compat.v1.placeholder(tf.float32, [nbatch], 'R_EX')
        #RET_EX = tf.compat.v1.placeholder(tf.float32, [nbatch], 'RET_EX')
        #V_MIX = tf.compat.v1.placeholder(tf.float32, [nbatch], 'V_MIX')
        #DIS_V_MIX_LAST = tf.compat.v1.placeholder(tf.float32, [nbatch], 'DIS_V_MIX_LAST')
        #COEF_MAT = tf.compat.v1.placeholder(tf.float32, [nbatch, nbatch], 'COEF_MAT')
        #LR_ALPHA = tf.compat.v1.placeholder(tf.float32, [], 'LR_ALPHA')
        #LR_BETA = tf.compat.v1.placeholder(tf.float32, [], 'LR_BETA')
        self.pg_loss = None
        self.vf_loss = None
        self.entropy = None
        self.apply_backprop = None
        self.train_model = None
        self.step_model = None
        self.proba_step = None
        self.value = None
        self.initial_state = None
        self.learning_rate_schedule = None
        self.summary = None

        ### super(): 1. Allows us to avoid using the base class name explicitly
        ### 2. Working with Multiple Inheritance
        super(A2C_Original, self).__init__(policy=policy, env=env, verbose=verbose, requires_vec_env=True,
                                  _init_setup_model=_init_setup_model, policy_kwargs=policy_kwargs,
                                  seed=seed, n_cpu_tf_sess=n_cpu_tf_sess)

        # if we are loading, it is possible the environment is not known, however the obs and action space are known
        if _init_setup_model:
            self.setup_model()

    def _make_runner(self) -> AbstractEnvRunner:
        return A2CRunner(self.env, self, n_steps=self.n_steps, gamma=self.gamma)

    def _get_pretrain_placeholders(self):
        policy = self.train_model
        if isinstance(self.action_space, gym.spaces.Discrete):
            return policy.obs_ph, self.actions_ph, policy.policy
        return policy.obs_ph, self.actions_ph, policy.deterministic_action

    def setup_model(self):    # Part of the init in LIRPG A2C
        with SetVerbosity(self.verbose):
            # check if the input policy is in the class of A2C policies
            assert issubclass(self.policy, ActorCriticPolicy), "Error: the input policy for the A2C model must be an " \
                                                                "instance of common.policies.ActorCriticPolicy."

            self.graph = tf.Graph()
            with self.graph.as_default():
                self.set_random_seed(self.seed)
                self.sess = tf_util.make_session(num_cpu=self.n_cpu_tf_sess, graph=self.graph)    # returns a session that will use <num_cpu> CPU's only

                self.n_batch = self.n_envs * self.n_steps

                #line 55-56: Create step and train models
                n_batch_step = None
                n_batch_train = None
                if issubclass(self.policy, RecurrentActorCriticPolicy):
                    n_batch_step = self.n_envs
                    n_batch_train = self.n_envs * self.n_steps

                step_model = self.policy(self.sess, self.observation_space, self.action_space, self.n_envs, 1,
                                         n_batch_step, reuse=False, **self.policy_kwargs)
                # A context manager for defining ops that creates variables (layers).
                with tf.compat.v1.variable_scope("train_model", reuse=True,
                                       custom_getter=tf_util.outer_scope_getter("train_model")):
                    train_model = self.policy(self.sess, self.observation_space, self.action_space, self.n_envs,
                                              self.n_steps, n_batch_train, reuse=True, **self.policy_kwargs)

                #r_mix = r_ex_coef * R_EX + r_in_coef * tf.reduce_sum(train_model.r_in * tf.one_hot(A, nact), axis=1)
                # print("dimensions:", train_model.r_in, A, tf.reduce_sum(train_model.r_in * tf.one_hot(A, nact), axis=1))
                #print("dimensions:", COEF_MAT, r_mix, tf.reshape(r_mix, [nbatch, 1]),
                #      tf.matmul(COEF_MAT, tf.reshape(r_mix, [nbatch, 1])),
                #      tf.squeeze(tf.matmul(COEF_MAT, tf.reshape(r_mix, [nbatch, 1])), [1]))
                #ret_mix = tf.squeeze(tf.matmul(COEF_MAT, tf.reshape(r_mix, [nbatch, 1])), [1]) + DIS_V_MIX_LAST
                #adv_mix = ret_mix - V_MIX

                with tf.compat.v1.variable_scope("loss", reuse=False):
                    #line 59-62: modify ex only to mix
                    self.actions_ph = train_model.pdtype.sample_placeholder([None], name="action_ph")
                    self.advs_ph = tf.compat.v1.placeholder(tf.float32, [None], name="advs_ph")
                    self.rewards_ph = tf.compat.v1.placeholder(tf.float32, [None], name="rewards_ph")
                    self.learning_rate_ph = tf.compat.v1.placeholder(tf.float32, [], name="learning_rate_ph")

                    #print('@tf:')
                    #print('acts:', self.actions_ph)
                    #print('advs:', self.advs_ph)
                    #print('rwds:', self.rewards_ph)

                    #line 64-74: calculate loss
                    neglogpac = train_model.proba_distribution.neglogp(self.actions_ph)    #train_model.pd.neglogp in LIRPG
                    self.entropy = tf.reduce_mean(train_model.proba_distribution.entropy())   #check if the ent formula same with LIRPG cat_entropy(train_model.pi)?
                    self.pg_loss = tf.reduce_mean(self.advs_ph * neglogpac)
                    self.vf_loss = tf.reduce_mean(mse(tf.squeeze(train_model.value_flat), self.rewards_ph))
                    #rewards_ph is ret in LIRPG
                    # https://arxiv.org/pdf/1708.04782.pdf#page=9, https://arxiv.org/pdf/1602.01783.pdf#page=4
                    # and https://github.com/dennybritz/reinforcement-learning/issues/34
                    # suggest to add an entropy component in order to improve exploration.
                    # Calculate the loss
                    # Total loss = Policy gradient loss - entropy * entropy coefficient + Value coefficient * value loss
                    # Policy loss
                    # L = A(s,a) * -logpi(a|s)

                    loss = self.pg_loss - self.entropy * self.ent_coef + self.vf_loss * self.vf_coef
                    #policy_loss = pg_mix_loss - ent_coef * entropy + v_mix_coef * v_mix_loss

                    # record in tf.summary for results printed
                    tf.summary.scalar('entropy_loss', self.entropy)
                    tf.summary.scalar('policy_gradient_loss', self.pg_loss)
                    tf.summary.scalar('value_function_loss', self.vf_loss)
                    tf.summary.scalar('loss', loss)

                    # Update parameters using loss (policy params update in LIRPG)
                    #line 77-91: cal grad and train (train is to update)
                    self.params = tf_util.get_trainable_vars("model")  # 1. Get the model parameters
                    grads = tf.gradients(loss, self.params)  # 2. Calculate the gradients
                    if self.max_grad_norm is not None:  # max_grad_norm defines the maximum gradient, needs to be normalized
                        # Clip the gradients (normalize)
                        grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
                    grads = list(zip(grads, self.params))  # zip pg and policy params correspondingly, policy_grads_and_vars in LIRPG

                with tf.compat.v1.variable_scope("input_info", reuse=False):
                    tf.summary.scalar('discounted_rewards', tf.reduce_mean(self.rewards_ph))
                    tf.summary.scalar('learning_rate', tf.reduce_mean(self.learning_rate_ph))
                    tf.summary.scalar('advantage', tf.reduce_mean(self.advs_ph))
                    if self.full_tensorboard_log:
                        tf.summary.histogram('discounted_rewards', self.rewards_ph)
                        tf.summary.histogram('learning_rate', self.learning_rate_ph)
                        tf.summary.histogram('advantage', self.advs_ph)
                        if tf_util.is_image(self.observation_space):
                            tf.summary.image('observation', train_model.obs_ph)
                        else:
                            tf.summary.histogram('observation', train_model.obs_ph)
                # 3. Make up for one policy and value update step of A2C
                trainer = tf.compat.v1.train.RMSPropOptimizer(learning_rate=self.learning_rate_ph, decay=self.alpha,
                                                    epsilon=self.epsilon, momentum=self.momentum)
                self.apply_backprop = trainer.apply_gradients(grads)  #policy_train = policy_trainer.apply_gradients(policy_grads_and_vars)

                # line 150-159
                self.train_model = train_model
                self.step_model = step_model
                self.step = step_model.step
                self.proba_step = step_model.proba_step
                self.value = step_model.value
                self.initial_state = step_model.initial_state
                tf.compat.v1.global_variables_initializer().run(session=self.sess)

                self.summary = tf.compat.v1.summary.merge_all()

    def _train_step(self, obs, states, rewards, masks, actions, values, update, writer=None): #, r_ex, ret_ex, v_ex, v_mix, dis_v_mix_last, coef_mat
        """
        applies a training step to the model

        :param obs: ([float]) The input observations
        :param states: ([float]) The states (used for recurrent policies)
        :param rewards: ([float]) The rewards from the environment
        :param masks: ([bool]) Whether or not the episode is over (used for recurrent policies)
        :param actions: ([float]) The actions taken
        :param values: ([float]) The logits values
        :param update: (int) the current step iteration
        :param writer: (TensorFlow Summary.writer) the writer for tensorboard
        :return: (float, float, float) policy loss, value loss, policy entropy
        """
        #line 120-136: train() in LIRPG, make the training part (feedforward and retropropagation of gradients)
        advs = rewards - values
        #advs_ex = ret_ex - v_ex
        cur_lr = None
        for _ in range(len(obs)):
            cur_lr = self.learning_rate_schedule.value()
            #cur_lr_alpha = lr_alpha.value()
            #cur_lr_beta = lr_beta.value()
        assert cur_lr is not None, "Error: the observation input array cannon be empty"
        #assert cur_lr_alpha is not None, "Error: the observation input array cannon be empty"
        #assert cur_lr_abeta is not None, "Error: the observation input array cannon be empty"
        #print("compare:", actions, advs, rewards)
        td_map = {self.train_model.obs_ph: obs, self.actions_ph: actions, self.advs_ph: advs, self.rewards_ph: rewards,
                  #R_EX: r_ex, V_MIX:v_mix, DIS_V_MIX_LAST:dis_v_mix_last, COEF_MAT:coef_mat,
                  self.learning_rate_ph: cur_lr}
                  #LR_ALPHA:cur_lr_alpha, LR_BETA:cur_lr_beta}
        if states is not None:
            td_map[self.train_model.states_ph] = states
            td_map[self.train_model.dones_ph] = masks

        ### Using tf.summary.writer, else is the same; Is it same as save/load() in the baselines/a2c.py?
        ### The tf.summary module provides APIs for writing summary data. This data can be visualized in TensorBoard, the visualization toolkit that comes with TensorFlow.
        if writer is not None:
            # run loss backprop with summary, but once every 10 runs save the metadata (memory, compute time, ...)
            if self.full_tensorboard_log and (1 + update) % 10 == 0:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, policy_loss, value_loss, policy_entropy, _ = self.sess.run(
                    [self.summary, self.pg_loss, self.vf_loss, self.entropy, self.apply_backprop],
                    td_map, options=run_options, run_metadata=run_metadata)
                writer.add_run_metadata(run_metadata, 'step%d' % (update * self.n_batch))
            else:
                summary, policy_loss, value_loss, policy_entropy, _ = self.sess.run(
                    [self.summary, self.pg_loss, self.vf_loss, self.entropy, self.apply_backprop], td_map)
            writer.add_summary(summary, update * self.n_batch)

        else:
            policy_loss, value_loss, policy_entropy, _ = self.sess.run(
                [self.pg_loss, self.vf_loss, self.entropy, self.apply_backprop], td_map)

        return policy_loss, value_loss, policy_entropy


### Very different learn function, consider use this learn()? or migrate to the baselines/a2c.py (have gamescoremean and v_mix_ev (explained varaince); just change the extrinsic performance evaluation?)
    def learn(self, total_timesteps, callback=None, log_interval=100, tb_log_name="A2C_Orignal",
              reset_num_timesteps=True):

        new_tb_log = self._init_num_timesteps(reset_num_timesteps)
        callback = self._init_callback(callback)

        with SetVerbosity(self.verbose), TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name, new_tb_log) \
                as writer:
            self._setup_learn()
            self.learning_rate_schedule = Scheduler(initial_value=self.learning_rate, n_values=total_timesteps,
                                                    schedule=self.lr_schedule)
            #self.lr_alpha = Scheduler(initial_value=lr_alpha, n_values=total_timesteps, schedule=self.lr_schedule)
            #self.lr_beta = Scheduler(initial_value=lr_beta, n_values=total_timesteps, schedule=self.lr_schedule)

            t_start = time.time()
            callback.on_training_start(locals(), globals())

            for update in range(1, total_timesteps // self.n_batch + 1):

                callback.on_rollout_start()
                # true_reward is the reward without discount
                rollout = self.runner.run(callback)
                # unpack
                obs, states, rewards, masks, actions, values, ep_infos, true_reward = rollout
                callback.update_locals(locals())
                callback.on_rollout_end()

                # Early stopping due to the callback
                if not self.runner.continue_training:
                    break

                self.ep_info_buf.extend(ep_infos)
                policy_loss, value_loss, policy_entropy = self._train_step(obs, states, rewards, masks, actions, values,
                                                                 self.num_timesteps // self.n_batch, writer)
                f = open(
                    "C:/Users/Zihang Guan/Desktop/Deep-Reinforcement-Learning-for-Automated-Stock-Trading-Ensemble-Strategy-ICAIF-2020-master/results/convergence2/loss.csv",
                    'a', newline='')
                to_append = [
                    [update, policy_entropy, policy_loss,value_loss]]
                csvwriter = csv.writer(f)
                csvwriter.writerows(to_append)
                f.close()
                n_seconds = time.time() - t_start
                fps = int((update * self.n_batch) / n_seconds)

                if writer is not None:
                    total_episode_reward_logger(self.episode_reward,
                                                true_reward.reshape((self.n_envs, self.n_steps)),
                                                masks.reshape((self.n_envs, self.n_steps)),
                                                writer, self.num_timesteps)

                if self.verbose >= 1 and (update % log_interval == 0 or update == 1):
                    explained_var = explained_variance(values, rewards)
                    logger.record_tabular("nupdates", update)
                    logger.record_tabular("total_timesteps", self.num_timesteps)
                    logger.record_tabular("fps", fps)
                    logger.record_tabular("policy_entropy", float(policy_entropy))
                    logger.record_tabular("value_loss", float(value_loss))
                    logger.record_tabular("explained_variance", float(explained_var))
                    if len(self.ep_info_buf) > 0 and len(self.ep_info_buf[0]) > 0:
                        logger.logkv('ep_reward_mean', safe_mean([ep_info['r'] for ep_info in self.ep_info_buf]))
                        logger.logkv('ep_len_mean', safe_mean([ep_info['l'] for ep_info in self.ep_info_buf]))
                    logger.dump_tabular()

        callback.on_training_end()
        return self

    def save(self, save_path, cloudpickle=False):
        data = {
            "gamma": self.gamma,
            "n_steps": self.n_steps,
            "vf_coef": self.vf_coef,
            #"v_mix_coef": self.v_mix_coef,
            #"v_ex_coef": self.v_ex_coef
            "ent_coef": self.ent_coef,
            "max_grad_norm": self.max_grad_norm,
            "learning_rate": self.learning_rate,
            #"lr_alpha": self.lr_alpha,
            #"lr_beta": self.lr_beta,
            "alpha": self.alpha,
            "epsilon": self.epsilon,
            "lr_schedule": self.lr_schedule,
            "verbose": self.verbose,
            "policy": self.policy,
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "n_envs": self.n_envs,
            "n_cpu_tf_sess": self.n_cpu_tf_sess,
            "seed": self.seed,
            "_vectorize_action": self._vectorize_action,
            "policy_kwargs": self.policy_kwargs
        }
        #line 138-141: save() in LIRPG, save the model
        params_to_save = self.get_parameters()

        self._save_to_file(save_path, data=data, params=params_to_save, cloudpickle=cloudpickle)


class A2CRunner(AbstractEnvRunner):
    def __init__(self, env, model, n_steps=5, gamma=0.99):
        """
        A runner to learn the policy of an environment for an a2c model

        :param env: (Gym environment) The environment to learn from
        :param model: (Model) The model to learn
        :param n_steps: (int) The number of steps to run for each environment
        :param gamma: (float) Discount factor
        """
        #line 162-176: needs to modify the parameters in runner
        super(A2CRunner, self).__init__(env=env, model=model, n_steps=n_steps)
        self.gamma = gamma

    def _run(self):
        """
        Run a learning step of the model

        :return: ([float], [float], [float], [bool], [float], [float])
                 observations, states, rewards, masks, actions, values
        """
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones = [], [], [], [], []
        mb_states = self.states
        ep_infos = []
        for _ in range(self.n_steps):
            actions, values, states, _ = self.model.step(self.obs, self.states, self.dones)  # pytype: disable=attribute-error
            mb_obs.append(np.copy(self.obs))
            mb_actions.append(actions)
            mb_values.append(values)
            mb_dones.append(self.dones)
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.env.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.env.action_space.low, self.env.action_space.high)
            obs, rewards, dones, infos = self.env.step(clipped_actions)

            self.model.num_timesteps += self.n_envs    # update of step counter t

            if self.callback is not None:
                # Abort training early
                self.callback.update_locals(locals())
                if self.callback.on_step() is False:
                    self.continue_training = False
                    # Return dummy values
                    return [None] * 8

            for info in infos:
                maybe_ep_info = info.get('episode')
                if maybe_ep_info is not None:
                    ep_infos.append(maybe_ep_info)

            self.states = states
            self.dones = dones
            self.obs = obs
            mb_rewards.append(rewards)
        mb_dones.append(self.dones)
        # batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype).swapaxes(1, 0).reshape(self.batch_ob_shape)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(0, 1)
        mb_actions = np.asarray(mb_actions, dtype=self.env.action_space.dtype).swapaxes(0, 1)
        mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(0, 1)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(0, 1)
        mb_masks = mb_dones[:, :-1]
        mb_dones = mb_dones[:, 1:]
        true_rewards = np.copy(mb_rewards)
        last_values = self.model.value(self.obs, self.states, self.dones).tolist()  # pytype: disable=attribute-error
        # discount/bootstrap off value fn
        for n, (rewards, dones, value) in enumerate(zip(mb_rewards, mb_dones, last_values)):
            rewards = rewards.tolist()
            dones = dones.tolist()
            if dones[-1] == 0:
                rewards = discount_with_dones(rewards + [value], dones + [0], self.gamma)[:-1]
            else:
                rewards = discount_with_dones(rewards, dones, self.gamma)
            #print("rewards",rewards)
            mb_rewards[n] = rewards

        # convert from [n_env, n_steps, ...] to [n_steps * n_env, ...]
        mb_rewards = mb_rewards.reshape(-1, *mb_rewards.shape[2:])
        mb_actions = mb_actions.reshape(-1, *mb_actions.shape[2:])
        mb_values = mb_values.reshape(-1, *mb_values.shape[2:])
        mb_masks = mb_masks.reshape(-1, *mb_masks.shape[2:])
        true_rewards = true_rewards.reshape(-1, *true_rewards.shape[2:])
        return mb_obs, mb_states, mb_rewards, mb_masks, mb_actions, mb_values, ep_infos, true_rewards
#import argparse
    #parser = argparse.ArgumentParser()
    ## parser.add_argument('--env', help='Environment ID', default='BreakoutNoFrameskip-v4'), use env_train instead
    #parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    ##parser.add_argument('--policy', help='Policy architecture', choices=['cnn', 'lstm', 'lnlstm', 'cnn_int'],
    ##                    default='cnn_int'), policies for each algorithm is specified out already, may consider to modify to intrinsic augmented policy
    ##parser.add_argument('--lrschedule', help='Learning rate schedule', choices=['constant', 'linear'], default='linear'), sepcified as 'constant' in models (a2c etc.)
    ##parser.add_argument('--num-timesteps', type=int, default=int(50E6)), alr specified as input to each algorithm
    #parser.add_argument('--v-ex-coef', type=float, default=0.1)
    #parser.add_argument('--r-ex-coef', type=float, default=1)
    #parser.add_argument('--r-in-coef', type=float, default=0.01)
    ##parser.add_argument('--lr-alpha', type=float, default=7E-4)
    #parser.add_argument('--lr-beta', type=float, default=7E-4)
    #args = parser.parse_args()
    #logger.configure()
    #train( seed=args.seed,
    #      v_ex_coef=args.v_ex_coef, r_ex_coef=args.r_ex_coef, r_in_coef=args.r_in_coef)