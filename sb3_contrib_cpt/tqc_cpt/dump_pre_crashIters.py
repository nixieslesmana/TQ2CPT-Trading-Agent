# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 13:39:35 2023

@author: Nixie S Lesmana
"""

## after actor_loss ##
if self._n_updates > 600 and self._n_updates % 60 == 0: # n_updates > self._total_timesteps - 200
    #print('check current_qtiles:', current_quantiles.shape) # LOSS ~ 112
    
    ### CHOOSE OBSERVATIONS THAT ARE VERY "FAR" FROM EACH OTHER #####
    # visualize replay_data get 256 from 1000?
    # far - compute_CPT(current_quantiles) - qf_pi DIFFERENT
    # Qtile seems to be ALL NEGATIVE - WHY???- ORI (Expectation) seemed to output -182 (means there are positive values?)
    # print('any positive CPT?', sorted(qf_pi.flatten().tolist()))
    # B << 0?
    
    for batchID in np.linspace(0, 255, 4):
        batchID = int(batchID)
        # replace '0' with batchID
        
        plt.figure() #fig = plt.figure(figsize = (20, 10))
        
        plt.plot(current_quantiles[batchID][0].tolist(), label = 'criticID=0')
        plt.plot(current_quantiles[batchID][1].tolist(), label = 'criticID=1')
        plt.plot(current_quantiles[batchID][2].tolist(), label = 'criticID=2')
        plt.legend()
        
        filename = './output/logs_{}_{}_{}.png'.format(self.name, self._n_updates, batchID)#, runID)                
        plt.savefig(filename, dpi = 200)
        plt.close()

'''
print('old:', compute_CPT(actor_quantiles.flatten()).item())
print('flat qtiles, sorted:', actor_quantiles.flatten().sort())
print('1st batch data, dim:', actor_quantiles[0].shape, '= (3 ,25)?', actor_quantiles[0])
print('---')
print('aft modif:', qf_pi)
print('qtiles_avged:', actor_quantiles.mean(1).shape, actor_quantiles.mean(1))
print('try compute cpt:', compute_CPT(actor_quantiles.mean(1)[0], sort = False), ', from qtiles[0]:', actor_quantiles.mean(1)[0])
print('===')

# old, aft modif outputs v different values! old is wrong as it applies CPT to \eta(x, a) pooled across different x, a.
'''


'''
if ent_coef_loss.item() > 0:
    print('POSITIVE ENTCOEF LOSS!')
    #print('log_ent_coef:', self.log_ent_coef) # --> must be positive (requires alpha > 1.)
    print('acts_pi:', actions_pi[0].tolist())
    print('log_prob:', log_prob.tolist()[0]) #log_policy --> positive by 'log'
    print('SEE distrib.py-Squashed for more debugs')
else:
    print('acts_pi:', actions_pi[0].tolist())#, '\n', actions_pi[-1].tolist())
    #print('try log_prob:', self.actor.action_dist.log_prob(th.Tensor(actions_pi[0].tolist())))
    print('actual log_prob:', log_prob.tolist()[0])
'''

####################### mostly NEGATIVE ZPred (k = 0 vs 2) ###########################
# Sort and drop top k quantiles to control overestimation.
next_quantiles, _ = th.sort(next_quantiles.reshape(batch_size, -1))
 
if len(pos_idx) > 1:    
    print('[!!] next_qtiles..')
    print('post sort[batch=0]:', min(next_quantiles[pos_idx[0]].tolist()),
          max(next_quantiles[pos_idx[0]].tolist())) #, next_quantiles[0][1].tolist(), next_quantiles[0][2].tolist())
    print('post sort[batch=-1]:', min(next_quantiles[pos_idx[-1]].tolist()),
          max(next_quantiles[pos_idx[-1]].tolist())) #, next_quantiles[-1][1].tolist(), next_quantiles[-1][2].tolist())

n_target_quantiles = self.critic.quantiles_total - self.top_quantiles_to_drop_per_net * self.critic.n_critics
next_quantiles = next_quantiles[:, :n_target_quantiles]

if len(pos_idx) > 1:                
    print('post drop[batch=0]:', min(next_quantiles[pos_idx[0]].tolist()),
          max(next_quantiles[pos_idx[0]].tolist()))
    print('post drop[batch=-1]:', min(next_quantiles[pos_idx[-1]].tolist()), 
          max(next_quantiles[pos_idx[-1]].tolist()))
   
# td error + entropy term: Distributional Soft Bellman operator
target_quantiles_ = next_quantiles - ent_coef * next_log_prob.reshape(-1, 1)
target_quantiles = replay_data.rewards + (1 - replay_data.dones) * self.gamma * target_quantiles_
target_quantiles.unsqueeze_(dim=1)
 
if len(pos_idx) > 1: 
    print('---')
    print('[!!] target_qtiles:', target_quantiles.shape) 
    print('batch[0]:', target_quantiles[pos_idx[0]].flatten().tolist())
    print('batch[-1]:', target_quantiles[pos_idx[-1]].flatten().tolist())
    print('---')
    print('rewards:', replay_data.rewards.shape)
    print('batch[0]:', replay_data.rewards[pos_idx[0]])
    print('batch[-1]:', replay_data.rewards[pos_idx[-1]])
    print('---')
    print('done_multiplier:', 1 - replay_data.dones[pos_idx[0]], 1 - replay_data.dones[pos_idx[-1]])
    print('discount:', self.gamma)
    print('[!!] z-alpha*next_logprob:', target_quantiles_.shape) 
    # CHECK HOW LARGE COMPARED TO TARGETQUANTILES -- THIS PART IS THE ISSUE!!!
    # Main cause: NEXT_QUANTILES (entropy not as much a problem)
    # TO-DO:
        # 1. Why becomes negative after some time? --> STLL HAVE POSITIVE!!!  but decrease from 256/256 to 53/256
        # 2. Is it from .rewards all negative?
        # 3. .dones have output 1 (by train.csv), but why .asset_memory keeps increasing when qtilePredict 
        # is very negative?
    print('batch[0]:', target_quantiles_[pos_idx[0]])
    print('batch[-1]:', target_quantiles_[pos_idx[-1]])
    print('---')
     
####################### STACK UPDATE TOGETHER #####################

# --- TQC-ori: Policy and alpha loss ---
ction, log_pi = self.actor(state)
_loss = -self.log_alpha * (log_pi + self.target_entropy).detach().mean()
_loss = (alpha * log_pi - self.critic(state, new_action).mean(2).mean(1, keepdim=True)).mean()

# Optimize the critic
self.critic.optimizer.zero_grad()
critic_loss.backward()
self.critic.optimizer.step()

# Optimize the actor
self.actor.optimizer.zero_grad()
#with th.autograd.detect_anomaly():
actor_loss.backward()
self.actor.optimizer.step()

# Optimize entropy coefficient, also called entropy temperature or alpha in the paper
if ent_coef_loss is not None:
    self.ent_coef_optimizer.zero_grad()
    ent_coef_loss.backward()
    self.ent_coef_optimizer.step()
###################################################################