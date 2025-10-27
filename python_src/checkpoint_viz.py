#!/usr/bin/env python
# coding: utf-8

# In[32]:


import torch
import os
import matplotlib.pyplot as plt


# In[46]:


checkpoints = os.listdir('checkpoints')

print(len(checkpoints))

all_rewards = []

for checkpoint in checkpoints:
    pth = os.path.join('checkpoints', checkpoint)
    rewards = torch.load(pth)
    if rewards.shape[-1] == 10_000:
        tot = max(rewards.sum(-1)).item()
        print(pth)
        print(tot)
        all_rewards.append((pth, rewards, tot))

print(len(all_rewards))


# In[47]:


all_rewards.sort(key=lambda x: x[-1])


# In[48]:


path, rewards, tot = all_rewards[-1]
print(path)
print(tot)
n_runs, n_episodes = rewards.shape
idxs = range(n_episodes)
fig, ax = plt.subplots(n_runs, figsize=(10,12))
for i, _ax in enumerate(ax):
    _ax.plot(idxs, rewards[i], c='red')
    _ax.plot(idxs, rewards[i], c='blue')
    _ax.set_ylim(0, 550)
    _ax.set_ylabel('Rewards');
    if i == n_runs - 1:
        _ax.set_xlabel('Episodes')


# In[39]:


min_e = 0.01
max_e = 1.0
decay = 0.999
import numpy as np
import math
(min_e/max_e)/np.log(decay)


# In[6]:


x = (min_e/max_e)/math.log(decay, decay)


# In[7]:


print(x)


# In[ ]:




