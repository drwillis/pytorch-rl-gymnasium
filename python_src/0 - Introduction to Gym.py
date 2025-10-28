#!/usr/bin/env python
# coding: utf-8

# ## introduction

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt
import numpy as np
import ale_py
import gymnasium

print(gymnasium.__version__)


# https://gym.openai.com/envs/

# In[2]:


# all_envs = gymnasium.envs.registry.all
# env_ids = [env.id for env in all_envs]
key_names = list(gymnasium.envs.registry.keys())
for i in key_names:
    print(i)
print(f'There are {len(gymnasium.envs.registry.keys())} gym environments. Such as {key_names[5:10]}')


# https://gym.openai.com/envs/CartPole-v1/

# ## discrete action space environment

# In[3]:


env = gymnasium.make('CartPole-v1')


# In[4]:


print('observation space is:', env.observation_space)

print('is observation space discrete?', isinstance(env.observation_space, gymnasium.spaces.Discrete))
print('is observation space continuous?', isinstance(env.observation_space, gymnasium.spaces.Box))

print('observation space shape:', env.observation_space.shape)

print('observation space high values?', env.observation_space.high)
print('observation space low values?', env.observation_space.low)


# In[5]:


print('action space is:', env.action_space)

print('is action space discrete?', isinstance(env.action_space, gymnasium.spaces.Discrete))
print('is action space continuous?', isinstance(env.action_space, gymnasium.spaces.Box))

print('action space shape:', env.action_space.n)


# In[6]:


print(env.spec.max_episode_steps)
print(env.spec.reward_threshold)
print(env.spec.nondeterministic)


# ## continuous action space environment

# In[7]:


env = gymnasium.make('Pendulum-v1')


# In[8]:


print('observation space is:', env.observation_space)

print('is observation space discrete?', isinstance(env.observation_space, gymnasium.spaces.Discrete))
print('is observation space continuous?', isinstance(env.observation_space, gymnasium.spaces.Box))

print('observation space shape:', env.observation_space.shape)

print('observation space high values?', env.observation_space.high)
print('observation space low values?', env.observation_space.low)


# In[9]:


print('action space is:', env.action_space)

print('is action space discrete?', isinstance(env.action_space, gymnasium.spaces.Discrete))
print('is action space continuous?', isinstance(env.action_space, gymnasium.spaces.Box))

print('action space shape:', env.action_space.shape)

print('action space high values?', env.action_space.high)
print('action space low values?', env.action_space.low)


# In[10]:


print(env.spec.max_episode_steps)
print(env.spec.reward_threshold)
print(env.spec.nondeterministic)


# ## atari environments

# In[11]:


env = gymnasium.make('Freeway-v4')


# In[12]:


print('observation space is:', env.observation_space)
print('is observation space discrete?', isinstance(env.observation_space, gymnasium.spaces.Discrete))
print('is observation space continuous?', isinstance(env.observation_space, gymnasium.spaces.Box))
print('observation space shape:', env.observation_space.shape)


# In[13]:


print('action space is:', env.action_space)
print('action space shape:', env.action_space.n)
print('is action space discrete?', isinstance(env.action_space, gymnasium.spaces.Discrete))
print('is action space continuous?', isinstance(env.action_space, gymnasium.spaces.Box))
# print('action meanings:', env.unwrapped.get_action_meanings())


# In[14]:


print(env.spec.max_episode_steps)
print(env.spec.reward_threshold)
print(env.spec.nondeterministic)


# - Pong-v0 => 10k steps, randomly skips 2-4 frames, repeat action probability of 25%
# - Pong-v4 => 100k steps, randomly skips 2-4 frames
# - PongDeterministic-v0 => 100k steps, always skips 4 frames, repeat action probability of 25%
# - PongDeterministic-v4 => 100k steps, always skips 4 frames
# - PongNoFrameskip-v0 => 100k steps, returns every frame, repeat action probability of 25%
# - PongNoFrameskip-v4 => 100k steps, returns every frame
# 
# information about environments: https://github.com/openai/gym/blob/master/gym/envs/__init__.py#L604
# spaceinvadersdeterministic always skips 3 frames instead of 4: https://github.com/openai/gym/blob/master/gym/envs/__init__.py#L620
# default frameskip when one not provided: https://github.com/openai/gym/blob/master/gym/envs/atari/atari_env.py#L30
# when skipping frames, you repeat the last action: https://github.com/openai/gym/blob/master/gym/envs/atari/atari_env.py#L94

# ## wrappers
# 
# not exclusive to atari, but most commonly used for atari
# 
# commonly used atari wrappers: https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py

# In[15]:


class ClipRewardEnv(gymnasium.RewardWrapper):
    def __init__(self, env):
        gymnasium.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign."""
        return np.sign(reward)

class ScaledFloatFrame(gymnasium.ObservationWrapper):
    def __init__(self, env):
        gymnasium.ObservationWrapper.__init__(self, env)
        self.observation_space = gymnasium.spaces.Box(low=0, high=1, shape=env.observation_space.shape, dtype=np.float32)

    def observation(self, observation):
        return np.array(observation).astype(np.float32) / 255.0


# In[16]:


env = gymnasium.make('Pong-v4')

env = ClipRewardEnv(env)

env = ScaledFloatFrame(env)


# ## interacting with an environment

# In[17]:


env = gymnasium.make('CartPole-v1')

# env.seed(1234)

state = env.reset()

print('state type:', type(state))
print('state shape:', env.observation_space.shape)
print('state:', state)


# https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py#L21
# 
# cart pos, cart velocity, pole angle, pole velocity

# In[18]:


action = env.action_space.sample() #select random action, uniformly between high and low for continuous

print('selected action:', action)


# https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py#L29
# 
# 0 = left, 1 = right

# In[19]:


state, reward, done, info, _ = env.step(action) #perform action on environment

print('state:', state)
print('reward:', reward)
print('done:', done)
print('info:', info)


# ## interacting with the atari environment

# In[20]:


env = gymnasium.make('FreewayNoFrameskip-v4', render_mode="rgb_array")

# env.seed(1234)

state = env.reset()

print('state type:', type(state))
print('state shape:', env.observation_space.shape)


# In[21]:


frame = env.render()
plt.imshow(frame)


# In[22]:


action = env.action_space.sample() #select random action, uniformly between high and low for continuous

print('selected action:', action)
print('action meaning:', env.unwrapped.get_action_meanings()[action])


# In[23]:


state, reward, done, info, _ = env.step(action) #perform action on environment

print('reward:', reward)
print('done:', done)
print('info:', info)


# In[24]:


plt.imshow(state);


# In[25]:


up_action = env.unwrapped.get_action_meanings().index('UP')

for i in range(50):
    state, reward, done, info, _ = env.step(up_action) #presses up 10 times

plt.imshow(state);


# ## reinforcement learning loop

# In[26]:


env = gymnasium.make('SpaceInvadersNoFrameskip-v4', render_mode="rgb_array")

# env.seed(1234)

n_episodes = 10

for episode in range(n_episodes):

    episode_reward = 0
    done = False
    state = env.reset()

    while not done:

        action = env.action_space.sample()

        state, reward, done, _, _ = env.step(action)

        episode_reward += reward

    print(f'episode: {episode+1}, reward: {episode_reward}')


# In[ ]:




