# -*- coding: utf-8 -*-
# macOS problem
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import gym
import pandas as pd
from stable_baselines3 import PPO
# from stable_baselines3.common.policies import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv

from TradingEnv import TradingEnv

dataset_dir = '/Users/tong/Downloads/bnb20_binance_normal.csv'
# dataset_dir = './data/eth_test.csv'

df = pd.read_csv(dataset_dir, error_bad_lines=False, usecols=['time','bp1','bp2','bp3','bp4','bp5','bv1','bv2','bv3','bv4','bv5','ap1','ap2','ap3','ap4','ap5','av1','av2','av3','av4','av5','lp','lv','cf'])
df = df.sort_values('time')

env = DummyVecEnv([lambda: TradingEnv(df)])

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)

obs = env.reset()

rewardlist = []

for i in range(50000):
    action, _states = model.predict(obs)
    # print(action)
    obs, rewards, done, info = env.step(action)
    rewardlist.append(rewards)
    env.render()

model.save("models/"+str(rewards[0]))

# head of csv
# time,update,Depth,bp1,bp2,bp3,bp4,bp5,bp6,bp7,bp8,bp9,bp10,bp11,bp12,bp13,bp14,bp15,bp16,bp17,bp18,bp19,bp20,bv1,bv2,bv3,bv4,bv5,bv6,bv7,bv8,bv9,bv10,bv11,bv12,bv13,bv14,bv15,bv16,bv17,bv18,bv19,bv20,ap1,ap2,ap3,ap4,ap5,ap6,ap7,ap8,ap9,ap10,ap11,ap12,ap13,ap14,ap15,ap16,ap17,ap18,ap19,ap20,av1,av2,av3,av4,av5,av6,av7,av8,av9,av10,av11,av12,av13,av14,av15,av16,av17,av18,av19,av20,Ticker,lp,lv,cf
