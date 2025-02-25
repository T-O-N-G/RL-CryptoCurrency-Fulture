# -*- coding: utf-8 -*-

from TradingEnv import TradingEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
import pandas as pd
import numpy as np
import gym
import matplotlib.pyplot as plt

# macOS problem
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

dataset_dir = '/Users/tong/Downloads/eth20_binance_normal.csv'
# dataset_dir = './data/eth_test.csv'
cols = ['time', 'bp1', 'bp2', 'bp3', 'bp4', 'bp5', 'bv1', 'bv2', 'bv3', 'bv4', 'bv5', 'ap1', 'ap2', 'ap3', 'ap4', 'ap5', 'av1', 'av2', 'av3', 'av4', 'av5', 'lp', 'lv', 'cf']
df = pd.read_csv(dataset_dir, error_bad_lines=False, skip_blank_lines=True, usecols=cols, low_memory=False)

df = df.sort_values('time')
df = df.dropna()

env = DummyVecEnv([lambda: TradingEnv(df)])

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=47000)
# model.load("models/fucking_long")
obs = env.reset()

rewardlist = []
pricelist = []
pricelistL = []
pricelistS = []

for i in range(3000):
    action, _states = model.predict(obs)
    # print(action)
    obs, rewards, done, info = env.step(action)
    rewardlist.append(info[0]["profit"])
    # pricelist.append(info[0]["price"])
    if info[0]["position"] > 0.5:
        pricelistL.append(info[0]["price"])
        pricelistS.append(None)
        pricelist.append(None)

    elif info[0]["position"] < 0.5:
        pricelistS.append(info[0]["price"])
        pricelistL.append(None)
        pricelist.append(None)
    else:
        pricelist.append(info[0]["price"])
        pricelistL.append(None)
        pricelistS.append(None)

    # print("action: %.4f     position: %.4f" % (action[0][0], action[0][1]))
    if action[0][0] >= 0:
        print("action: \033[1;32m %.4f \033[0m " % action[0][1])
    else:
        print("action: \033[1;31m %.4f \033[0m " % action[0][1])
    env.render()

print("\n")
print("max_profit", max(rewardlist))
print("min_profit", min(rewardlist))
model.save("models/"+str(rewards[0]))

fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(range(len(pricelist)), pricelist, color='black')
ax1.plot(range(len(pricelist)), pricelistL, color='green')
ax1.plot(range(len(pricelist)), pricelistS, color='red')
ax1.set_title("price")
ax2.plot(range(len(rewardlist)), rewardlist)
ax2.set_title("profit")
plt.show()


# head of csv
# time,update,Depth,bp1,bp2,bp3,bp4,bp5,bp6,bp7,bp8,bp9,bp10,bp11,bp12,bp13,bp14,bp15,bp16,bp17,bp18,bp19,bp20,bv1,bv2,bv3,bv4,bv5,bv6,bv7,bv8,bv9,bv10,bv11,bv12,bv13,bv14,bv15,bv16,bv17,bv18,bv19,bv20,ap1,ap2,ap3,ap4,ap5,ap6,ap7,ap8,ap9,ap10,ap11,ap12,ap13,ap14,ap15,ap16,ap17,ap18,ap19,ap20,av1,av2,av3,av4,av5,av6,av7,av8,av9,av10,av11,av12,av13,av14,av15,av16,av17,av18,av19,av20,Ticker,lp,lv,cf


# clean train dataset
# cat eth20_binance_normal.csv | awk '{if (length($0) > 300 ) print $0}' > neweth
