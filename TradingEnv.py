# -*- coding: utf-8 -*-

import random

import gym
import numpy as np
import pandas as pd
from gym import spaces

INITIAL_ACCOUNT_BALANCE = 600
MAX_ACCOUNT_BALANCE = 10000
MAX_Price = 1000
fea_num = 23
leverage = 100


class TradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, df):
        super(TradingEnv, self).__init__()
        np.seterr(invalid='ignore')
        self.df = df
        self.reward_range = (-1*MAX_ACCOUNT_BALANCE, MAX_ACCOUNT_BALANCE)
        self.action_space = spaces.Box(low=np.array([-1, 0]), high=np.array([1, 1]), dtype=np.float16)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(fea_num, 1), dtype=np.float16)

        self.balance = INITIAL_ACCOUNT_BALANCE
        self.position = 0.5
        self.avg_price = 0
        self.times = 0
        self.orderBook = df

        self.orderBook.set_index(self.orderBook.columns[0], inplace=True)

        self.orderBook = self.orderBook[~self.orderBook.index.duplicated()]

        self.orderBook.loc[:, 'wp'] = self.orderBook['cf'].abs()

        lr = np.log(self.orderBook['wp']).diff(1)
        bpd1 = np.log(self.orderBook['bp1']*self.orderBook['bv1']).diff(1)
        bpd2 = np.log(self.orderBook['bp2']*self.orderBook['bv2']).diff(1)
        bpd3 = np.log(self.orderBook['bp3']*self.orderBook['bv3']).diff(1)
        bpd4 = np.log(self.orderBook['bp4']*self.orderBook['bv4']).diff(1)
        bpd5 = np.log(self.orderBook['bp5']*self.orderBook['bv5']).diff(1)
        apd1 = np.log(self.orderBook['ap1']*self.orderBook['av1']).diff(1)
        apd2 = np.log(self.orderBook['ap2']*self.orderBook['av2']).diff(1)
        apd3 = np.log(self.orderBook['ap3']*self.orderBook['av3']).diff(1)
        apd4 = np.log(self.orderBook['ap4']*self.orderBook['av4']).diff(1)
        apd5 = np.log(self.orderBook['ap5']*self.orderBook['av5']).diff(1)

        s1 = (self.orderBook['ap1']-self.orderBook['bp1']) / (self.orderBook['ap1']+self.orderBook['bp1'])
        s2 = (self.orderBook['ap2']-self.orderBook['bp2']) / (self.orderBook['ap2']+self.orderBook['bp2'])
        s3 = (self.orderBook['ap3']-self.orderBook['bp3']) / (self.orderBook['ap3']+self.orderBook['bp3'])
        s4 = (self.orderBook['ap4']-self.orderBook['bp4']) / (self.orderBook['ap4']+self.orderBook['bp4'])
        s5 = (self.orderBook['ap5']-self.orderBook['bp5']) / (self.orderBook['ap5']+self.orderBook['bp5'])

        cf2 = self.orderBook['cf'].rolling(2).sum()
        cf3 = self.orderBook['cf'].rolling(3).sum()
        cf4 = self.orderBook['cf'].rolling(4).sum()
        cf5 = self.orderBook['cf'].rolling(5).sum()

        lr = self.scale(lr, 5)
        bpd1 = self.scale(bpd1, 10)
        bpd2 = self.scale(bpd2, 10)
        bpd3 = self.scale(bpd3, 10)
        bpd4 = self.scale(bpd4, 10)
        bpd5 = self.scale(bpd5, 10)
        apd1 = self.scale(apd1, 10)
        apd2 = self.scale(apd2, 10)
        apd3 = self.scale(apd3, 10)
        apd4 = self.scale(apd4, 10)
        apd5 = self.scale(apd5, 10)
        s1 = self.scale(s1, 10)
        s2 = self.scale(s2, 10)
        s3 = self.scale(s3, 10)
        s4 = self.scale(s4, 10)
        s5 = self.scale(s5, 10)
        cf = self.scale(self.orderBook['cf'], 10)
        cf2 = self.scale(cf2, 10)
        cf3 = self.scale(cf3, 10)
        cf4 = self.scale(cf4, 10)
        cf5 = self.scale(cf5, 10)

        self.fea = pd.DataFrame([lr, bpd1, bpd2, bpd3, bpd4, bpd5, apd1, apd2, apd3, apd4, apd5, s1, s2, s3, s4, s5, cf, cf2, cf3, cf4, cf5], index=['lr', 'bpd1', 'bpd2',
                                                                                                                                                     'bpd3', 'bpd4', 'bpd5', 'apd1', 'apd2', 'apd3', 'apd4', 'apd5', 's1', 's2', 's3', 's4', 's5', 'cf', 'cf2', 'cf3', 'cf4', 'cf5']).T.dropna()
        self.orderBook = self.orderBook.loc[self.fea.index, :]
        print('load data down')

    def scale(self, series, d):
        series = series.dropna()
        return (series-series.rolling(d).min())/(series.rolling(d).max()-series.rolling(d).min())

    def getobs(self, curIdx):
        obs = np.r_[np.array(self.fea.iloc[curIdx, :]), np.array([self.position,
                                                                  self.avg_price / MAX_Price])].reshape(-1, 1)
        return obs

    def step(self, action):
        # low=np.array([-1, 0), high=np.array([1, 1]   order_side 正负表示多空，0忽略；order_size 是下单的仓位
        order_side = action[0]
        order_size = action[1]
        bidPrice = self.orderBook.iloc[self.curIdx]["bp1"]   # sell
        askPrice = self.orderBook.iloc[self.curIdx]["ap1"]   # buy
        markPrice = (bidPrice+askPrice)/2
        self.avg_price = self.avg_price*MAX_Price
        cur_positon = self.position - 0.5
        close_long_position = 0.0
        close_short_position = 0.0
        open_long_postion = 0.0
        open_short_postion = 0.0
        next_position = 0.0

        # open LONG / buy
        if order_side > 0:
            open_long_postion = order_size
            # 需要平空
            if cur_positon < 0:
                # 空减仓
                if abs(cur_positon) > order_size:
                    close_short_position = order_size

                # 平空开多
                else:
                    close_short_position = abs(cur_positon)

                open_long_postion = order_size - close_short_position
                self.avg_price = askPrice
                self.profit += (self.avg_price - askPrice) * close_short_position * leverage     # 平空收益

            # 做多头寸 无需平空
            if cur_positon == 0:
                self.avg_price = askPrice

            # 检查最大仓位
            next_position = cur_positon + order_size
            if next_position > 0.5:
                open_long_postion = open_long_postion - (next_position - 0.5)
                next_position = 0.5

            # 多头加仓
            if cur_positon > 0:
                self.avg_price = (cur_positon*self.avg_price + open_long_postion*askPrice)/(cur_positon+open_long_postion)

            self.profit -= (close_short_position+open_long_postion)*askPrice*0.00005   # 手续费

        # open SHORT / sell
        elif order_side < 0:
            open_short_postion = order_size
            # 需要平多
            if cur_positon > 0:
                # 多减仓
                if abs(cur_positon) > order_size:
                    close_long_position = order_size

                # 平多开空
                else:
                    close_long_position = abs(cur_positon)

                open_short_postion = order_size - close_long_position
                self.avg_price = bidPrice
                self.profit += (bidPrice - self.avg_price) * close_long_position * leverage     # 平多收益

            # 做空头寸 无需平多
            if cur_positon == 0:
                self.avg_price = bidPrice

            # 检查最大仓位
            next_position = cur_positon - order_size
            if next_position < -0.5:
                open_short_postion = open_short_postion - (-0.5 - next_position)
                next_position = -0.5

            # 空头加仓
            if cur_positon > 0:
                self.avg_price = (cur_positon*self.avg_price + open_short_postion*bidPrice)/(cur_positon+open_short_postion)

            self.profit -= (close_short_position+open_short_postion)*bidPrice*0.00005   # 手续费

        elif order_side == 0:
            pass

        if next_position > 0:
            self.unPNL = (markPrice - self.avg_price) * next_position * leverage

        if next_position < 0:
            self.unPNL = (self.avg_price - markPrice) * next_position * leverage

        self.position = next_position + 0.5
        self.curIdx = self.curIdx+1
        self.avg_price = self.avg_price/MAX_Price

        reward = self.unPNL + self.profit

        self.profits.append(reward/INITIAL_ACCOUNT_BALANCE)
        if len(self.profits) > 10:
            self.sharp = np.mean(np.array(self.profits))/np.std(np.array(self.profits))
        else:
            self.sharp = 0
        # reward = self.sharp

        obs = self.getobs(self.curIdx)

        if self.profit > self.max_profit:
            self.max_profit = self.profit
        if self.profit < self.max_loss:
            self.max_loss = self.profit

        # done = reward < -1*INITIAL_ACCOUNT_BALANCE/5 or reward > MAX_ACCOUNT_BALANCE-INITIAL_ACCOUNT_BALANCE or self.curIdx > (len(self.fea)-10)
        done = self.curIdx > (len(self.fea)-10)
        return obs, reward, done, {}

    def reset(self):
        # Reset the state of the environment to an initial state
        self.balance = INITIAL_ACCOUNT_BALANCE
        self.position = 0.5
        self.unPNL = 0.0
        self.avg_price = 0.0
        self.profit = 0.0
        self.profits = []
        self.sharp = 0.0
        self.curIdx = random.randint(0, 100000)
        self.times = self.times+1
        self.max_profit = -MAX_ACCOUNT_BALANCE
        self.max_loss = MAX_ACCOUNT_BALANCE
        obs = self.getobs(self.curIdx)
        # if self.times > 5:
        #     print('-------------------', self.times, ': sharp=,', self.sharp, '---------------------')
        return obs

    def render(self, mode='human', close=False):
        currentPrice = (self.orderBook['ap1'].iloc[self.curIdx] + self.orderBook['bp1'].iloc[self.curIdx]) / 2
        # netWorth = self.balance + self.position*self.orderBook['wp'].iloc[self.curIdx]
        # profit = netWorth - INITIAL_ACCOUNT_BALANCE
        # sharp = self.sharp
        curIdx = self.curIdx
        times = self.times
        profit = self.unPNL + self.profit
        # print('-----------------------times:', times, '--------------------------')
        print('curIdx: ', curIdx)
        print('currentPrice: ', currentPrice)
        print('position: ', self.position - 0.5)
        print('profit: ', profit)
        print('sharp: ', self.sharp)

        print('------------------------------------------------------------------------')


# head of csv
# time, update, Depth, bp1,bp2,bp3,bp4,bp5,bp6,bp7,bp8,bp9,bp10,bp11,bp12,bp13,bp14,bp15,bp16,bp17,bp18,bp19,bp20,bv1,bv2,bv3,bv4,bv5,bv6,bv7,bv8,bv9,bv10,bv11,bv12,bv13,bv14,bv15,bv16,bv17,bv18,bv19,bv20,ap1,ap2,ap3,ap4,ap5,ap6,ap7,ap8,ap9,ap10,ap11,ap12,ap13,ap14,ap15,ap16,ap17,ap18,ap19,ap20,av1,av2,av3,av4,av5,av6,av7,av8,av9,av10,av11,av12,av13,av14,av15,av16,av17,av18,av19,av20,Ticker,lp,lv,cf
