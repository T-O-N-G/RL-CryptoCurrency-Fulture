import random
import json
import gym
from gym import spaces
import pandas as pd
import numpy as np
import pdb
import datetime
INITIAL_ACCOUNT_BALANCE=2000
MAX_ACCOUNT_BALANCE=10000
MAX_Shape          =5
MIN_Shape          =-3
MAX_Coin           =10
fea_num            = 21

class StockTradingEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df):
        super(StockTradingEnv, self).__init__()

        self.df = df
        self.reward_range = (0, MAX_ACCOUNT_BALANCE)
        self.action_space = spaces.Box(
                    low=np.array([-1, 1, 2]), high=np.array([1,5,10]), dtype=np.float16)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(fea_num, 1), dtype=np.float16)
        
        self.quote=INITIAL_ACCOUNT_BALANCE
        self.base=0
        self.times=0
        self.btcOrderBook = df
        self.btcOrderBook.iloc[:,0]=self.btcOrderBook.iloc[:,0]/1000
        self.btcOrderBook.iloc[:,0]=\
        self.btcOrderBook.iloc[:,0].apply(datetime.datetime.utcfromtimestamp)
        self.btcOrderBook.set_index(self.btcOrderBook.columns[0],inplace=True)
        self.btcOrderBook=self.btcOrderBook[self.btcOrderBook['isu']]
        self.lp=self.btcOrderBook['lp']
        self.lv=self.btcOrderBook['lv']
        self.btcOrderBook=self.btcOrderBook[~self.btcOrderBook.index.duplicated()]
        ld=self.btcOrderBook['ld']
        ld[ld==1]=1
        ld[ld==2]=-1
        self.btcOrderBook['cf']=self.btcOrderBook['lv']*self.btcOrderBook['lp']*ld
        self.btcOrderBook['wp']=self.btcOrderBook['lv']*self.btcOrderBook['lp']
        x  = self.btcOrderBook.groupby(['bp1','bv1','ap1','av1'],sort=False)['wp'].sum()
        y  = self.btcOrderBook.groupby(['bp1','bv1','ap1','av1'],sort=False)['lv'].sum()
        cf = self.btcOrderBook.groupby(['bp1','bv1','ap1','av1'],sort=False)['cf'].sum()
        o  = self.btcOrderBook.groupby(['bp1','bv1','ap1','av1'],sort=False)['lp'].first()
        l  = self.btcOrderBook.groupby(['bp1','bv1','ap1','av1'],sort=False)['lp'].min()
        c  = self.btcOrderBook.groupby(['bp1','bv1','ap1','av1'],sort=False)['lp'].last()
        h  = self.btcOrderBook.groupby(['bp1','bv1','ap1','av1'],sort=False)['lp'].max()

        duplicate_row = self.btcOrderBook.duplicated(subset=['bp1','bv1','ap1','av1'],keep='first')
        self.btcOrderBook = self.btcOrderBook.loc[~duplicate_row,:]
        wp =x/y
        wp.index=self.btcOrderBook.index
        y.index=self.btcOrderBook.index
        cf.index=self.btcOrderBook.index
        o.index=self.btcOrderBook.index
        h.index=self.btcOrderBook.index
        l.index=self.btcOrderBook.index
        c.index=self.btcOrderBook.index
        self.btcOrderBook['lv']=y
        self.btcOrderBook['wp']=wp
        self.btcOrderBook['cf']=cf
        self.btcOrderBook['o']=o
        self.btcOrderBook['c']=c
        self.btcOrderBook['h']=h
        self.btcOrderBook['l']=l
        cf[cf>0]=1
        cf[cf<0]=-1
        self.btcOrderBook['ld']=cf
        self.btcOrderBook.drop(columns=['lp'],inplace=True)
        lr=np.log(self.btcOrderBook['wp']).diff(1)
        bpd1=np.log(self.btcOrderBook['bp1']*self.btcOrderBook['bv1']).diff(1)
        bpd2=np.log(self.btcOrderBook['bp2']*self.btcOrderBook['bv2']).diff(1)
        bpd3=np.log(self.btcOrderBook['bp3']*self.btcOrderBook['bv3']).diff(1)
        bpd4=np.log(self.btcOrderBook['bp4']*self.btcOrderBook['bv4']).diff(1)
        bpd5=np.log(self.btcOrderBook['bp5']*self.btcOrderBook['bv5']).diff(1)
        apd1=np.log(self.btcOrderBook['ap1']*self.btcOrderBook['av1']).diff(1)
        apd2=np.log(self.btcOrderBook['ap2']*self.btcOrderBook['av2']).diff(1)
        apd3=np.log(self.btcOrderBook['ap3']*self.btcOrderBook['av3']).diff(1)
        apd4=np.log(self.btcOrderBook['ap4']*self.btcOrderBook['av4']).diff(1)
        apd5=np.log(self.btcOrderBook['ap5']*self.btcOrderBook['av5']).diff(1)
        s1  =(self.btcOrderBook['ap1']-self.btcOrderBook['bp1'])/\
        (self.btcOrderBook['ap1']+self.btcOrderBook['bp1'])
        s2  =(self.btcOrderBook['ap2']-self.btcOrderBook['bp2'])/\
        (self.btcOrderBook['ap2']+self.btcOrderBook['bp2'])
        s3  =(self.btcOrderBook['ap3']-self.btcOrderBook['bp3'])/\
        (self.btcOrderBook['ap3']+self.btcOrderBook['bp3'])
        s4  =(self.btcOrderBook['ap4']-self.btcOrderBook['bp4'])/\
        (self.btcOrderBook['ap4']+self.btcOrderBook['bp4'])
        s5  =(self.btcOrderBook['ap5']-self.btcOrderBook['bp5'])/\
        (self.btcOrderBook['ap5']+self.btcOrderBook['bp5'])
        cf5= self.btcOrderBook['cf'].rolling(5).sum()
        cf3= self.btcOrderBook['cf'].rolling(3).sum()
        lr =self.scale(lr,5)
        bpd1 =self.scale(bpd1,10)
        bpd2 =self.scale(bpd2,10)
        bpd3 =self.scale(bpd3,10)
        bpd4 =self.scale(bpd4,10)
        bpd5 =self.scale(bpd5,10)
        apd1 =self.scale(apd1,10)
        apd2 =self.scale(apd2,10)
        apd3 =self.scale(apd3,10)
        apd4 =self.scale(apd4,10)
        apd5 =self.scale(apd5,10)
        s1 =self.scale(s1,10)
        s2 =self.scale(s2,10)
        s3 =self.scale(s3,10)
        s4 =self.scale(s4,10)
        s5 =self.scale(s5,10)
        cf =self.scale(self.btcOrderBook['cf'],10)
        cf5 =self.scale(cf5,10)
        cf3 =self.scale(cf3,10)
        self.fea=pd.DataFrame([lr,bpd1,bpd2,bpd3,bpd4,bpd5,apd1,apd2,apd3,apd4,apd5,s1,s2,s3,s4,s5,cf,cf5,cf3],index=\
                ['lr','bpd1','bpd2','bpd3','bpd4','bpd5','apd1','apd2','apd3','apd4','apd5','s1','s2','s3','s4','s5','cf','cf5','cf3']).T.dropna()
        self.btcOrderBook=self.btcOrderBook.loc[self.fea.index,:]
        print('load data down')


    def scale(self,series,d):
        series = series.dropna()
        return (series-series.rolling(d).min())/(series.rolling(d).max()-series.rolling(d).min())
    def getobs(self,curIdx):
        obs = np.r_[np.array(self.fea.iloc[curIdx, :]),\
                    np.array([self.quote / MAX_ACCOUNT_BALANCE,\
                              self.base  / MAX_Coin])].reshape(-1,1)
        return obs
    
    def step(self, action):
        volume,dw,cdSec = action
        cdSec = float(cdSec)
        dt    = self.btcOrderBook.iloc[self.curIdx,:].name
        bp    = self.btcOrderBook.iloc[self.curIdx,:25]
        bv    = self.btcOrderBook.iloc[self.curIdx,25:50]
        ap    = self.btcOrderBook.iloc[self.curIdx,50:75]
        av    = self.btcOrderBook.iloc[self.curIdx,75:100]
        highp = self.btcOrderBook['h'].loc[dt:dt+datetime.timedelta(seconds=cdSec)]
        lowp  = self.btcOrderBook['l'].loc[dt:dt+datetime.timedelta(seconds=cdSec)]
        if volume > 0:
            bidOrderPrice = np.array(bp)[bv.cumsum()>dw][0]
            if len(lowp[lowp<=bidOrderPrice])>0:
                self.quote = self.quote - volume * bidOrderPrice
                self.base  = self.base  + volume
                self.curIdx=len(self.btcOrderBook[self.btcOrderBook.index[0]:\
                                             lowp[lowp<=bidOrderPrice].index[0]])+1
            else:
                self.curIdx=self.curIdx+len(self.btcOrderBook[dt:dt+datetime.timedelta(seconds=cdSec)])
        elif volume < 0:
            askOrderPrice = np.array(ap)[av.cumsum()>dw][0]
            if len(highp[highp>=askOrderPrice])>0:
                self.quote = self.quote - volume * askOrderPrice
                self.base  = self.base  + volume
                self.curIdx= len(self.btcOrderBook[self.btcOrderBook.index[0]:\
                                                highp[highp>=askOrderPrice].index[0]])+1
            else:
                self.curIdx=self.curIdx+len(self.btcOrderBook[dt:dt+datetime.timedelta(seconds=cdSec)])
        elif volume == 0:
                self.curIdx=self.curIdx+len(self.btcOrderBook[dt:dt+datetime.timedelta(seconds=cdSec)])
        profit=self.quote + self.base*self.btcOrderBook['wp'].iloc[self.curIdx]-INITIAL_ACCOUNT_BALANCE
        self.profit.append(profit)
        if len(self.profit) > 10:
            reward = np.mean(np.array(self.profit))/np.std(np.array(self.profit))
        else:
            reward = 0
        self.sharp = reward
        obs   = self.getobs(self.curIdx)
        done  = reward <= MIN_Shape or reward > MAX_Shape or self.curIdx>(len(self.fea)-10)
        return obs, reward, done, {}

        return obs, reward, done, {}

    def reset(self):
        # Reset the state of the environment to an initial state
        self.quote=INITIAL_ACCOUNT_BALANCE
        self.base=0
        self.profit=[]
        self.curIdx = random.randint(0,10)
        self.times =  self.times+1
        obs   = self.getobs(self.curIdx)
        if self.times>5:
           print('-------------------第',self.times,'次开始，上次的tick级夏普是,',self.sharp,'---------------------')
        return obs

    def render(self, mode='human', close=False):
        currentPrice =  self.btcOrderBook['wp'].iloc[self.curIdx]
        netWorth     =  self.quote + self.base*self.btcOrderBook['wp'].iloc[self.curIdx]
        profit       =  netWorth - INITIAL_ACCOUNT_BALANCE
        sharp        =  self.sharp
        curIdx       =  self.curIdx
        times        =  self.times
        print('-----------------------',f'times: {times}','--------------------------')
        print(f'curIdx: {curIdx}')
        print(f'currentPrice: {currentPrice}')
        print(f'netWorth: {netWorth}')
        print(f'Profit: {profit}')
        print(f'sharp: {sharp}')