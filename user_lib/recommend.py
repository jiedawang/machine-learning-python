# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import time
from numba import jit
from concurrent.futures import ThreadPoolExecutor,as_completed
from multiprocessing import cpu_count
import math

#（参数校验和帮助信息暂时先不写了，太麻烦）

#协同过滤
#目前性能还很不满意，以后会优化
class CollaborativeFiltering:
    
    #top_n: 根据相似度前n的对象进行推荐
    #mode: 模式，'item'->基于项目，'user'->基于用户 
    #similarity: 相似度指标，'cosine'->余弦相似度，'distance'->欧式距离相似度
    def __init__(self,top_n=10,mode='item',similarity='distance'):
        self.top_n=top_n
        self.mode=mode
        self.similarity=similarity
    
    #基于用户的协同过滤    
    def user_cf_(self,X,X0):
        #推荐评分矩阵
        #将用户未接触过的项目初始化为1
        #已接触过的项目初始化为0
        R=np.zeros_like(X0)
        R[X0==0]=1.
        for i in range(X0.shape[0]):
            #X0中当前用户与X中其他用户在项目上的相似度
            if self.similarity=='cosine':
                s=cosine_similarity_(X0[i],X)
            else:
                s=distance_similarity_(X0[i],X)
            #相似度排序
            sort_idx=np.argsort(-s)
            #剔除相似度为1的用户(没有可推荐内容)
            ignore=np.where(s==1)[0]
            sort_idx=sort_idx[~np.isin(sort_idx,ignore)]
            #相似度排序并选择前n个用户
            top_s=sort_idx[:self.top_n]
           #将X0中当前用户与X中其他用户的相似度和X中其他用户对各项目评分相乘
           #然后再归一化
            s_=s[top_s]
            r=np.dot(s_,X[top_s])
            if s_.sum()!=0:
                r/=s_.sum()
            R[i]*=r
        return R
    
    #基于项目的协同过滤
    #该种方式更稳定且计算更快
    #可以将相似度矩阵存储下来以加快预测速度，此处先不实现了
    def item_cf_(self,X,X0):
        #推荐评分矩阵
        #将用户未接触过的项目初始化为1
        #已接触过的项目初始化为0
        R=np.zeros_like(X0)
        R[X0==0]=1.
        R_=np.zeros_like(X0)
        S_=np.zeros_like(X0)
        '''
        if self.similarity=='cosine':
            S=cosine_similarity_(X.T,X.T)
        else:
            S=distance_similarity_(X.T,X.T)
        '''
        for j in range(X0.shape[1]):
            #X中当前项目与X中其他项目在用户上的相似度
            #注意，此处当前对象的选择和user-cf有些不同
            if self.similarity=='cosine':
                s=cosine_similarity_(X[:,j],X.T)
            else:
                s=distance_similarity_(X[:,j],X.T)
            #s=S[j]
            #相似度排序并根据前n个进行推荐
            #相似度第一的是当前项目自身，需要剔除
            top_s=np.argsort(-s)[1:self.top_n+1]
            #将X中当前项目与其他项目的相似度和X0中当前项目评分相乘
            s_=np.zeros_like(s)
            s_[top_s]=s[top_s]
            x0_j=X0[:,j]
            R_+=x0_j.reshape((x0_j.shape[0],1))*s_
            S_+=s_
        #归一化并剔除用户已接触过的项目
        S_[S_==0.]=1.
        R*=R_/S_
        return R
    
    #拟合预测
    #X为参考矩阵，X0为需要进行推荐的矩阵
    #shape=(users_n,items_n)
    def fit_predict(self,X,X0=None):
        start=time.clock()
        #未传入X0时直接在X内部计算推荐
        if type(X0)==type(None):
            X0=X
        #X0变形
        if len(X0.shape)==1:
            X0=X0.reshape((1,X0.shape[0]))
        #两种推荐模式：基于用户/基于项目
        if self.mode=='user':
            R=self.user_cf_(X,X0)
        else:
            R=self.item_cf_(X,X0)
        print('\ntime used for predicting: %f'%(time.clock()-start))
        if R.shape[0]==1:
            R=R[0]
        return R
    
    #获取推荐列表
    def recommend_list(self,r,items,n=None,to_list=False):
        if type(n)==type(None):
            n=r.shape[0]
        sort_idx=np.argsort(-r)
        r=sort_idx[np.where(r[sort_idx]>0)[0]][:n]
        if to_list==False:
            return items[r]
        else:
            return [items[idx].tolist() for idx in r]
            
#余弦相似度
#=cos(向量夹角)=<a,b>/sqrt(<a,a>*<b,b>)
#该相似度只对比了向量的方向，没有考虑向量的长度，
#一旦数据是诸如评分的有大小之分的度量，就不是很合适
#(但在面对正交向量时，余弦相似度的值似乎更为合理)
def cosine_similarity_(A,B):
    A2=(A*A).T.sum(axis=0)
    B2=(B*B).T.sum(axis=0)
    if (len(A.shape)>1)&(len(B.shape)>1):
        A2=A2.reshape((A.shape[0],1))
        B2=B2.reshape((1,B.shape[0]))
    AB=np.dot(A,B.T)
    S=AB/np.sqrt(A2*B2)
    return S
   
#距离相似度
#=1/(1+d(a,b))
#d(a,b)=sqrt(<a,a>+<b,b>-2*<a,b>)
def distance_similarity_(A,B):
    A2=(A*A).T.sum(axis=0)
    B2=(B*B).T.sum(axis=0)
    if (len(A.shape)>1)&(len(B.shape)>1):
        A2=A2.reshape((A.shape[0],1))
        B2=B2.reshape((1,B.shape[0]))
    AB=np.dot(A,B.T)
    S=A2+B2-2*AB
    if type(S)==type(A):
        S[S<0]=0.
    else:
        if S<0:S=0.
    D=np.sqrt(S)
    return 1./(1.+D)        
        