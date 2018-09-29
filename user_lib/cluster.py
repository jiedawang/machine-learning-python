# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import time
from numba import jit
from concurrent.futures import ThreadPoolExecutor,as_completed
from multiprocessing import cpu_count
import math

#（参数校验和帮助信息暂时先不写了，太麻烦）

#k均值
#基于划分的聚类算法，不适用于非凸簇分布
#由于该算法基于空间距离，应用核函数可以显著加强聚类能力，不过暂时不准备实现
#算法流程：随机选取几个对象作为初始的簇中心，
#将所有对象各自划分给最近的簇中心，
#更新每个簇中心为簇内对象的质心，
#迭代划分簇和更新簇中心直至收敛
class KMeans:
    
    def __init__(self,clusters_n=10,iter_max=100):
        self.clusters_n=clusters_n
        self.iter_max=iter_max
        self.is_fitted=False
        
    #代价函数
    #所有数据点与各自所属簇的中心点的距离的平均值
    def cost_(self,X,g,C,n):
        D=np.zeros(X.shape[0])
        for i in range(n):
            D[g==i]=distance_(X[g==i],C[i])
        return D.mean()
    
    #划分数据集
    def divide_(self,D):
        g=np.argmin(D,axis=1)
        return g
    
    #初始化聚类中心
    #kmpp代表k-means++模式，会倾向于选择相隔较远的聚类中心
    def initial_centers_(self,X,n,kmpp=False):
        if kmpp==False:
            random_idx=np.random.randint(0,X.shape[0],n)
            return X[random_idx].copy()
        else:
            C=np.zeros((n,X.shape[1]))
            for i in range(n):
                if i==0:
                    p=np.ones(X.shape[0])
                    p/=p.sum()
                else:
                    D=distance_(X,C[:i])
                    if len(D.shape)>1:
                        D=D.min(axis=1)
                    D2=D*D
                    p=D2/D2.sum()
                random_idx=np.random.choice(range(X.shape[0]),size=1,p=p)[0]
                C[i]=X[random_idx]
            return C

    #更新聚类中心
    def update_centers_(self,X,g,n):
        C=np.zeros((n,X.shape[1]))
        for i in range(n):
            C[i]=X[g==i].mean(axis=0)
        return C
    
    #拟合
    #kmpp表示k-means++模式，初始化时会倾向于选择相隔较远的聚类中心
    def fit(self,X,kmpp=False):
        start=time.clock()
        #初始化簇中心
        C=self.initial_centers_(X,self.clusters_n,kmpp)
        #迭代优化
        cost_h=[]
        for i in range(self.iter_max):
            #数据点与簇中心的距离计算
            D=distance_(X,C)
            #将数据点分配给最近的簇中心
            g=self.divide_(D)
            #代价计算
            cost_h.append(self.cost_(X,g,C,self.clusters_n))
            #更新簇中心为簇内数据点的质心
            C_=self.update_centers_(X,g,self.clusters_n)
            #无变化时提前结束
            if np.abs((C-C_)).max()==0.:
                break
            C=C_
        self.centers=C
        self.cost_h=pd.Series(cost_h,name='cost')
        self.is_fitted=True
        print('\nfinished at iter %d'%(i+1))
        print('time used for training: %f'%(time.clock()-start))
    
    #预测    
    def predict(self,X):
        start=time.clock()
        D=distance_(X,self.centers)
        g=self.divide_(D)
        print('\ntime used for predicting: %f'%(time.clock()-start))
        return g

    #评估
    #外在方法评估
    #return_dist表示是否返回混淆矩阵
    def assess(self,pred,y,return_dist=False):
        start=time.clock()
        a,b,c,d=match_(pred,y)
        print('\ntime used for assessing: %f'%(time.clock()-start))
        if return_dist==False:
            return jaccard_(a,b,c)
        else:
            dist=pd.DataFrame([[a,b],[c,d]],
                              index=['y_same,y_diff'],
                              columns=['p_same','p_diff'])
            return jaccard_(a,b,c),dist
    
    #内在方法评估
    #s_detail表示是否返回全部对象的轮廓系数，False时只会返回整体平均值
    def assess_inner(self,pred,X,s_detail=False):
        start=time.clock()
        dbi=dbi_(X,pred,self.centers,self.clusters_n)
        if X.shape[0]<=10000:
            dvi=dvi_fast_compute_(X,pred,self.clusters_n)
            s=silhouette_fast_compute_(X,pred,self.clusters_n)
        else:
            dvi=dvi_low_memory_(X,pred,self.clusters_n)
            s=silhouette_low_memory_(X,pred,self.clusters_n)
        result=pd.Series([dbi,dvi,s.mean()],
                          index=['dbi','dvi','silhouette'],
                          name='value')
        print('\ntime used for assessing: %f'%(time.clock()-start))
        if s_detail==False:
            return result
        else:
            return result,s

#具有噪声应用的基于密度的空间聚类
#将足够高密度的区域划分为簇，能够在含有噪声的数据空间中发现任意形状的簇
class DBSCAN:
    
    def __init__(self,eps=0.1,min_pts=10):
        self.eps=eps
        self.min_pts=min_pts

    #数据点访问
    def visit_(self,X,eps,min_pts):
        #访问标记，分组标记，可达点数量
        visit=np.zeros(X.shape[0]).astype(np.int64)
        group=-np.ones(X.shape[0]).astype(np.int64)
        reach=np.zeros(X.shape[0]).astype(np.int64)
        #查找核心点起点
        for i in range(X.shape[0]):
            #已访问过，跳过
            if visit[i]==1:
                continue
            #计算领域内可达点
            reached=self.reach_(i,X,eps,visit)
            #可达点数量不满足要求，非核心点，跳过
            reach[i]=len(reached)
            if len(reached)<min_pts:
                continue
            #标记已访问，标记分组
            visit[i]=1
            group[i]=group.max()+1
            #访问密度可达点
            while len(reached)>0:
                #取一个待访问点
                current=reached.pop(0)
                #已访问过，跳过
                if visit[current]==1:
                    continue
                #标记已访问，标记分组（与核心点起点一致）
                visit[current]=1
                group[current]=group[i]
                #计算领域内可达点
                reached_=self.reach_(current,X,eps,visit)
                #该点也是核心点，将对应的密度可达点添加进待访问队列
                reach[current]=len(reached_)
                if len(reached_)>=min_pts:
                    reached.extend(reached_)
        return group,reach
    
    #数据点可达             
    def reach_(self,i,X,eps,visit):
        #起点
        x0=X[i]
        #与其他点的距离
        d=distance_(x0,X)
        #可达点，需要筛去已访问过的点和起点
        reached=np.where((d<=eps)&(visit==0))[0].tolist()
        if visit[i]==0:
            reached.remove(i)
        return reached
    
    #离群点划分
    def divide_outlier_(self,group,X):
        outlier_idx=(group==-1)
        outlier=np.where(outlier_idx)[0]
        normal=np.where(~outlier_idx)[0]
        for i in range(outlier.shape[0]):
            x0=X[outlier[i]]
            d=distance_(x0,X[~outlier_idx])
            nestest=np.argmin(d)
            group[outlier[i]]=group[normal[nestest]]
        return group
    
    #拟合预测（该算法没有将拟合和预测分开的必要）
    def fit_predict(self,X,divide_outlier=False,return_reach_n=False):
        start=time.clock()
        #通过基于密度可达的数据点访问发现簇分布
        group,reach=self.visit_(X,self.eps,self.min_pts)
        #反馈一些统计信息
        print('\nsamples: %d  noises: %d'
              %(X.shape[0],(group==-1).sum()))
        print('groups: %d  reachs: %d~%d'
              %(np.unique(group[group!=-1]).shape[0],reach.min(),reach.max()))
        #离群点处理
        if divide_outlier==True:
            group=self.divide_outlier_(group,X)
        print('\ntime used for predicting: %f'%(time.clock()-start))
        if return_reach_n==False:
            return group
        else:
            return group,reach
        
    #评估
    #外在方法评估
    #return_dist表示是否返回混淆矩阵
    def assess(self,pred,y,return_dist=False):
        start=time.clock()
        a,b,c,d=match_(pred,y)
        print('\ntime used for assessing: %f'%(time.clock()-start))
        if return_dist==False:
            return jaccard_(a,b,c)
        else:
            dist=pd.DataFrame([[a,b],[c,d]],
                              index=['y_same,y_diff'],
                              columns=['p_same','p_diff'])
            return jaccard_(a,b,c),dist

#凝聚式层次聚类
class Agglomerative:
    
    def __init__(self,clusters_n=10):
        self.clusters_n=clusters_n
    
    #拟合预测
    #return_all=True时返回所有层次
    def fit_predict(self,X,return_all=False):
        start=time.clock()
        D=distance_(X,X)
        g_h=agglomerative(D)
        if return_all==False:
            g=g_h[:,self.clusters_n-1]
            g_l=np.unique(g)
            #重新给聚类编号
            for k in range(self.clusters_n):
                g[g==g_l[k]]=k
            print('\ntime used for predicting: %f'%(time.clock()-start))
            return g
        else:
            print('\ntime used for predicting: %f'%(time.clock()-start))
            return g_h
    
#凝聚
@jit(nopython=True,nogil=True,cache=True)
def agglomerative(D):
    g=np.arange(0,D.shape[0])
    g_l=g.copy()
    g_h=np.zeros(D.shape).astype(np.int32)
    g_h[:,g_l.shape[0]-1]=g
    #每次将最近的两个聚类凝聚为一个
    while g_l.shape[0]>1:
        d_min=np.inf
        a,b=-1,-1
        #聚类之间两两配对计算平均距离
        for i in range(g_l.shape[0]-1):
            for j in range(i+1,g_l.shape[0]):
                D_=D[g==g_l[i]][:,g==g_l[j]]
                d_avg=D_.mean()
                if d_avg<d_min:
                    d_min=d_avg
                    a,b=i,j
        g[g==g_l[a]]=g_l[b]
        g_l=np.unique(g)
        g_h[:,g_l.shape[0]-1]=g
    return g_h
            
#两个集合数据点之间的距离计算
#x1,x2各为A,B中的一个坐标向量
#d=sqrt(sum((x1-x2)**2))=sqrt(<x1,x1>+<x2,x2>-2*<x1,x2>)
#A.shape=(samples_n,features_n)
#B.shape=(centers_n,features_n)
#D.shape=(samples_n,centers_n)
def distance_(A,B):
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
    return D

#聚类结果与基准结果的符合程度的几个基本度量，可构成混淆矩阵
#记录的两两配对，统计符合如下条件的配对数量 
#   预测同属一类且基准同属一类/预测同属一类且基准不属一类
#   预测不属一类且基准同属一类/预测不属一类且基准不属一类
@jit(nopython=True,nogil=True,cache=True)
def match_(pred,y):
    a,b,c,d=0,0,0,0
    n=len(y)
    for i in range(n-1):
        for j in range(i+1,n):
            if (y[i]==y[j])&(pred[i]==pred[j]):
                a+=1
            elif (y[i]==y[j])&(pred[i]!=pred[j]):
                b+=1
            elif (y[i]!=y[j])&(pred[i]==pred[j]):
                c+=1
            else:
                d+=1
    return a,b,c,d
    
#Jaccard系数,评估聚类质量的外在方法之一
def jaccard_(a,b,c):
    return a/(a+b+c)

#紧致性
#簇内元素与簇中心的平均距离的平均
def compactness_(X,g,C,n):
    CP_=0.
    for i in range(n):
        D=distance_(X[g==i],C[i])
        CP_+=D.mean()
    return CP_/n
    
#间隔性
#簇间簇中心之间的平均距离
def separation_(C,n):
    SP_=0.
    for i in range(n-1):
        D=distance_(C[i],C[i+1:])
        SP_+=D.sum()
    return SP_*2/n/(n-1)
    
#Davies-Bouldin指数
#任意两簇的簇内元素与中心的平均距离的两簇之和除以簇中心间的距离，所有组合中取最大值
#该指数越小意味着类内距离越小,同时类间距离越大
#不适用于非凸簇分布
def dbi_(X,g,C,n):
    buf1=np.zeros(n)
    for i in range(n):
        buf2=np.zeros(n)
        for j in range(n):
            if i==j:
                buf2[j]=-1
                continue
            CP1=distance_(X[g==i],C[i]).mean()
            CP2=distance_(X[g==j],C[j]).mean()
            SP=distance_(C[i],C[j])
            buf2[j]=(CP1+CP2)/SP
        buf1[i]=buf2.max()
    return buf1.mean()
    
#Dunn指数
#同簇元素间最小距离除以异簇元素间最大距离
#该指数越大意味着类间距离越大，同时类内距离越小
#不适用于非凸簇分布

#低内存占用版本
@jit(nopython=True,nogil=True,cache=True)
def dvi_low_memory_(X,g,n):
    outer_min,inner_max=np.inf,0.
    for i in range(X.shape[0]-1):
        for j in range(i+1,X.shape[0]):
            buf=X[i]-X[j]
            d=np.sqrt(np.dot(buf,buf))
            #d=np.sqrt(((X[i]-X[j])**2).sum())
            if (g[i]==g[j])&(d>inner_max)&(i!=j):
                inner_max=d
            if (g[i]!=g[j])&(d<outer_min):
                outer_min=d
    return outer_min/inner_max

#性能更高的版本，是上面一个的三倍
#但内存占用很高，X大小超过50000基本就炸了
def dvi_fast_compute_(X,g,n):
    inner_max=np.zeros(n)
    outer_min=np.zeros(n)
    for i in range(n):
        D=distance_(X[g==i],X[g==i])
        inner_max[i]=D.max()
        D=distance_(X[g==i],X[g!=i])
        outer_min[i]=D.min()
    return outer_min.min()/inner_max.max()
    
#轮廓系数
#对每个数据点o，
#a(o)=与同簇数据点距离的平均值
#b(o)=与异簇数据点距离的簇平均值的最小值
#s(o)=(b(o)-a(o))/max(a(o),b(o))
@jit(nopython=True,nogil=True,cache=True)
def silhouette_low_memory__(X,g,n):
    sum_=np.zeros((X.shape[0],n))
    cnt_=np.zeros((X.shape[0],n))
    for i in range(X.shape[0]-1):
        g_i=g[i]
        for j in range(i+1,X.shape[0]):
            g_j=g[j]
            buf=X[i]-X[j]
            d=np.sqrt(np.dot(buf,buf))
            #d=np.sqrt(((X[i]-X[j])**2).sum())
            sum_[i,g_j]+=d
            cnt_[i,g_j]+=1
            sum_[j,g_i]+=d
            cnt_[j,g_i]+=1
    mean_=sum_/cnt_
    return mean_

def silhouette_low_memory_(X,g,n):
    mean_=silhouette_low_memory__(X,g,n)
    a_o=np.zeros(X.shape[0])
    for k in range(n):
        a_o[g==k]=mean_[g==k,k]
        mean_[g==k,k]=np.inf
    b_o=mean_.min(axis=1)
    ab_max=a_o.copy()
    ab_max[b_o>a_o]=b_o[b_o>a_o]
    s=(b_o-a_o)/ab_max
    return s

def silhouette_fast_compute_(X,g,n):
    s=np.zeros(X.shape[0])
    for i in range(n):
        X_=X[g==i]
        D=distance_(X_,X_)
        a_o=D.sum(axis=1)/(X_.shape[0]-1)
        buf=np.zeros((X_.shape[0],n))
        for j in range(n):
            if i==j:
                buf[:,j]=np.inf
                continue
            D=distance_(X_,X[g==j])
            buf[:,j]=D.mean(axis=1)
        b_o=buf.min(axis=1)
        ab_max=a_o.copy()
        ab_max[b_o>a_o]=b_o[b_o>a_o]
        s[g==i]=(b_o-a_o)/ab_max
    return s

#测试   
if ( __name__ == '__main__' ):
    test=KMeans()
    a=np.array([[1,2]]).repeat(10000,axis=0)
    b=np.array([[3,4]]).repeat(10,axis=0)
    start=time.clock()
    d=distance_(a,b)
    print(time.clock()-start)