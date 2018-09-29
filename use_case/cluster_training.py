# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import user_lib.data_prep as dp
import user_lib.cluster as ct
import time
from sklearn import datasets,cluster,metrics

#拟合结果绘图
def scatter_plot(X,y,pred,n,C=None,outlier=False,edgecolors='black',alpha=1.0,
                 title=''):
    colors=['red','blue','green','orange']
    plt.figure(figsize=(10,4))
    #基准结果
    plt.subplot(121)
    plt.title('target',fontsize=14)
    for i in range(n):
        plt.scatter(X[:,0][y==i],X[:,1][y==i],c=colors[i],alpha=alpha,
                    edgecolors=edgecolors,label='y=%d'%i)
    #plt.legend()
    #预测结果
    plt.subplot(122)
    plt.title('predict',fontsize=14)
    for i in range(n):
        plt.scatter(X[:,0][pred==i],X[:,1][pred==i],c=colors[i],alpha=alpha,
                    edgecolors=edgecolors,label='p=%d'%i)
    #簇中心
    if type(C)!=type(None):
        plt.scatter(C[:,0],C[:,1],s=70,c='yellow',marker='v',alpha=alpha,
                    edgecolors=edgecolors,label='centers')
    if outlier==True:
        plt.scatter(X[pred==-1,0],X[pred==-1,1],c='yellow',alpha=alpha,
                edgecolors=edgecolors,label='outlier')
    #plt.legend()
    plt.suptitle(title,fontsize=16,y=1.02)
    plt.show()
    
#简单数据集
#第一个是线性可分的，第二第三个线性不可分
cn=3
X,y=datasets.make_blobs(n_samples=1000,n_features=2,centers=cn)

cn=2
X,y=datasets.make_circles(n_samples=1000,noise=0.08,factor=0.5)

cn=2
X,y=datasets.make_moons(n_samples=1000,noise=0.1)

#k-means
km0=ct.KMeans(clusters_n=cn,iter_max=100)
km0.fit(X,kmpp=True)
pred0=km0.predict(X)

#外在方法评估
score0=km0.assess(pred0,y)
print('k-means train score: %f'%score0)
#内在方法评估(DBI,DVI,轮廓系数)
a0,s0=km0.assess_inner(pred0,X,s_detail=True)
  
#绘制拟合结果
scatter_plot(X,y,pred0,cn,km0.centers,title='[k-means]')

#DBSCAN
#由于是基于密度的，所以在类的分界不明显时将类区分开会变得很困难
eps,min_pts=1,8
eps,min_pts=0.12,4
eps,min_pts=0.15,10

ds0=ct.DBSCAN(eps=eps,min_pts=min_pts)
pred0=ds0.fit_predict(X,divide_outlier=False)

#分类过少需要提高密度要求，即减少eps或增加min_pts
#分类过多需要降低密度要求，即增加eps或减小min_pts
scatter_plot(X,y,pred0,cn,outlier=True,title='[dbscan]')

#外在方法评估
score0=ds0.assess(pred0,y)
print('dbscan train score: %f'%score0)

#凝聚式层次聚类
#注意：这里是标准实现，时间复杂度为O(n^3),空间复杂度为O(n^2)，
#      时间和空间开销都很大，只能在很小的数据集上运行
ag0=ct.Agglomerative(clusters_n=3)
pred0=ag0.fit_predict(X)
#g_h=ag0.fit_predict(X,return_all=True)
    
scatter_plot(X,y,pred0,cn,title='[agglomerative]')

#与sklearn对照

#k-means
sk_km0=cluster.KMeans(n_clusters=cn)
start=time.clock()
sk_km0.fit(X)
print('time used for training: %f'%(time.clock()-start))
sk_pred0=sk_km0.predict(X)
#轮廓系数
start=time.clock()
sk_s0=metrics.silhouette_score(X,sk_pred0,metric='euclidean')
print('time used for assessing: %f'%(time.clock()-start))
#绘制拟合结果
scatter_plot(X,y,sk_pred0,cn,sk_km0.cluster_centers_,title='[k-means(sklearn)]')

#dbscan
sk_ds0=cluster.DBSCAN(eps=eps,min_samples=min_pts)
start=time.clock()
sk_pred0=sk_ds0.fit_predict(X)
print('time used for predicting: %f'%(time.clock()-start))
#绘制拟合结果
scatter_plot(X,y,sk_pred0,cn,outlier=True,title='[dbscan(sklearn)]')

#小麦种子数据集
f = open('D:\\training_data\\used\\wheat_seed.txt')
buf = pd.read_table(f,header=None,delim_whitespace=True)
buf.columns=['区域','周长','压实度','籽粒长度','籽粒宽度','不对称系数','籽粒腹沟长度','类']
describe=buf.describe()

#分割训练集和测试集
X1,y1,test_X1,test_y1=dp.split_train_test(buf)

ref1=dp.scaler_reference(X1)
X1_=dp.minmax_scaler(X1,ref1)
test_X1_=dp.minmax_scaler(test_X1,ref1)

#训练模型
km1=ct.KMeans(clusters_n=3,iter_max=100)
km1.fit(X1.values)
pred1=km1.predict(test_X1.values)

#外在方法评估
score1=km1.assess(pred1,test_y1.values)
print('k-means train score: %f'%score1)
#内在方法评估
a1,s1=km1.assess_inner(pred1,test_X1.values,s_detail=True)
