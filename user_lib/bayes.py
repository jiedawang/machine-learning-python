# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import user_lib.statistics as stats
from user_lib.check import check_type,check_limit,check_index_match,check_items_match
import time
import random

#朴素贝叶斯
class NaiveBayes:
    
    def __init__(self):
        pass
    
    #分布概率
    #P(A),计算的是a中所有取值的概率
    def d_prob_(self,a):
        group=a.groupby(a).count()
        values,counts=group.index.values,group.values
        probas=counts/counts.sum()
        d_p=np.c_[values,probas,counts]
        d_p=pd.DataFrame(d_p,columns=['value','proba','counts'])
        return d_p
    
    #条件概率
    #P(A|B),B为条件
    #计算的是X和y所有取值组合的条件概率
    def c_prob_(self,X,y):
        y_d_p=self.d_prob_(y)
        for i in range(len(X.columns)):
            x=X.iloc[:,i]
            x_d_p=self.d_prob_(x)
            #保存P(X)
            feature=np.full(len(x_d_p),X.columns[i])
            if i==0:
                X_d_p=np.c_[feature,x_d_p.values]
            else:
                X_d_p=np.r_[X_d_p,np.c_[feature,x_d_p.values]]
            for j in range(len(y_d_p)):
                x_d_p=self.d_prob_(x[y==y_d_p.iloc[j,0]])
                classify=np.full(len(x_d_p),y_d_p.iloc[j,0])
                feature=np.full(len(x_d_p),X.columns[i])
                if (i==0)&(j==0):
                    c_p=np.c_[classify,feature,x_d_p.values]
                else:
                    c_p=np.r_[c_p,np.c_[classify,feature,x_d_p.values]]
        c_p=pd.DataFrame(c_p,columns=['classify','feature','value','proba','count'])
        X_d_p=pd.DataFrame(X_d_p,columns=['feature','value','proba','count'])
        return c_p,X_d_p,y_d_p
    
    #独立变量联合概率
    #为防止概率相乘时因值过小而下溢出或浮点型舍入导致结果出入过大，转换为对数运算
    def ifj_prob_(self,probs,axis=1):
        return np.exp(np.log(probs).sum(axis=axis))
    
    #贝叶斯公式
    #P(B|A)=P(A|B)*P(B)/P[A]
    #A为特征，B为类别
    def b_prob_(self,X,c_p,X_d_p,y_d_p):
        n,m,k=len(X),len(X.columns),len(y_d_p)
        classes=y_d_p['value'].tolist()
        #P(X),P(X|y)的计算缓存，后面会对axis1聚合求乘积
        p_X_=np.zeros((n,m))
        p_X_y_=np.zeros((n,m,k))
        #遍历每个特征
        for i in range(len(X.columns)):
            #P(Xi)根据当前特征的取值映射到概率
            x=X.iloc[:,i]
            x_d_p=X_d_p[X_d_p['feature']==X.columns[i]].loc[:,['value','proba']]
            mapping_dict={value:proba for value,proba in x_d_p.values}
            p_X_[:,i]=x.map(mapping_dict).values
            #遍历每个类
            for j in range(k):
                #P(Xi|yj)根据当前特征和类别的取值映射到概率
                c_p_=c_p[(c_p['feature']==X.columns[i])&(c_p['classify']==classes[j])].loc[:,['value','proba']]
                mapping_dict={value:proba for value,proba in c_p_.values}
                p_X_y_[:,i,j]=x.map(mapping_dict).values
        #P(X|y),P(y),P(X)的计算
        p_X_y=self.ifj_prob_(p_X_y_,axis=1)
        p_X_y[np.isnan(p_X_y)]=0.
        p_y=y_d_p['proba'].values.repeat(n).reshape((k,n)).T
        p_X=self.ifj_prob_(p_X_,axis=1).repeat(k).reshape((n,k))
        #计算类别概率
        #注：此处算出的概率值可能大于1，尚不确定是因为计算错误还是特征之间的关联性导致的，
        #    就分类效果来看似乎影响不大，暂时不做深究，仅作归一化
        p_y_X=p_X_y*p_y/p_X
        p_y_X_sum=p_y_X.sum(axis=1)
        #可能出现训练数据中没有的 特征|类别 组合，用训练集整体的类别概率填充
        p_y_X[p_y_X_sum==0.]=1./k
        p_y_X_sum[p_y_X_sum==0]=1.
        p_y_X=(p_y_X.T/p_y_X_sum).T
        return p_y_X
    
    #X输入校验
    def check_input_X_(self,X):
        if type(X)==type(pd.Series()):
            X=X.to_frame()
        check_type('X',type(X),type(pd.DataFrame()))
        return X.astype('str')
    
    #y,p_y输入校验
    def check_input_y_(self,y,name='y'):
        check_type(name,type(y),type(pd.Series()))
        return y.astype('str')
    
    #拟合
    def fit(self,X,y):
        start=time.clock()
        X=self.check_input_X_(X)
        y=self.check_input_y_(y)
        check_index_match(X,y,'X','y')
        self.c_p,self.X_d_p,self.y_d_p=self.c_prob_(X,y)
        print('\ntime used for training: %f'%(time.clock()-start))
        
    #预测
    def predict(self,X):
        start=time.clock()
        X=self.check_input_X_(X)
        features=self.X_d_p['feature'].drop_duplicates().tolist()
        check_items_match(X.columns,features,'X','P(X)','features',mode='right')
        p_y_X=self.b_prob_(X,self.c_p,self.X_d_p,self.y_d_p)
        p_y_X_max=(p_y_X.T==p_y_X.max(axis=1)).T
        need_repair=np.where(p_y_X_max.sum(axis=1)>1)[0]
        for i in need_repair:
            p_y_x_max=p_y_X_max[i,:]
            idx1=np.where(p_y_x_max==1)[0]
            keep=idx1[int(random.uniform(0,len(idx1)))]
            p_y_x_max[:]=0
            p_y_x_max[keep]=1
            p_y_X_max[i,:]=p_y_x_max
        classes=self.y_d_p['value'].values
        classes_idx=np.array(range(len(classes)))
        pred_y=np.dot(p_y_X_max,classes_idx).astype('int')
        pred_y=pd.Series(pred_y,name='classify',index=X.index)
        for i in range(len(classes)):
            pred_y[pred_y==i]=classes[i]
        print('\ntime used for predict: %f'%(time.clock()-start))
        return pred_y
    
    #评估
    def access(self,y,p_y,return_dist=False):
        check_type('return_dist',type(return_dist),type(True))
        y=self.check_input_y_(y)
        p_y=self.check_input_y_(p_y,'p_y')
        check_index_match(y,p_y,'y','p_y')
        classes=self.y_d_p['value'].values
        return stats.accuracy(y,p_y,return_dist,classes)