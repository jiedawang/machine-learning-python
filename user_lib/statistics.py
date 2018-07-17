# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

#计算两个向量的相关系数
#注：pandas有提供相关系数的计算，此处只是尝试一下自己实现
def corrf(a,b):
    '''
    a: 向量，Series或narray(m,1)类型
    b: 向量，Series或narray(m,1)类型
    return
    0: 相关系数向量，narray(m,1)类型
    '''
    if len(a)!=len(b):
        raise TypeError('a,b should have the same length')
    da=a-a.mean()
    db=b-b.mean()
    n=len(a)
    Da=np.dot(da.T,da)/n
    Db=np.dot(db.T,db)/n
    Covab=np.dot(da.T,db)/n
    return Covab/(np.sqrt(Da)*np.sqrt(Db))

#计算相关系数向量/矩阵
def corr(X,y=None):
    '''
    X: 首要输入，DataFrame或narray(m,n)类型
    y: 次要输入，Series或narray(m,1)类型，默认值None，
       None时计算X列与列之间的相关系数矩阵，否则计算X各列与y的相关系数向量
    return
    0: 相关系数向量/矩阵，DataFrame或Series类型
    '''
    if type(y)==type(None):
        k=len(X.columns)
        correl=np.empty((k,k),dtype=float)
        for ia,ca in enumerate(X.columns):
            for ib,cb in enumerate(X.columns):
                correl[ia][ib]=corrf(X[ca],X[cb])
        return pd.DataFrame(correl,index=X.columns,columns=X.columns)
    else:
        k = len(X.columns)
        correl=np.empty(k, dtype=float)
        for ia,ca in enumerate(X.columns):
            correl[ia]=corrf(X[ca],y)
        return pd.Series(correl,index=X.columns,name='y')
 
#回归模型评估：R方
#范围0~1，越大拟合结果越好
def r_sqr(y,p_y):
    '''
    y: 观测值，Series或narray(m,1)类型
    p_y: 预测值，Series或narray(m,1)类型
    '''
    #总平方和
    buf1=y-y.mean()
    SST=np.dot(buf1.T,buf1)
    #SST=np.sum((y-y.mean())**2)
    #残差平方和
    buf2=p_y-y
    SSE=np.dot(buf2.T,buf2)
    #SSE=np.sum((p_y-y)**2)
    #回归平方和=总平方和-残差平方和
    #R方=回归平方和/总平方和
    return (SST-SSE)/SST
    
#回归模型评估：调整R方
#（消除样本数和参数个数带来的影响）
def adj_r_sqr(r2,n,k):
    '''
    r2: R方，float类型
    n: 样本容量，int类型
    k: theta参数个数，int类型
    '''
    if n-k==0:
        #print('adj_r_sqr error: n=k')
        return 0
    else:
        return 1-(1-r2)*(n-1)/(n-k)
              
'''
#（这个没搞明白什么意思）
#模型评估-回归显著性：F检验
#(用于判断回归模型是否能真实反映数据之间的关系)
#n:样本容量
#k:theta参数个数
def f_test(self,p_y,y,n,k):
    #回归平方和
    #（回归平方和+残差平方和=总平方和）
    SSR=np.sum((p_y-y.mean())**2)
    #残差平方和
    SSE=np.sum((p_y-y)**2)
    #F统计量
    return (SSR/(k-1))/(SSE/(n-k))
''' 

#分类模型评估:准确率
def accuracy(y,p_y,return_dist=False,classes=None):
    '''
    y: 观测值，Series或narray(m,1)类型
    p_y: 预测值，Series或narray(m,1)类型
    return_dist: 是否返回预测分布(也称作混淆矩阵)，bool类型，默认False
    classes: 类标签，list(str)类型，None是从输入中提取，默认为None
    '''
    cp=pd.DataFrame()
    cp['y'],cp['p']=y,p_y
    a=len(cp[cp['y']==cp['p']])*1.0/len(y)
    if return_dist==False:
        return a
    else:
        if classes==None:
            y_values=y.sort_values().drop_duplicates().tolist()
            p_values=p_y.sort_values().drop_duplicates().tolist()
            values=list(set(y_values+p_values))
        else:
            values=classes
        pred_dist=np.zeros((len(values),len(values)))
        for i in range(len(values)):
            for j in range(len(values)):
                bool_index=(cp['y']==values[j])&(cp['p']==values[i])
                pred_dist[i][j]=len(cp[bool_index])*1.0/len(y)
        pred_dist=pd.DataFrame(pred_dist,
                               columns='y_'+pd.Series(values).astype('str'),
                               index='p_'+pd.Series(values).astype('str'))
        return a,pred_dist
    
#对max函数的光滑近似函数
#该函数的梯度为softmax函数
#界限：max(x)<=LSE(x)<max(x)+log(len(x))
#当除了一个参数之外的所有参数接近负无穷大时，满足下限，
#并且当所有参数相等时满足上限
def logsumexp(fx,axis=1):
    return np.log(np.sum(np.e**fx,axis=axis))

#归一化指数函数
def softmax(fx,axis=1):
    exp=np.e**fx
    if axis==1:
        return (exp.T/exp.sum(axis=1)).T
    elif axis==0:
        return exp/exp.sum(axis=0)
    else:
        raise ValueError('support axis for 0 or 1')
    