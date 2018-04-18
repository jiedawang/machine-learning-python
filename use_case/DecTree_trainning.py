# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import user_lib.data_prep as dp
import user_lib.deci_tree as dt
from datetime import datetime
import time

#常规数据集
f = open('D:\\训练数据集\\used\\小麦种子数据集\\data.txt')
buf = pd.read_table(f,header=None,delim_whitespace=True)
buf.columns=['区域','周长','压实度','籽粒长度','籽粒宽度','不对称系数','籽粒腹沟长度','类']
describe=buf.describe()
#分割训练集和测试集
train=buf.sample(frac=0.8,random_state=1)
test=buf[~buf.index.isin(train.index)]

#根据第i列特征分割数据集
def split(data,i,continuous=False,value=0):
        #抽取第i列特征
        x=data.iloc[:,i]
        #连续特征和离散特征采用不同的处理方式
        if continuous==True:
            #根据分裂点将数据集拆分
            values=['<=%s'%str(value),'>%s'%str(value)]
            result=[data[x<=value],data[x>value]]
        else:
            #去重得到特征值列表
            values=x.drop_duplicates().sort_values().tolist()
            #根据不同的特征值进行分割
            result=[]
            for i in range(len(values)):
                result.append(data[x==values[i]])
        return result,values
    
    #信息熵,可以用于求类别的熵，也可以用于求特征的熵,只能计算单列
    #表示随机变量不确定性的度量，范围0~log2(n)，数值越大不确定性越大,n为离散值种类数
    #0log0=0 ；当对数的底为2时，熵的单位为bit；为e时，单位为nat。
def getEntropy(info,continuous=False,value=0):
        if continuous==True:
            #计算值的概率分布
            p_l=len(info[info<=value])/len(info)
            p_r=len(info[info>value])/len(info)
            #计算信息熵
            etp=-p_l*np.log2(p_l)-p_r*np.log2(p_r)
        else:
            #计算值的概率分布
            values_count=info.groupby(info).count()
            p=values_count/len(info)
            #计算信息熵
            etp=-np.sum(p*np.log2(p))
        return etp

        #条件熵
    #在x中第i个随机变量确定的情况下，随机变量y的不确定性
    #即按第i个特征划分数据后的信息熵
def getCondiEntropy(data,i,continuous=False,value=0):
        x=data.iloc[:,i]
        y=data.iloc[:,len(data.columns)-1]
        n=len(data)
        if continuous==True:
            boolIdx=(x<=value)
            p=len(boolIdx)/n
            con_ent=p*getEntropy(y[boolIdx])+(1-p)*getEntropy(y[~boolIdx])
        else:
            values=x.drop_duplicates().tolist()
            for i in range(len(values)):
                boolIdx=(x==values[i])
                p=len(boolIdx)/n
                con_ent+=p*getEntropy(y[boolIdx])
        return con_ent
    
    #最优分裂点选择
def chooseSplitValue(data,i):
        #计算分裂前的信息熵
        y_col=len(data.columns)-1
        baseEntropy=getEntropy(data.iloc[:,y_col])
        #排序
        sorted_series=data.iloc[:,i].drop_duplicates().sort_values().tolist()
        #初始化变量
        bestInfGain=0.0
        bestSplitValue=sorted_series[0]
        infGainList=[]
        #逐个计算所有可能分裂点的条件熵
        for j in range(len(sorted_series)-1):
            split_value=sorted_series[j]
            infGain=baseEntropy-getCondiEntropy(data,i,True,split_value)
            infGainList.append(infGain)
            if infGain>bestInfGain:
                bestInfGain=infGain
                bestSplitValue=split_value
        return bestSplitValue,len(sorted_series),sorted_series,infGainList
  
start = time.clock()
    
bestSplitValue,n,x,y=chooseSplitValue(train,0)

x.pop(len(x)-1)

elapsed = (time.clock() - start)
print("Time used:",elapsed)

#plt.scatter(x,y,c='b')

#plt.plot(x,y,c='r')

data=train[['区域','类']].sort_values('区域')
data.index=np.linspace(0,len(data)-1,len(data)).astype(np.int64)

subsequent=data.copy()
subsequent.index=subsequent.index-1
subsequent.columns=subsequent.columns+'_next'

data=data.join(subsequent,how='inner')

change_points=data[data['类']!=data['类_next']]

change_points_f=change_points.iloc[:,0:2]
change_points_b=change_points.iloc[:,2:]
change_points_b.columns=change_points_f.columns

check_points=pd.concat([change_points_f,change_points_b])

'''
#简单数据集
dataValues = [['youth', 'no', 'no', '1', 'refuse'],
           ['youth', 'no', 'no', '2', 'refuse'],
           ['youth', 'yes', 'no', '2', 'agree'],
           ['youth', 'yes', 'yes', '1', 'agree'],
           ['youth', 'no', 'no', '1', 'refuse'],
           ['mid', 'no', 'no', '1', 'refuse'],
           ['mid', 'no', 'no', '2', 'refuse'],
           ['mid', 'yes', 'yes', '2', 'agree'],
           ['mid', 'no', 'yes', '3', 'agree'],
           ['mid', 'no', 'yes', '3', 'agree'],
           ['elder', 'no', 'yes', '3', 'agree'],
           ['elder', 'no', 'yes', '2', 'agree'],
           ['elder', 'yes', 'no', '2', 'agree'],
           ['elder', 'yes', 'no', '3', 'agree'],
           ['elder', 'no', 'no', '1', 'refuse'],
           ]
labels = ['age', 'working?', 'house?', 'credit_situation','decition']
data=pd.DataFrame(dataValues,columns=labels)

dtModel=dt.DecisionTree()
deciTree0=dtModel.createTree(data,model_type='C4.5')

dtModel.createPlot(deciTree0)
dtModel.printNodes(deciTree0)

test0=data.drop('decition',axis=1)
result0=dtModel.predict(deciTree0,test0)
'''
'''
#常规数据集
f = open('D:\\训练数据集\\used\\小麦种子数据集\\data.txt')
buf = pd.read_table(f,header=None,delim_whitespace=True)
buf.columns=['区域','周长','压实度','籽粒长度','籽粒宽度','不对称系数','籽粒腹沟长度','类']
describe=buf.describe()
#分割训练集和测试集
train=buf.sample(frac=0.8,random_state=1)
test=buf[~buf.index.isin(train.index)]

#ID3(没有处理连续特征的能力，需要预处理)
train_cp1=train.copy()
test_cp1=test.copy()
#生成离散化区间
dp_tool=dp.DataPreprocessing()
rg=dp_tool.discret_reference(train,10)
#进行特征离散化
x=train.drop('类',axis=1)
x=dp_tool.discret(x,rg,return_label=True,open_bounds=True)
train_cp1.update(x)
#训练决策树
dtModel=dt.DecisionTree()
deciTree1=dtModel.createTree(train_cp1,model_type='ID3')
#dtModel.createPlot(deciTree1)
#dtModel.printNodes(deciTree1)
#存储和读取测试
dtModel.saveTree(deciTree1,'D:\\Model\\deciTree.txt')
deciTree1=dtModel.readTree('D:\\Model\\deciTree.txt')
#预测
test_x1=test_cp1.drop('类',axis=1)
test_x1=dp_tool.discret(test_x1,rg,return_label=True,open_bounds=True)
result1=dtModel.predict(deciTree1,test_x1)
score1=dtModel.assessment(test_cp1['类'],result1)
print('\nID3 test score: %f'%score1)
result1=dtModel.predict(deciTree1,test_x1,fill_empty=False)
score1=dtModel.assessment(test_cp1['类'],result1)
print('\nID3 test score(ignore empty): %f'%score1)
'''
'''
#C4.5
train_cp2=train.copy()
test_cp2=test.copy()
#训练决策树
dtModel=dt.DecisionTree()
deciTree2=dtModel.createTree(train_cp2,model_type='C4.5')
#dtModel.createPlot(deciTree2)
#dtModel.printNodes(deciTree2)
#预测
test_x2=test_cp2.drop('类',axis=1)
result2=dtModel.predict(deciTree2,test_x2)
score2=dtModel.assessment(test_cp2['类'],result2)
print('\nC4.5 test score: %f'%score2)
result2=dtModel.predict(deciTree2,test_x2,fill_empty=False)
score2=dtModel.assessment(test_cp2['类'],result2)
print('\nC4.5 test score(ignore empty): %f'%score2)
'''
'''
#与sklearn对照
from sklearn import tree

x=train.iloc[:,:len(train.columns)-1]
y=train.iloc[:,len(train.columns)-1]

#criterion:默认为"gini"
#支持的标准有"gini"代表的是Gini impurity(不纯度)
#与"entropy"代表的是information gain（信息增益）。
clf = tree.DecisionTreeClassifier()
clf = clf.fit(x, y)
print('\nsklearn train score:%f'%clf.score(x,y))

#保存模型
from sklearn.externals import joblib
joblib.dump(clf, 'D:\\Model\\deciTree2.pkl')
clf = joblib.load('D:\\Model\\deciTree2.pkl') 

test_x=test.iloc[:,:len(test.columns)-1]
test_y=test.iloc[:,len(test.columns)-1]
print('\nsklearn test score:%f'%clf.score(test_x,test_y))
result = clf.predict(x)
'''