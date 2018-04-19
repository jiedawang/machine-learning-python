# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import user_lib.data_prep as dp
import user_lib.deci_tree as dt
from datetime import datetime
import os, time, random
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