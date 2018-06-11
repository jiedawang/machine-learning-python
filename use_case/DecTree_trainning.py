# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import user_lib.data_prep as dp
import user_lib.deci_tree as dt
import user_lib.deci_tree_old as dto
from datetime import datetime
import time

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

X=data.iloc[:,:(len(data.columns)-1)]
y=data.iloc[:,len(data.columns)-1]

dtModel0=dt.DecisionTree()
deciTree0=dtModel0.fit(X,y,model_type='id3',output=True)

dtModel0.plot()
dtModel0.print_nodes()

test=data.drop('decition',axis=1)
result=dtModel0.predict(test)

#常规数据集
f = open('D:\\训练数据集\\used\\小麦种子数据集\\data.txt')
buf = pd.read_table(f,header=None,delim_whitespace=True)
buf.columns=['区域','周长','压实度','籽粒长度','籽粒宽度','不对称系数','籽粒腹沟长度','类']
describe=buf.describe()
dp_tool=dp.DataPreprocessing()
#分割训练集和测试集
X,y,test_X,test_y=dp_tool.split_train_test(buf)
X=X.round(4)
#生成离散化区间
rg=dp_tool.discret_reference(X,3)
#进行特征离散化
X_=dp_tool.discret(X,rg,str_label=True)
test_X_=dp_tool.discret(test_X,rg,str_label=True)

#ID3(没有处理连续特征的能力，需要预处理)
#训练决策树
dtModel1=dt.DecisionTree()
deciTree1=dtModel1.fit(X_,y,model_type='id3',output=True)
dtModel1.plot()
#dtModel1.plot(start_id=4,print_loc=True)
#dtModel1.print_nodes()
#预测
result1=dtModel1.predict(test_X_)
score1=dtModel1.assess(test_y,result1)
print('\nID3 test score: %f'%score1)

#存储和读取测试
dtModel1.save_tree('D:\\Model\\deciTree.txt')
dtModel1.read_tree('D:\\Model\\deciTree.txt')

#C4.5
#训练决策树
dtModel2=dt.DecisionTree()
deciTree2=dtModel2.fit(X,y,model_type='c4.5',output=True)
dtModel2.plot()
#dtModel2.print_nodes()
#预测
result2,paths=dtModel2.predict(test_X,return_paths=True)
score2=dtModel2.assess(test_y,result2)
print('\nC4.5 test score: %f'%score2)

#dtModel2.decition_path(X.iloc[0,:])

#CART分类
#连续特征
#训练决策树
dtModel3=dt.DecisionTree()
deciTree3=dtModel3.fit(X,y,model_type='cart',output=True)
dtModel3.plot()
#dtModel3.print_nodes()
#预测
result3=dtModel3.predict(test_X)
score3=dtModel3.assess(test_y,result3)
print('\nCART test score: %f'%score3)

#离散特征
#训练决策树
dtModel3_2=dt.DecisionTree()
deciTree3_2=dtModel3_2.fit(X_,y,model_type='cart',output=True)
dtModel3_2.plot()
#dtModel3_2.print_nodes()
#预测
result3_2=dtModel3_2.predict(test_X_)
score3_2=dtModel3_2.assess(test_y,result3_2)
print('\nCART test score: %f'%score3_2)

#dtModel3_2.decition_path(X_.iloc[0,:])

#与sklearn对照
from sklearn import tree
start = time.clock()
#criterion:默认为"gini"
#支持的标准有:
# "gini"代表的是Gini impurity(不纯度)
# "entropy"代表的是information gain（信息增益）。
sk_dtModel = tree.DecisionTreeClassifier()
sk_dtModel.fit(X, y)
end = time.clock()
print('\ntime used for trainning:%f'%(end-start))
print('\nsklearn train score:%f'%sk_dtModel.score(X,y))

#保存模型
from sklearn.externals import joblib
joblib.dump(sk_dtModel, 'D:\\Model\\deciTree2.pkl')
sk_dtModel = joblib.load('D:\\Model\\deciTree2.pkl') 

result = sk_dtModel.predict(test_X)
print('\nsklearn test score:%f'%sk_dtModel.score(test_X,test_y))