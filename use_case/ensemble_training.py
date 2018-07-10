# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import user_lib.data_prep as dp
import user_lib.ensemble as es
import user_lib.deci_tree as dt
from datetime import datetime
from sklearn import tree
import time

#小麦种子数据集(离散输出)
f = open('D:\\training_data\\used\\wheat_seed.txt')
buf = pd.read_table(f,header=None,delim_whitespace=True)
buf.columns=['区域','周长','压实度','籽粒长度','籽粒宽度','不对称系数','籽粒腹沟长度','类']
describe=buf.describe()

#分割训练集和测试集
X1,y1,test_X1,test_y1=dp.split_train_test(buf)
X1=X1.round(4)

#cart分类树
#训练决策树
dtModel1=dt.DecisionTree(model_type='cart_c',features_use='all')
dtModel1.fit(X1,y1,show_time=True)
#dtModel1.plot()
#dtModel1.print_nodes()
#预测
result1=dtModel1.predict(X1)
score1=dtModel1.assess(y1,result1)
print('\nCART(Classifier) train score: %f'%score1)
result1=dtModel1.predict(test_X1)
score1=dtModel1.assess(test_y1,result1)
print('\nCART(Classifier) test score: %f'%score1)

#随机森林分类
#训练
rfModel2=es.RandomForest(units_n=10,units_type='cart_c')
results=rfModel2.fit(X1,y1,show_time=True)
#预测
result2=rfModel2.predict(X1)
score2=rfModel2.assess(y1,result2)
print('\nRandomForest(Classifier) train score: %f'%score2)
result2=rfModel2.predict(test_X1)
score2=rfModel2.assess(test_y1,result2)
print('\nRandomForest(Classifier) test score: %f'%score2)

#随机森林模型选择
s_units2=rfModel2.selection(test_X1,test_y1,return_units=True)
#预测
result2_2=rfModel2.predict(X1,units=s_units2)
score2_2=rfModel2.assess(y1,result2_2)
print('\nRandomForest(Classifier) after selection train score: %f'%score2_2)
result2_2=rfModel2.predict(test_X1,units=s_units2)
score2_2=rfModel2.assess(test_y1,result2_2)
print('\nRandomForest(Classifier) after selection test score: %f'%score2_2)
print('\nunits_n change after selection: %d -> %d'%
      (len(rfModel2.units),len(s_units2)))

#分类决策面可视化
#基本参数
classes=y1.sort_values().drop_duplicates().tolist()
n_classes=len(classes)
plot_colors="ryb"
plot_step=0.02

#获取所有特征的两两组合
column_idx=np.linspace(0,len(X1.columns)-1,len(X1.columns)).astype('int')
pair_enum=dp.combines_paired(column_idx)

for pairidx,pair in enumerate(pair_enum):
    #只挑一对演示
    #if pair!=[0,1]:
        #continue
    #每次只用两个特征训练
    X1_2=X1.iloc[:,pair]
    dtModel0=dt.DecisionTree(model_type='cart_c')
    dtModel0.fit(X1_2,y1)
    rfModel0=es.RandomForest(units_n=10,units_type='cart_c')
    rfModel0.fit(X1_2,y1)
    #dtModel4.plot()
    plt.figure(figsize=(10,5))
    #绘制决策树决策边界
    plt.rcParams['font.sans-serif']=['SimHei']
    plt.rcParams['axes.unicode_minus']=False
    x_min, x_max = X1_2.iloc[:, 0].min() - 1, X1_2.iloc[:, 0].max() + 1
    y_min, y_max = X1_2.iloc[:, 1].min() - 1, X1_2.iloc[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))
    xy_=pd.DataFrame(np.c_[xx.ravel(), yy.ravel()],columns=X1_2.columns)
    Z=dtModel0.predict(xy_)
    Z=Z.values.reshape(xx.shape)
    plt.subplot(1,2,1)
    ctf=plt.contourf(xx,yy,Z,cmap=plt.cm.RdYlBu) 
    #绘制训练集散点图
    for i,color in zip(range(n_classes),plot_colors):
        boolIdx=(y1==classes[i])
        plt.scatter(X1_2[boolIdx].iloc[:,0],X1_2[boolIdx].iloc[:,1],
                    edgecolors='black',c=color,
                    label=classes[i],cmap=plt.cm.RdYlBu)
    plt.xlabel(X1_2.columns[0])
    plt.ylabel(X1_2.columns[1])
    plt.legend()
    #绘制随机森林决策边界
    Z=rfModel0.predict(xy_)
    Z=Z.values.reshape(xx.shape)
    plt.subplot(1,2,2)
    ctf=plt.contourf(xx,yy,Z,cmap=plt.cm.RdYlBu) 
    #绘制训练集散点图
    for i,color in zip(range(n_classes),plot_colors):
        boolIdx=(y1==classes[i])
        plt.scatter(X1_2[boolIdx].iloc[:,0],X1_2[boolIdx].iloc[:,1],
                    edgecolors='black',c=color,
                    label=classes[i],cmap=plt.cm.RdYlBu)
    #其他配置
    plt.xlabel(X1_2.columns[0])
    plt.ylabel(X1_2.columns[1])
    plt.suptitle('''Decision surface of decition tree(left) and random forest(right) 
    using paired features %s'''%str(X1_2.columns.values))
    plt.legend()
    plt.show()
    

#波士顿房价数据(连续输出)
f = open('D:\\training_data\\used\\boston_house_price.txt')
buf = pd.read_table(f,header=None,delim_whitespace=True)
buf.columns=['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD',
                     'TAX','PTRATIO','B','LSTAT','MEDV']
#分割训练集和测试集
X2,y2,test_X2,test_y2=dp.split_train_test(buf)
X2=X2.round(4)

#cart回归树
#训练决策树
dtModel3=dt.DecisionTree(model_type='cart_r',features_use='all')
dtModel3.fit(X2,y2,show_time=True)
#dtModel3.plot()
#dtModel3.print_nodes()
#预测
result3=dtModel3.predict(X2)
score3=dtModel3.assess(y2,result3)
print('\nCART(Regressor) train score: %f'%score3)
result3=dtModel3.predict(test_X2)
score3=dtModel3.assess(test_y2,result3)
print('\nCART(Regressor) test score: %f'%score3)

#随机森林回归
#训练
rfModel4=es.RandomForest(units_n=10,units_type='cart_r')
rfModel4.fit(X2,y2,show_time=True)
#预测
result4=rfModel4.predict(X2)
score4=rfModel4.assess(y2,result4)
print('\nRandomForest(Regressor) train score: %f'%score4)
result4=rfModel4.predict(test_X2)
score4=rfModel4.assess(test_y2,result4)
print('\nRandomForest(Regressor) test score: %f'%score4)

#随机森林模型选择
s_units4=rfModel4.selection(test_X2,test_y2,return_units=True)
#预测
result4_2=rfModel4.predict(X2,units=s_units4)
score4_2=rfModel4.assess(y2,result4_2)
print('\nRandomForest(Regressor) after selection train score: %f'%score4_2)
result4_2=rfModel4.predict(test_X2,units=s_units4)
score4_2=rfModel4.assess(test_y2,result4_2)
print('\nRandomForest(Regressor) after selection test score: %f'%score4_2)
print('\nunits_n change after selection: %d -> %d'%
      (len(rfModel4.units),len(s_units4)))
