# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import user_lib.regression as rg
import user_lib.data_prep as dp
import time
import matplotlib.pyplot as plt

#从txt文件读取数据集（波士顿房价）
'''
每个类的观察值数量是均等的，共有 506 个观察，13 个输入变量和1个输出变量。变量名如下：
CRIM：城镇人均犯罪率。
ZN：住宅用地超过 25000 sq.ft. 的比例。
INDUS：城镇非零售商用土地的比例。
CHAS：查理斯河空变量（如果边界是河流，则为1；否则为0）。
NOX：一氧化氮浓度。
RM：住宅平均房间数。
AGE：1940 年之前建成的自用房屋比例。
DIS：到波士顿五个中心区域的加权距离。
RAD：辐射性公路的接近指数。
TAX：每 10000 美元的全值财产税率。
PTRATIO：城镇师生比例。
B：1000（Bk-0.63）^ 2，其中 Bk 指代城镇中黑人的比例。
LSTAT：人口中地位低下者的比例。
MEDV：自住房的平均房价，以千美元计。
'''
f = open('D:\\training_data\\used\\boston_house_price.txt')
buf = pd.read_table(f,header=None,delim_whitespace=True)
buf.columns=['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD',
                     'TAX','PTRATIO','B','LSTAT','MEDV']
describe=buf.describe()

#拆分训练集和测试集
X,y,test_X,test_y=dp.split_train_test(buf)

#特征缩放
ref=dp.scaler_reference(X)
X_=dp.minmax_scaler(X,ref)
test_X_=dp.minmax_scaler(test_X,ref)

#多元线性回归
#分别使用两种方法拟合模型

print('\n\n<拟合时间对比>')
#1：正规方程法
print('\n正规方程法：')
linear1=rg.LinearRegression(fit_mode='ne')
theta_by_ne=linear1.fit(X_,y,show_time=True,output=True)

#2：梯度下降法
print('\n梯度下降法：')
#最大迭代次数,学习率
linear2=rg.LinearRegression(fit_mode='sgd',learning_rate=0.3,iter_max=1000)
#拟合
theta_by_gd=linear2.fit(X_,y,show_time=True,output=True)
theta_h=linear2.theta_h
cost_h=linear2.cost_h

#cost,theta变化曲线
linear2.plot_change_h()

#预测
result1=linear1.predict(test_X_)
score1=linear1.assess(test_y,result1)
result2=linear2.predict(test_X_)
score2=linear2.assess(test_y,result2)

print('\n\n<拟合结果对比>')
print('\n正规方程法：')
print('train score：%f'%linear1.score)
print('test score：%f'%score1)
print('\n梯度下降法：')
print('train score：%f'%linear2.score)
print('test score：%f'%score2)

#与sklearn对照
from sklearn.linear_model import LinearRegression
#建模
lrModel = LinearRegression()
#训练模型
lrModel.fit(X, y)
#评分
train_score3=lrModel.score(X, y)
#查看参数
lrModel.coef_
#查看截距
lrModel.intercept_
#预测
result3=lrModel.predict(test_X)
test_score3=lrModel.score(test_X, test_y)
print('\n与sklearn对比')
print('train score：%f'%train_score3)
print('test score：%f'%test_score3)

#多项式回归和L2正则化测试
sp_dt=[(1.5,5),(2.4,9),(3.1,30),(4.8,110),(5.2,70),(6.7,220),(8,500)]
sp_dt=pd.DataFrame(sp_dt,columns=['x','y'])

x2=sp_dt['x']
y2=sp_dt['y']

#用于描绘结果曲线的点集
x2_0=pd.Series(np.linspace(x2.min(),x2.max(),101),name='x')

#拟合并绘图
def fit_test(x,y,x0,h,L2_n=0.0):
    lr_temp=rg.LinearRegression(L2_n=L2_n)
    x_h=dp.feature_mapping(x,h)
    lr_temp.fit(x_h,y)
    x0_h=dp.feature_mapping(x0,h)
    y0=lr_temp.predict(x0_h)
    print('\n\n<h=%d,L2_n=%f>'%(h,L2_n))
    print('train score:%f'%(lr_temp.score))
    plt.scatter(x,y,c='b')
    plt.plot(x0,y0,c='r')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

#求线性解(欠拟合)
fit_test(x2,y2,x2_0,h=1)

#映射为多项式
#正常拟合
fit_test(x2,y2,x2_0,h=3)

#过拟合
fit_test(x2,y2,x2_0,h=6)

#使用L2正则化避免过拟合
fit_test(x2,y2,x2_0,h=6,L2_n=1.0)








