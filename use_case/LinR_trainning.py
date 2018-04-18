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
f = open('D:\\训练数据集\\used\\波士顿房价数据集\\data.txt')
buf = pd.read_table(f,header=None,delim_whitespace=True)
buf.columns=['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD',
                     'TAX','PTRATIO','B','LSTAT','MEDV']
describe=buf.describe()

dp_tool=dp.DataPreprocessing()

#拆分训练集和测试集
train=buf.sample(frac=0.8,random_state=1)
test=buf[~buf.index.isin(train.index)]

#多元线性回归
#创建模型
linear1=rg.LinearRegression()
linear2=rg.LinearRegression()

#拆分训练集的输入变量和输出变量
x=train.iloc[:,:13]
y=train.iloc[:,13]

#分别使用两种算法求解模型

#1：梯度下降法
#特征缩放
dp_tool.scaler_reference(x)
x=dp_tool.minmax_scaler(x)
#学习率a，最大迭代次数iter_max,代价变化停止阀值stop_threshold
#模型参数个数k,梯度下降参数配置init_dir,梯度下降类型gd_type
iter_max=1000
k=len(x.columns)+1
init_dir={'a':3e-1}
gd_type='SGD'
#计算结果
theta_by_gd=linear1.fit_by_gd(x,y,init_dir,iter_max,gd_type)
theta_h=linear1.theta_h
cost_h=linear1.cost_h

print('\n\n<cost下降过程>')
cost_h.plot()
plt.show()

#2：正规方程法
theta_by_ne=linear2.fit_by_ne(x,y)

#预测测试
test_x=test.iloc[:,:13]
test_y=test.iloc[:,13]
test_x=dp_tool.minmax_scaler(test_x)
test1_fx,test1_score,test1_a_result=linear1.predict_test(test_x,test_y)
test2_fx,test2_score,test2_a_result=linear2.predict_test(test_x,test_y)

print('\n\n<拟合结果对比>')
print('\n梯度下降法：')
print('train score：%f'%linear1.score)
print('test score：%f'%test1_score)
print('\n正规方程法：')
print('train score：%f'%linear2.score)
print('test score：%f'%test2_score)

#与sklearn对照
from sklearn.linear_model import LinearRegression
#建模
lrModel = LinearRegression()
#训练模型
lrModel.fit(x, y)
#评分
train3_score=lrModel.score(x, y)
#查看参数
lrModel.coef_
#查看截距
lrModel.intercept_
#预测
test3_fx=lrModel.predict(test_x)
test3_score=lrModel.score(test_x, test_y)
print('\n<与sklearn对比>')
print('train score：%f'%train3_score)
print('test score：%f'%test3_score)

#梯度下降优化算法测试
#(在线性回归中无法表现出相应的优势，仅作运行测试)
linear3=rg.LinearRegression()
#设置最大迭代次数,代价变化的停止阀值
iter_max=1000
k=len(x.columns)+1
gd_type_list=['SGD','Momentum','Nesterov',
              'Adagrad','RMSProp','Adadelta',
              'Adam','Adamax','Nadam']
init_dir_list=[{'a':3e-1},{'a':3e-1,'p':0.9,'k':k},{'a':3e-1,'p':0.9,'k':k},
               {'a':1,'e':1e-8,'k':k},{'a':3e-1,'p':0.9,'e':1e-8,'k':k},
               {'p':0.9,'e':1e-2,'k':k},
               {'a':3,'u':0.9,'v':0.999,'e':1e-8,'k':k},
               {'a':3,'u':0.9,'v':0.999,'e':1e-8,'k':k},
               {'a':3e-1,'u':0.9,'v':0.999,'e':1e-8,'k':k}]
cost_cont=pd.DataFrame(np.zeros([iter_max,len(gd_type_list)]),columns=gd_type_list)
iter_cont=[]

print('\n\n<梯度下降优化算法测试>')
#计算结果
for i in range(len(gd_type_list)):
    print('正在执行：%s'%gd_type_list[i])
    start = time.clock()
    linear3.fit_by_gd(x,y,init_dir_list[i],iter_max,gd_type_list[i],feedback=False)
    end = time.clock()
    cost_cont[gd_type_list[i]]=linear3.cost_h
    iter_cont.append((linear3.iter_num,end-start))
iter_cont=pd.DataFrame(iter_cont,index=gd_type_list,columns=['iter_num','time_cost'])
cost_cont[cost_cont.index<=100].plot()
plt.show()

#多项式回归和正则化测试
sp_dt=[(1.5,5),(2.4,9),(3.1,30),(4.8,110),(5.2,70),(6.7,220),(8,500)]
sp_dt=pd.DataFrame(sp_dt,columns=['x','y'])

x=sp_dt['x']
y=sp_dt['y']

#用于描绘结果曲线的点集
x0=pd.Series(np.linspace(x.min(),x.max(),101),name='x')

#拟合并绘图
def fit_test(x,y,x0,h,L2_n=0):
    lr_temp=rg.LinearRegression()
    dp_tool=dp.DataPreprocessing()
    x_h=dp_tool.feature_mapping(x,h)
    lr_temp.fit_by_ne(x_h,y,L2_n)
    x0_h=dp_tool.feature_mapping(x0,h)
    y0=lr_temp.predict(x0_h)
    print('\n\n<h=%d,L2_n=%f>'%(h,L2_n))
    print('train score:%f'%(lr_temp.score))
    plt.scatter(x,y,c='b')
    plt.plot(x0,y0,c='r')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

#求线性解(欠拟合)
fit_test(x,y,x0,h=1)

#映射为多项式
#正常拟合
fit_test(x,y,x0,h=3)

#过拟合
fit_test(x,y,x0,h=6)

#使用L2正则化避免过拟合
fit_test(x,y,x0,h=6,L2_n=1)








