# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import user_lib.svm as svm
import matplotlib.pyplot as plt
import user_lib.data_prep as dp

#绘制散点图和等高线图，仅限下面两个简单数据集可用
def q_plot(X,y,svm,broaden=0.1):
    #X值域
    x1_min,x1_max=X['x1'].min(),X['x1'].max()
    x2_min,x2_max=X['x2'].min(),X['x2'].max()
    #生成网格数据，用于绘制等高线图
    X0=pd.DataFrame()
    X0_1,X0_2=np.mgrid[
            x1_min-broaden:x1_max+broaden:101j,
            x2_min-broaden:x2_max+broaden:101j
            ]
    X0['x1']=X0_1.reshape(101*101)
    X0['x2']=X0_2.reshape(101*101)
    y0=svm.predict(X0,return_u=True)
    y0=y0.iloc[:,0].values.reshape(101,101)
    #设置图像大小
    plt.figure(figsize=(6,6),dpi=80)
    #数据集散点图
    plt.scatter(X.iloc[:,0][y==1],X.iloc[:,1][y==1],c='b',edgecolors='black')
    plt.scatter(X.iloc[:,0][y==-1],X.iloc[:,1][y==-1],c='r',edgecolors='black')
    #模型预测等高线图
    plt.contour(X0_1[:,0],X0_2[0,:],y0,[-1,0,1],colors='black',
                linestyles=['dashed','solid','dashed'])
    #支持向量点
    plt.scatter(svm.sv_X[0][:,0][svm.sv_y[0]==1],svm.sv_X[0][:,1][svm.sv_y[0]==1],
                c='b',edgecolors='yellow',linewidths=1)
    plt.scatter(svm.sv_X[0][:,0][svm.sv_y[0]==-1],svm.sv_X[0][:,1][svm.sv_y[0]==-1],
                c='r',edgecolors='yellow',linewidths=1)
    #其他设置
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.xlim((x1_min-2*broaden,x1_max+2*broaden))
    plt.ylim((x2_min-2*broaden,x2_max+2*broaden))
    plt.legend(labels=['y=1', 'y=-1'],loc='best')
    plt.show()

#线性可分简单数据集
f = open('D:\\training_data\\used\\simple_data1.txt')
buf = pd.read_table(f,header=None,sep=',')
buf.columns=['x1','x2','y']
describe=buf.describe()

X1=buf.iloc[:,:2].copy()
y1=buf.iloc[:,2].copy()
y1[y1==0]=-1
ref1=dp.scaler_reference(X1)
X1_=dp.minmax_scaler(X1,ref1)

svm1=svm.SupportVectorMachine(mode='c',iter_max=20,k_type='lin',C=100.)
#w,b=svm1.qp_optimize_(X1_,y1,C=10)
svm1.fit(X1_,y1,show_time=True)

p_y1=svm1.predict(X1_,show_time=True)
score1=svm1.assess(y1,p_y1)

print('\nscore:%f'%score1)
q_plot(X1_,y1,svm1)

#非线性可分简单数据集
f = open('D:\\training_data\\used\\simple_data2.txt')
buf = pd.read_table(f,header=None,sep=',')
buf.columns=['x1','x2','y']
describe=buf.describe()

X2=buf.iloc[:,:2].copy()
y2=buf.iloc[:,2].copy()
#X2_=dp.minmax_scaler(X2)
y2[y2==0]=-1

svm2=svm.SupportVectorMachine(mode='c',iter_max=20,k_type='rbf',
                              k_args={'sigma':1.0},C=100.)
#w,b=svm1.qp_optimize_(X1_,y1,C=10)
svm2.fit(X2,y2,show_time=True)

p_y2=svm2.predict(X2,show_time=True)
score2=svm2.assess(y2,p_y2)

print('\nscore:%f'%score2)
q_plot(X2,y2,svm2)

#二分类
#钞票数据集：类1为假钞，0为真钞
f = open('D:\\training_data\\used\\bill.txt')
buf = pd.read_table(f,header=None,sep=',')
buf.columns=['小波变换图像','小波偏斜变换图像','小波峰度变换图像','图像熵','类']
describe=buf.describe()

X3,y3,test_X3,test_y3=dp.split_train_test(buf)

ref3=dp.scaler_reference(X3)
X3_=dp.minmax_scaler(X3,ref3)
test_X3_=dp.minmax_scaler(test_X3,ref3)

svm3=svm.SupportVectorMachine(mode='c',iter_max=20,k_type='lin',C=100.)
svm3.fit(X3_,y3,show_time=True)
cost_h3=svm3.cost_h[0]
optimize_h3=svm3.optimize_h[0]

cost_h3.plot()
plt.xlabel('iter')
plt.ylabel('cost')
plt.show()

result3=svm3.predict(X3_)
score3=svm3.assess(y3,result3)
print('\nuser train score:%f'%score3)
result3=svm3.predict(test_X3_)
score3=svm3.assess(test_y3,result3)
print('\nuser test score:%f'%score3)

#多分类
#one vs rest/one vs one 两种模式测试
#小麦种子数据集
f = open('D:\\training_data\\used\\wheat_seed.txt')
buf = pd.read_table(f,header=None,delim_whitespace=True)
buf.columns=['区域','周长','压实度','籽粒长度','籽粒宽度','不对称系数','籽粒腹沟长度','类']
describe=buf.describe()

X4,y4,test_X4,test_y4=dp.split_train_test(buf)

ref4=dp.scaler_reference(X4)
X4_=dp.minmax_scaler(X4,ref4)
test_X4_=dp.minmax_scaler(test_X4,ref4)

iter_max=20
#逻辑回归中多分类模式已经实现过ovo了，所以这里尝试了tree
multi_class_list=['ovr','tree']

for multi_class in multi_class_list:
    print('\n[Classifier]')
    print('<multi_class: '+multi_class+'>')
    svm4=svm.SupportVectorMachine(mode='c',iter_max=iter_max,multi_class=multi_class,
                                  k_type='lin',C=100.)
    svm4.fit(X4_,y4,show_time=True)
    
    result4=svm4.predict(X4_)
    score4=svm4.assess(y4,result4)
    print('\nuser train score:%f'%score4)
    result4=svm4.predict(test_X4_)
    score4,dist4=svm4.assess(test_y4,result4,return_dist=True)
    print('\nuser test score:%f'%score4)
    print('\nclasses:')
    print(svm4.classes)
    print('\npredict distribution:')
    print(dist4)
 
#回归
#波士顿房价数据集
f = open('D:\\training_data\\used\\boston_house_price.txt')
buf = pd.read_table(f,header=None,delim_whitespace=True)
buf.columns=['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD',
                     'TAX','PTRATIO','B','LSTAT','MEDV']
describe=buf.describe()

#拆分训练集和测试集
X5,y5,test_X5,test_y5=dp.split_train_test(buf)

#特征缩放
ref=dp.scaler_reference(X5)
X5_=dp.minmax_scaler(X5,ref)
test_X5_=dp.minmax_scaler(test_X5,ref)

print('\n[Regressor]')

svm5=svm.SupportVectorMachine(mode='r',iter_max=20,k_type='lin',C=10.,eps=0.5)
svm5.fit(X5_,y5,show_time=True)
cost_h5=svm5.cost_h[0]
optimize_h5=svm5.optimize_h[0]

cost_h5.plot()
plt.xlabel('iter')
plt.ylabel('cost')
plt.show()

result5=svm5.predict(X5_)
score5=svm5.assess(y5,result5)
print('\nuser train score:%f'%score5)
result5=svm5.predict(test_X5_)
score5=svm5.assess(test_y5,result5)
print('\nuser test score:%f'%score5)

#简单数据集可视化
sp_dt=[(1.5,5),(2.4,9),(3.1,30),(4.8,110),(5.2,70),(6.7,220),(8,500)]
sp_dt=pd.DataFrame(sp_dt,columns=['x','y'])

x6=sp_dt['x']
y6=sp_dt['y']

#用于描绘结果曲线的点集
x6_0=pd.Series(np.linspace(x6.min(),x6.max(),101),name='x')

#拟合并绘图
eps=20.
svm6=svm.SupportVectorMachine(mode='r',iter_max=20,
                              k_type='pol',k_args={'R':1.,'d':4},
                              C=100.,eps=eps)
svm6.fit(x6,y6,keep_nonsv=True)
y6_0=svm6.predict(x6_0)
p_y6=svm6.predict(x6)
score6=svm6.assess(y6,p_y6)
print('score:%f'%score6)
plt.scatter(x6,y6,c='b')
#支持向量点
sv_idx=(svm6.a[0]!=0)
plt.scatter(svm6.sv_X[0][sv_idx,0],y6.values[sv_idx],
            c='b',edgecolors='yellow',linewidths=1)
plt.plot(x6_0,y6_0+eps,c='r',linestyle='--')
plt.plot(x6_0,y6_0,c='r')
plt.plot(x6_0,y6_0-eps,c='r',linestyle='--')
plt.xlabel('x')
plt.ylabel('y')
plt.show()