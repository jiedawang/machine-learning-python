# -*- coding: utf-8 -*-
import user_lib.mssql as mssql
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import mpl_toolkits.mplot3d
import user_lib.linear as li

host='ntgprodbi08'
user='tableau001'
db='Maybelline_PROD'
pwd='t32i*7)k&d'

print('''
[登录设置]
host : '''+host+''' 
db   : '''+db+''' 
user : '''+user  
)

st_time=dt.datetime.now()

ms = mssql.MSSQL_Manager(host=host, user=user, pwd=pwd, db=db)
#resList,defList=ms.GetTables()

resList,defList=ms.ExecQuery(
        sql='''
        SELECT SUM(LPK_AMOUNT) AMOUNT,
            SUM(CASE WHEN QTY_FLAG=1 THEN LPK_QTY END) QTY
        FROM Maybelline_TAB..MBL_SYS_PURCHASE
        WHERE PUR_DT>='2017-12-24' AND PUR_DT<'2017-12-27'
        GROUP BY PUR_ID
        ''')
df=pd.DataFrame(resList,columns=defList)
df['AMOUNT']=df['AMOUNT'].astype('float')

df=df[-df['QTY'].isna()]

y,x = df['AMOUNT'],df['QTY']

#线性回归(一元)
linear=li.linear()
print('\n[线性回归]')

#1：梯度下降法
#步长和最大迭代次数,代价变化的停止阀值
step_length=0.01
iter_max=100
stopping_threshold=0.001
#特征缩放
#x=linear.feature_scaling(x)
#y=linear.feature_scaling(y)
#计算结果
theta=linear.fit_by_gd(x,y,step_length,iter_max,stopping_threshold)
thetas_h=linear.thetas_h
costs_h=linear.costs_h
costs_change_rate_h=linear.costs_change_rate_h

#2：正规方程法
theta_by_ne=linear.fit_by_ne(x,y)

#图1：绘制梯度下降的路径
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

#plt.figure(1)

 #创建一个三维的绘图工程
ax=plt.subplot(111,projection='3d')
#描点，红色，不透明
ax.scatter(thetas_h[0],thetas_h[1],costs_h,c='r',alpha=1)
ax.set_xlabel('theta0')  
ax.set_ylabel('theta1')  
ax.set_zlabel('costs')   

#图1：绘制代价函数与theta的关系曲面
theta_max=thetas_h.max()
theta_min=thetas_h.min()
theta_st=thetas_h.iloc[0,:]
theta_ed=thetas_h.iloc[len(thetas_h)-1,:]
theta_range_extend=2*theta_ed-theta_st
theta_max[theta_range_extend>theta_max]=theta_range_extend[theta_range_extend>theta_max]
theta_min[theta_range_extend<theta_min]=theta_range_extend[theta_range_extend<theta_min]
#theta_max[theta_range_extend>theta_max]
#生成一个theta可能取值的数列
theta0,theta1=np.mgrid[
        theta_min[0]:theta_max[0]:20j,
        theta_min[1]:theta_max[1]:20j
        ]
rs=[]
for i in range(len(theta0)):
    rs_row=[]
    for j in range(len(theta0)): 
        fx=linear.model(linear.fill_x0(x),pd.Series([theta0[i][j],theta1[i][j]]))
        rs_row.append(linear.cost(fx,y))
        #print('theta0=%d,theta1=%d,cost=%d'%(theta0[i][j],theta1[i][j],cost))
    rs.append(rs_row)
#绘制空间曲面，透明度50%
ax.plot_surface(theta0,theta1,np.array(rs),rstride=2,cstride=1,
                cmap=plt.cm.coolwarm,alpha=0.5)
plt.show()

#图2：绘制costs，thetas变化曲线
#plt.figure(2)
linear.plot_costs_h()
linear.plot_thetas_h()

#图3：绘制拟合结果
#plt.figure(3)
#描点，蓝色
plt.scatter(x,y,c='b')
#最终模型随便取两点
x0=pd.Series([min(x),max(x)])
y0=linear.model(linear.fill_x0(x0),theta)
#绘制直线，红色
plt.plot(x0,y0,c='r')
plt.xlabel('QTY')
plt.ylabel('AMOUNT')
plt.show()
