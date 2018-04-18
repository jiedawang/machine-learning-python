# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import user_lib.mssql as mssql
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import user_lib.logistic as lg

host='ntgprodbi08'
user='tableau001'
db='Maybelline_TAB'
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
        SELECT * FROM Maybelline_TAB..TMP_LOGISTIC_TEST_DATA
        ''')
df=pd.DataFrame(resList,columns=defList)

df=df[(df['R6_TRANS']<30)&(df['R7T12_TRANS']<30)]

x=df[['R6_TRANS','R7T12_TRANS']]
y=df['PUR_FLAG']

logistic=lg.logistic()

theta=logistic.fit_ln_bd(x,y,100,0.00005,0.01)

cost_h=logistic.cost_h
theta_h=logistic.theta_h

#cost_h.plot()
#theta_h.plot()

p=logistic.predict(theta,x)

'''
X=df.iloc[:,0:2]
y=df.iloc[:,2]

y_t=(y==1)
y_f=(y==0)

plt.scatter(X[y_t].iloc[:,0],X[y_t].iloc[:,1],marker='o',c='b')
plt.scatter(X[y_f].iloc[:,0],X[y_f].iloc[:,1],marker='x',c='r')

plt.xlabel('Feature1/R7T12 Trans')  
plt.ylabel('Feature2/R6 Trans')  
plt.legend(['return', 'no pur'])  
plt.show()
'''
