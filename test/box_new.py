# -*- coding: UTF-8 -*-

import user_lib.mssql as mssql
import pandas as pd
import matplotlib.pyplot as plt

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

#pwd=input("请输入密码：")

#源数据
ms = mssql.MSSQL_Manager(host=host, user=user, pwd=pwd, db=db)
#resList,defList=ms.GetTables()
resList,defList=ms.ExecQuery(
        sql='''
        SELECT CONVERT(DATE,PUR_DT) PUR_DT,SUM(LPK_AMOUNT) AMOUNT,
            COUNT(DISTINCT PUR_ID) TRANS
        FROM MBL_SYS_PURCHASE_DUMMY WITH(NOLOCK) 
        WHERE PUR_DT>='2017-10-01' AND PUR_DT<'2017-12-13'
        GROUP BY CONVERT(DATE,PUR_DT)
        ''')
df=pd.DataFrame(resList,columns=defList)

#绘制箱型图
def plot_box(df,col):
    df[col]=df[col].astype('int')
    #df['TRANS']=df['TRANS'].astype('int')
    #df['PUR_DT']=pd.to_datetime(df['PUR_DT'])
    #df = df.set_index('PUR_DT')
    
    plt.rcParams['font.sans-serif']=['SimHei']
    plt.rcParams['axes.unicode_minus']=False
    plt.figure()
    p = df[[col]].boxplot(return_type='dict')
    x = p['fliers'][0].get_xdata()
    y = p['fliers'][0].get_ydata()
    y.sort()

    for i in range(len(x)):
        if i>0:
            if abs(y[i]-y[i-1])/y.max()<0.033:
                plt.annotate(y[i],xy=(x[i],y[i]),xytext=(x[i]-0.12,y[i]))
            else:  
                plt.annotate(y[i],xy=(x[i],y[i]),xytext=(x[i]+0.03,y[i]))
        else:
            plt.annotate(y[i],xy=(x[i],y[i]),xytext=(x[i]+0.03,y[i])) 
            
    plt.show()

#统计量分析
def statistics(df):   
    statistics=df.describe()
    statistics.loc['range']=statistics.loc['max']-statistics.loc['min'] #极差
    statistics.loc['var']=statistics.loc['std']/statistics.loc['mean'] #变异系数
    statistics.loc['dis']=statistics.loc['75%']-statistics.loc['25%'] #四分位数间距
    
    print(statistics)