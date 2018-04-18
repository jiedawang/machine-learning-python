# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import user_lib.excel as ex
import datetime as dt

#源数据
path='D:\JiedaWang\SSIS\TESTDATA\Test2.xlsx'
#excel=xlrd.open_workbook(path)
st_time=dt.datetime.now()

excel=ex.Excel_Manager(path)
df_cnt=excel.getSheetData(sheet_name='V_DM_COUNTER')
df_pur=excel.getSheetData(sheet_name='BPURCHASE')

ed_time=dt.datetime.now()
print((ed_time-st_time).seconds)

#关联
st_time=dt.datetime.now()

df_merge=pd.merge(df_cnt,df_pur,left_on='CNT_ID',right_on='PUR_CNT_ID')

ed_time=dt.datetime.now()
print((ed_time-st_time).seconds)

#聚合
result=df_merge.groupby(['MEMO04','CHANNEL'])\
.agg({
      'PUR_AMOUNT':np.sum , 
      'PUR_CST_ID':pd.Series.nunique,
      'PUR_ID':pd.Series.nunique
      })

result.rename(columns={
        'PUR_AMOUNT':'Sales',
        'PUR_CST_ID':'Mems',
        'PUR_ID':'Trans'
        },inplace=True)

#绘图
def plot_pareto(df,col):
    df.sort_values([col], ascending=[False],inplace=True)

    plt.rcParams['font.sans-serif']=['SimHei']
    plt.rcParams['axes.unicode_minus']=False
    
    plt.figure()
    df[col].plot(kind='bar')
    plt.ylabel(col)
    p=1.0*df[col].cumsum()/df[col].sum()
    p.plot(color='r',secondary_y=True,style='-o',linewidth=2)
    plt.ylabel(col+' cumsum rate')
    r_idx=list(p).index(np.min(p[p>=0.80]))
    plt.annotate(
            format(p[r_idx],'.4%'),
            xy=(r_idx,p[r_idx]),
            xytext=(r_idx*0.9,p[r_idx]*0.9),
            arrowprops=dict(facecolor='black', shrink=0.05)
            )
    plt.show()

