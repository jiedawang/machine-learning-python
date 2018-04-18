# -*- coding: utf-8 -*-
import user_lib.mssql as mssql
import pandas as pd
import user_lib.iforest as iforest
import datetime as dt

host=''
user=''
db=''
pwd=''

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
        SELECT PUR_REFAMOUNT
        FROM BPURCHASE WITH(NOLOCK) 
        WHERE PUR_DT>='2017-12-10' AND PUR_DT<'2017-12-12'
        ''')
df=pd.DataFrame(resList,columns=defList)
df['PUR_REFAMOUNT']=df['PUR_REFAMOUNT'].astype('float')

if_mg=iforest.iforest_manager(
        sample_size=256,
        itree_num=100,
        height_limit=8
        )

if_mg.build(df)

#access=if_mg.data_assess(df)

ed_time=dt.datetime.now()
print((ed_time-st_time).seconds)
    


