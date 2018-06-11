# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

#数据预处理
class DataPreprocessing:
    
    #拆分训练集和测试集
    def split_train_test(self,data,frac=0.8,random_state=None):
        train=data.sample(frac=frac,random_state=random_state)
        test=data[~data.index.isin(train.index)]
        train_X,train_y=train.iloc[:,:-1],train.iloc[:,-1]
        test_X,test_y=test.iloc[:,:-1],test.iloc[:,-1]
        return train_X,train_y,test_X,test_y
    
    #常数列补齐： x首列填充1,即常数位忽略x的影响
    def fill_x0(x_):
        #将Series转化为DataFrame
        x=x_.copy()
        if type(x)==type(pd.Series()):
            x=x.to_frame()
        #首位填充1，对应常量theta0
        if 'x0' not in x.columns.values:
            x.insert(0,'x0',np.ones(len(x)))
        return x
        
    #虚拟变量生成，可用于处理离散特征或多分类下的y
    def dummy_var(y):
        values=y.drop_duplicates().sort_values().tolist()
        result=pd.DataFrame()
        for i in range(len(values)):
            col_name=y.name+'_'+str(values[i])
            ovr=(y==values[i])
            result[col_name]=ovr.astype('int')
        return result
    
    #特征离散化(等距分)
    #计算参考标准
    def discret_reference(self,X,n,cols=None):
        if cols==None:
            cols=X.columns
        drange=[]
        for col in cols:
            x=X.loc[:,col]
            drange.append(np.linspace(x.min(),x.max(),n+1))
        drange=pd.DataFrame(drange,index=cols)
        return drange.T
    #根据参考标准离散化   
    def discret(self,X,drange,return_label=True,str_label=False,open_bounds=True):
        #按区间划分数据
        result=X.copy()
        for col in drange.columns:
            x=X.loc[:,col]
            rg=drange.loc[:,col].tolist()
            if return_label==True:
                if str_label==True:
                    result.loc[:,col]=pd.cut(x,bins=rg).astype('str')
                else:
                    result.loc[:,col]=pd.cut(x,bins=rg)
            else:
                result.loc[:,col]=pd.cut(x,bins=rg,labels=False)
            temp=result.loc[:,col]
            temp=temp[~temp.isnull()]
            if open_bounds==True:
                result.loc[x<=rg[0],col]=temp.min()
                result.loc[x>rg[len(rg)-1],col]=temp.max()
            else:
                result.loc[x==rg[0],col]=temp.min()
        return result
        
    
    #特征缩放
    #注：缩放后求得的theta会不一样，预测数据时也需要进行缩放 
    #scaler_reference用于设置缩放的参照标准，一般设为训练集的x
    def scaler_reference(self,x):
        self.min=x.min()
        self.max=x.max()
        self.mean=x.mean()
        k = len(x.columns)
        std=np.empty(k, dtype=float)
        for i,c in enumerate(x.columns):
            buf=x[c]-x[c].mean()
            std[i]=np.sqrt(np.dot(buf.T,buf)/len(buf))
        self.std=pd.Series(std,index=x.columns)
    
    #两种不同的缩放方式
    def minmax_scaler(self,x):
        if hasattr(self, 'min')==False:
            self.scaler_reference(x)
        return (x-self.min)/(self.max-self.min)  
    
    def standard_scaler(self,x):
        if hasattr(self, 'mean')==False:
            self.scaler_reference(x)
        return (x-self.mean)/(self.std)  
    
    #特征映射（多项式）
    #注：h是多项式的最高次数。配合正则化使用，不然容易出现过拟合
    #cross参数控制是否加入不同特征的组合项（目前只有两两组合）
    def feature_mapping(self,x,h,cross=False):
        if type(x)==type(pd.Series()):
            x=x.to_frame()
        if h<1:
            print('h should >=1')
            return x
        xh=x.copy()
        for i in range(h):
            if i==0:
                continue
            new_x=x**(i+1)
            new_x.columns=new_x.columns+'^%d'%(i+1)
            xh=xh.join(new_x,how='inner')
            if cross==True:
                cfg=[]
                for m in range(len(x.columns)-1):
                    for n in range(len(x.columns)-1-m):
                        cfg.append((m,m+n+1))
                for c in cfg:
                    for j in range(i):
                        x1=x.iloc[:,c[0]]
                        x2=x.iloc[:,c[1]]
                        new_x=(x1**(j+1))*(x2**(i-j))
                        new_x.name=x1.name+'^%d'%(j+1)+'_'+x2.name+'^%d'%(i-j)
                        xh=xh.join(new_x,how='inner')
        return xh