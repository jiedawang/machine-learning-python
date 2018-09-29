# -*- coding: utf-8 -*-
import random as rd
import pandas as pd
import numpy as np

class inNode:
    
    def __init__(self,splitColumn=None,splitValue=None,
                 left=None,right=None,exNode=None):
        self.splitColumn=splitColumn
        self.splitValue=splitValue
        self.left=left
        self.right=right
        self.exNode=exNode

class exNode:
    
    def __init__(self,size=0):
        self.size=size
        
class itree_builder:
   
    def build(self,data,cur_height,height_limit):
        if type(data)!=type(pd.DataFrame()):
            raise TypeError('只接受DataFrame类型')  
        for i in range(data.columns.size):
            if data.dtypes[i] in ['O','str']:
                raise TypeError('存在无法处理的字段') 
        if cur_height>=height_limit or len(data)<=1:
            return exNode(size=len(data))
        else:
            q=rd.randint(0, data.columns.size-1)
            p=rd.uniform(min(data.iloc[:,q]),max(data.iloc[:,q]))
            child_l=data[data[data.columns[q]]<p]
            child_r=data[data[data.columns[q]]>=p]
            return inNode(
                    left=self.build(child_l,cur_height+1,height_limit),
                    right=self.build(child_r,cur_height+1,height_limit),
                    splitColumn=data.columns[q],
                    splitValue=p
                    )

class iforest_manager:
    
    def __init__(self,sample_size,itree_num,height_limit):
        self.sample_size=sample_size
        self.itree_num=itree_num
        self.height_limit=height_limit
        self.itrees=[]
        self.size=0
        
    def build(self,data):
        if type(data)!=type(pd.DataFrame()):
            raise TypeError('只接受DataFrame类型')
        data['sample_flag']=0
        builder=itree_builder()
        for i in range(self.itree_num):
            sp=data[data['sample_flag']==0].sample(n=256)
            sp['sample_flag']=1
            data.update(sp)
            sp.drop(['sample_flag'],axis=1,inplace=True)
            self.itrees.append(builder.build(sp,0,self.height_limit))
            self.size+=1
    
    def print_itree(self,itree_id):
        if itree_id>=self.size:
            raise IndexError('下标越界,当前最大下标%d'%(self.size-1))
        node_list=[[self.itrees[itree_id],0]]
        for i in node_list:
            print('--')
            print('节点：%s'%i[0])
            if type(i[0])==type(inNode()):
                node_list.append([i[0].left,i[1]+1])
                node_list.append([i[0].right,i[1]+1])
                print('层数%d , 分割字段：%s , 分割值：%s'
                      %(i[1],i[0].splitColumn,str(i[0].splitValue)))
            else:
                print('层数%d , 叶节点大小：%d'%(i[1],i[0].size))
                
    def itree_achieve(self,d_row,itree_id):
        if itree_id>=self.size:
            raise IndexError('下标越界,当前最大下标%d'%(self.size-1))
        node_list=[[self.itrees[itree_id],0]]
        for i in node_list:
            if type(i[0])==type(inNode()):
                if d_row[i[0].splitColumn]<i[0].splitValue:
                    node_list.append([i[0].left,i[1]+1])
                else:
                    node_list.append([i[0].right,i[1]+1])
            else:
                return i[1]
    
    def data_row_assess(self,d_row):
        h_list=[]
        for i in range(self.size):
            h_list.append(self.itree_achieve(d_row,i))
        return np.mean(h_list)
    
    def data_assess(self,data):
        data['isolation']=0
        for i in range(len(data)):
            data.loc[i,'isolation']=\
            1-self.data_row_assess(data.loc[i,:])/self.height_limit
            
        
        
        
