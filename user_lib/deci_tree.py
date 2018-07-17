# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import user_lib.statistics as stats
import user_lib.data_prep as dp
import numba as nb
from user_lib.check import check_type,check_limit,check_index_match,check_feats_match

#[数据结构]

#节点
class Node:
    '''\n 
    Note: 节点，可以通过首个参数传入series初始化（优先），也可以直接传入各属性
        
    Parameters
    ----------
    parent: 父节点索引，int类型
    sample_n: 流至该节点的训练样本数，int类型
    error: 该节点上的误差，分类模式下为分类错误的个数，回归模式下为方差，float类型
    is_leaf: 是否是叶节点，bool类型
    feature: 内节点参数，用于分裂的特征，str类型
    limit: 内节点参数，限制类型（=，<=,>,in），str类型
    value: 内节点参数，用于分裂的值，float/str/list(str)类型
    output: 叶节点参数，分类结果/回归预测值，dict(str->float)类型/float类型
    ----------
    
    Attributes
    ----------
    nid: 该节点索引，在构造树时自动赋值，int类型
    depth: 深度，int类型
    childs: 子节点索引列表，list(int)类型
    childs_feature: 分裂特征，str类型
    ----------
    '''

    def __init__(self,data=None,parent=-1,sample_n=0,error=0,is_leaf=False,
                 feature=None,limit=None,value=None,output=None):
        if limit not in ['<=','>','=','in',None]:
            raise ValueError('unknown limit')
        if type(data)!=type(None):
            self.load_series(data)
        else:
            self.load(parent,sample_n,error,is_leaf,feature,limit,value,output)
     
    #用于设置属性
    def load(self,parent,sample_n,error,is_leaf,feature,limit,value,output):
        self.parent=parent
        self.childs=[]
        self.childs_feature=None
        self.sample_n=sample_n
        self.error=error
        self.is_leaf=is_leaf
        self.feature=feature
        self.limit=limit
        self.value=value
        #output类型不符合要求时尝试转换
        try:
            if type(output)==type(None):
                output={}
            if type(output)==type(''):
                output=eval(output)
            if type(output)!=type({}):
                output=float(output)
        except:
            raise TypeError('The dtype of output should be dict or float')
        self.output=output
        self.depth=0
        self.nid=0
    
    #通过series加载
    def load_series(self,data):
        '''
        data: 节点信息，Series类型
        '''
        #判断输入格式是否正确
        if type(data)!=type(pd.Series()):
            raise TypeError('The input should be series')
        #标签是否对应
        label=Node.info_label()
        if data.index.tolist()!=label:
            raise ValueError('The index do not meet the requirements')
        #加载
        self.load(data['parent'],data['sample_n'],data['error'],data['is_leaf'],
                  data['feature'],data['limit'],data['value'],data['output'])
     
    #添加子节点信息
    def add_child(self,child):
        '''
        child: 子节点索引，int类型
        '''
        self.childs.append(child)
      
    #将节点属性转换为字符串
    def info_to_str(self):
        '''
        return
        0: 节点信息，str类型
        '''
        if self.is_leaf==False:
            if self.parent==-1:
                return '<inNode Id=%d pId=%d depth=%d> sample_n:%d error:%f'\
                    %(self.nid,self.parent,self.depth,self.sample_n,round(self.error,4))
            else:
                return '|-- %s %s %s'%(self.feature,self.limit,self.value)+\
                    '\n<inNode Id=%d pId=%d depth=%d> sample_n:%d error:%f'\
                    %(self.nid,self.parent,self.depth,self.sample_n,round(self.error,4))
        else:
            if self.parent==-1:
                return '<leafNode Id=%d pId=%d depth=%d> sample_n:%d error:%f\n output >> %s'\
                    %(self.nid,self.parent,self.depth,self.sample_n,round(self.error,4),str(self.output))
            else:
                return '|-- %s %s %s'%(self.feature,self.limit,self.value)+\
                    '\n<leafNode Id=%d pId=%d depth=%d> sample_n:%d error:%f\n output >> %s'\
                    %(self.nid,self.parent,self.depth,self.sample_n,round(self.error,4),str(self.output))
    
    #将节点属性转换为列表
    def info_to_list(self):
        '''
        return
        0: 节点属性列表，list类型
        '''
        return [self.nid,self.depth,self.parent,self.childs,self.sample_n,self.error,
                self.is_leaf,self.feature,self.limit,self.value,self.output]
    
    #节点属性标签
    def info_label():
        '''
        return
        0: 节点属性标签列表，list(str)类型
        '''
        return ['nid','depth','parent','childs','sample_n','error',
                'is_leaf','feature','limit','value','output']
    
    #复制节点
    def copy(self):
        '''
        return
        0: 复制的节点，Node类型
        '''
        new_node=Node()
        new_node.parent=self.parent
        new_node.childs=self.childs
        new_node.childs_feature=self.childs_feature
        new_node.sample_n=self.sample_n
        new_node.error=self.error
        new_node.is_leaf=self.is_leaf
        new_node.feature=self.feature
        new_node.limit=self.limit
        new_node.value=self.value
        new_node.output=self.output
        new_node.depth=self.depth
        new_node.nid=self.nid
        return new_node
        
#树
class Tree:
    '''\n 
    Note: 树，可传入dataframe初始化，也可不传入生成一个空的树
        
    Parameters
    ----------
    data: 节点数据，DataFrame类型
    ----------
    
    Attributes
    ----------
    nid: 该节点索引，在构造树时自动赋值，int类型
    depth: 深度，int类型
    childs: 子节点索引列表，list(int)类型
    childs_feature: 分裂特征，str类型
    ----------
    '''

    def __init__(self,data=None):
        if type(data)!=type(None):
            self.load_dataframe(data)
        else:
            self.reset()
     
    #重置树
    def reset(self):
        self.node_count=0
        self.nodes=[]
        self.depth=0
        self.mode=None
        self.classes=[]
        self.features=[]
    
    #添加节点，返回新添加节点的索引
    def add_node(self,node):
        '''
        node: 新节点，Node类型
        return
        0: 新增节点的索引，int类型
        '''
        if type(node)!=type(Node()):
            raise TypeError('Input should be a Node')
        #分配节点id
        if len(self.nodes)==0:
            nid=0
        else:
            nid=self.nodes[-1].nid+1
        node.nid=nid
        #根节点和非根节点有不一样的额外处理
        if node.parent!=-1:
            #更新节点的深度，和其父节点的子节点信息
            p_node,iloc=self.find_node(node.parent,return_iloc=True)
            node.depth=p_node.depth+1
            self.nodes[iloc].add_child(nid)
            self.nodes[iloc].childs_feature=node.feature
            #更新特征列表
            self.features=list(set(self.features+[node.feature]))
        else:
            if type(node.output)==type({}):
                self.mode='c'
            else:
                self.mode='r'
        #更新深度和节点数量
        if node.depth>self.depth:
            self.depth=node.depth
        self.nodes.append(node)
        self.node_count+=1
        #分类模式下更新类别列表
        if self.mode=='c':
            self.classes=list(set(self.classes+list(node.output)))
            self.classes.sort()
        return nid
    
    #查找节点
    def find_node(self,node_id,start=0,end=0,return_iloc=False):
        '''
        node_id: 节点标识，int/float类型
        start: 节点列表查找的起始位置，int类型
        end: 节点列表查找的结束位置(不包含)，int类型
        return_iloc: 是否同时返回节点在列表中的位置索引，bool类型
        return
        0: 查找到的节点，Node类型
        1: 节点在列表中的位置索引，int类型
        '''
        if end==0:
            end=self.node_count
        if (start<0)|(end>self.node_count)|(end<=start):
            raise IndexError('Invalid range')
        try:
            nodeId=int(node_id)
        except:
            raise TypeError('Unrecognized dtype of node_id')
        if nodeId<0:
            raise IndexError('Fail to find node(nid=%d)'%node_id)
        for i in range(start,end):
            if self.nodes[i].nid==node_id:
                if return_iloc==True:
                    return self.nodes[i],i
                else:
                    return self.nodes[i]
        raise IndexError('Fail to find node(nid=%d)'%node_id)
    
    #查找某节点的父节点
    def get_parent(self,node_id):
        '''
        node_id: 节点标识，int/float类型
        return
        0: 父节点，Node类型
        '''
        node,iloc=self.find_node(node_id,return_iloc=True)
        p_node=self.find_node(node.parent,end=iloc)
        return p_node
    
    #查找某节点的所有子节点
    def get_childs(self,node_id):
        '''
        node_id: 节点标识，int/float类型
        return
        0: 子节点列表，list(Node)类型
        '''
        node,iloc=self.find_node(node_id,return_iloc=True)
        childs=[]
        for nid in node.childs:
            node,iloc=self.find_node(nid,start=iloc+1)
            childs.append(node)
        return childs
    
    #将树结构转化为dataframe
    def to_dataframe(self):
        '''
        return
        0: 全部树节点信息，DataFrame类型
        '''
        label=Node.info_label()
        nodes=[]
        for n in self.nodes:
            nodes.append(n.info_to_list())
        df_tree=pd.DataFrame(nodes,columns=label)
        return df_tree
    
    #将dataframe还原为树结构
    def load_dataframe(self,data):
        '''
        data:全部树节点信息，DataFrame类型
        '''
        if type(data)!=type(pd.DataFrame()):
            raise TypeError('The input should be dataframe')
        label=Node.info_label()
        if data.columns.tolist()!=label:
            raise ValueError('The columns do not meet the requirements')
        self.reset()
        for i in range(len(data)):
            dr=data.iloc[i,:]
            self.add_node(Node(dr))
    
    #获取流至该节点的路径   
    def get_path(self,node_id,return_nodes=False):
        '''
        node_id: 节点标识，int/float类型
        return_nodes: 是否返回节点列表，False时返回id列表，bool类型
        return
        0: 流经节点索引列表，list(int)类型
        '''   
        node=self.find_node(node_id)
        if return_nodes==False: 
            path=[node.nid]
            while node.nid>0:
                node=self.get_parent(node.nid)
                path.append(node.nid)
        else:
            path=[node]
            while node.nid>0:
                node=self.get_parent(node.nid)
                path.append(node)
        path.reverse()
        return path
    
    #打印全部节点信息
    def print_nodes(self):
        for i in range(self.node_count):
            print(self.nodes[i].info_to_str())     
            
    #计算树的叶节点数
    def get_leaf_num(self,start_id=0):
        '''
        start_id: 起始的节点id，int类型
        return
        0: 叶节点数量，int类型
        '''
        leafNum=0
        node,iloc=self.find_node(start_id,return_iloc=True)
        queue=[node]
        while len(queue)>0:
            node=queue.pop(0)
            if node.is_leaf==True:
                leafNum+=1
            else:
                for childId in node.childs:
                    child,iloc=self.find_node(childId,start=iloc+1,return_iloc=True)
                    queue.append(child)
        return leafNum
    
    #计算树的深度
    def get_depth(self,start_id=0):
        '''
        start_id: 起始的节点id，int类型
        return
        0: 树的深度，int类型
        '''
        depth_max=0
        node,iloc=self.find_node(start_id,return_iloc=True)
        depth_start=node.depth
        queue=[node]
        while len(queue)>0:
            node=queue.pop(0)
            if node.is_leaf==True:
                depth=node.depth-depth_start
                if depth>depth_max:
                    depth_max=depth
            else:
                for childId in node.childs:
                    child,iloc=self.find_node(childId,start=iloc+1,return_iloc=True)
                    queue.append(child)
        return depth_max        
    
    #单次剪枝操作，将指定id的节点下属所有节点移除并将该节点改为叶节点
    def cut(self,node_id,return_trash=False):
        '''
        node_id: 指定节点id，int类型
        return_trash: 返回丢弃的节点列表，list(Node)类型
        '''
        cut_node,first_iloc=self.find_node(node_id,return_iloc=True)
        #叶节点不能剪枝
        if cut_node.is_leaf==True:
            raise IndexError('LeafNode can not be cut')
        #筛选所有需要删除的下属节点
        iloc=first_iloc
        remove=[(cut_node,first_iloc)]
        for node,iloc in remove:
            if node.is_leaf==False:
                for childId in node.childs:
                    child,iloc=self.find_node(childId,start=iloc+1,return_iloc=True)
                    remove.append((child,iloc))
        #倒序逐个按位置索引删除节点
        remove.reverse()
        trash=[]
        for node,iloc in remove:
            if iloc==first_iloc:
                trans_node=self.nodes[iloc].copy()
                trans_node.is_leaf=True
                trans_node.childs=[]
                trans_node.childs_feature=None
                self.nodes[iloc]=trans_node
            else:
                trash.append(self.nodes.pop(iloc))
        #更新树信息
        self.node_count=len(self.nodes)
        self.depth=self.get_depth()
        #返回删除的节点
        if return_trash==True:
            return trash
    
    #复制树
    def copy(self,deep=False):
        '''
        deep: 深度复制，True时会将每一个节点也复制，bool类型，默认False
        return
        0: 复制的树，Tree类型
        '''
        new_tree=Tree()
        new_tree.node_count=self.node_count
        if deep==True:
            new_tree.nodes=[node.copy() for node in self.nodes]
        else:
            new_tree.nodes=self.nodes.copy()
        new_tree.depth=self.depth
        new_tree.mode=self.mode
        new_tree.classes=self.classes.copy()
        return new_tree
    
    #对单行数据应用树
    def go(self,dr):
        node=self.nodes[0]
        while node.is_leaf==False:
            find_flag=False
            for child_id in node.childs:
                child=self.find_node(child_id)
                if child.limit=='<=':
                    if dr[child.feature]<=child.value:
                        node=child
                        find_flag=True
                        break
                elif child.limit=='>':
                    if dr[child.feature]>child.value:
                        node=child
                        find_flag=True
                        break
                elif child.limit=='in':
                    if dr[child.feature] in child.value:
                        node=child
                        find_flag=True
                        break
                else:
                    if dr[child.feature]==child.value:
                        node=child
                        find_flag=True
                        break
            if find_flag==False:
                break
        return node.nid

#[运算函数]

#信息熵etp=sum(-p*log2(p))
#p=每个取值value的占比
#表示随机变量不确定性的度量，范围0~log2(n)，数值越大不确定性越大,n为离散值种类数
#0log0=0 ；当对数的底为2时，熵的单位为bit；为e时，单位为nat。
@nb.jit(nopython=True,cache=True)
def entropy(array,weight):
    '''
    array: 数据列，narray类型
    weight: 样本权重，narray类型
    return 
    0: 熵，float类型
    '''
    values,weights=dp.weight_sum(array,weight)
    p=weights/weight.sum()
    etp=(-p*np.log2(p)).sum()
    return etp

#子集熵
#childs_etp=sum(p*etp)
#p=子集数据量占比,etp=子集的熵
@nb.jit(nopython=True,cache=True)
def childs_entropy(array,split,weight):
    '''
    array: 分类列，narray类型
    split: 子集索引，narray类型
    weight: 样本权重，narray类型
    return
    0: 分裂后的熵，float类型
    '''
    values,weights=dp.weight_sum(split,weight)
    result=0.0
    p=weights/weight.sum()
    for i in range(len(values)):
        boolIdx=(split==values[i])
        result+=p[i]*entropy(array[boolIdx],weight[boolIdx])
    return result

#基尼指数gini=1-sum(p*p)
#p=每个取值value的占比
#表示随机变量不确定性的度量，范围0~1，数值越大不确定性越大
#基尼指数的计算一般只针对离散分类，不用于特征
@nb.jit(nopython=True,cache=True)
def gini(array,weight):
    '''
    array: 数据列，narray类型
    weight: 样本权重，narray类型
    return
    0: 基尼指数，float类型
    '''
    values,weights=dp.weight_sum(array,weight)
    p=weights/weight.sum()
    g=1.0-(p**2).sum()
    return g

#子集基尼
@nb.jit(nopython=True,cache=True)
def childs_gini(array,split,weight):
    '''
    array: 需要求基尼指数的分类列,narray类型
    split: 子集索引，narray类型
    weight: 样本权重，narray类型
    return
    0: 分裂后的基尼指数，float类型
    '''
    values,weights=dp.weight_sum(split,weight)
    result=0.0
    p=weights/weight.sum()
    for i in range(len(values)):
        boolIdx=(split==values[i])
        result+=p[i]*gini(array[boolIdx],weight[boolIdx])
    return result

#平方误差
@nb.jit(nopython=True,cache=True)
def sqr_err(array,weight):
    '''
    array: 数据列,narray类型
    weight: 样本权重，narray类型
    return
    0: 平方误差，float类型
    '''
    re=array-array.mean()
    return np.dot(re.T,re*weight)/weight.sum()

#子集平方误差
@nb.jit(nopython=True,cache=True)
def childs_sqr_err(array,split,weight):
    '''
    array: 数据列,narray类型
    split: 子集索引，narray类型
    weight: 样本权重，narray类型
    return
    0: 分裂后的平方误差，float类型
    '''
    values,weights=dp.weight_sum(split,weight)
    result=0.0
    p=weights/weight.sum()
    for i in range(len(values)):
        boolIdx=(split==values[i])
        result+=p[i]*sqr_err(array[boolIdx],weight[boolIdx])
    return result

#数据集分预分裂
#连续数据集按阈值划分到两个子集，
#离散数据按每个取值划分到多个子集或按取值二分组合划分到两个子集
def pre_split(array,continuity=False,key=None):
    '''
    array: 数据列,narray类型
    continuity: 连续性，bool类型
    key: 分裂依据，连续数据->float类型，离散数据->int或list(int)类型
    return 
    0: 子集索引，narray类型
    1: 子集标签，list(tuple(str,float/int/list(int)))类型
    '''
    idx=np.zeros(len(array)).astype('int64')
    if continuity==True:
        idx[array>key]=1
        label=[('<=',key),('>',key)]
    else:
        if type(key)==type(None):
            values,counts=dp.unique_count(array)
            for i in range(1,len(values)):
                idx[array==values[i]]=i
            label=[('=',value) for value in values]
        else:
            idx[dp.isin(array,key[1])]=1
            label=[('in',key[0]),('in',key[1])]
    return idx,label

#数据集分裂
def split(X,y,weight,split_idx):
    '''
    X: 特征列，narray类型
    y: 观测值列，narray类型
    weight: 权重，narray类型
    split_idx: 子集索引，narray类型
    return 
    0: X分裂后的集合，list(narray) 
    1: y分裂后的集合，list(narray) 
    2: weight分裂后的集合，list(narray) 
    3: 最小子集样本数，int类型
    '''
    values,counts=dp.unique_count(split_idx)
    result_X,result_y,result_w=[],[],[]
    min_sample_n=len(y)
    for value in values:
        result_X.append(X[split_idx==value])
        result_y.append(y[split_idx==value])
        result_w.append(weight[split_idx==value])
        sample_n=len(result_y[-1])
        if sample_n<min_sample_n:
            min_sample_n=sample_n
    return result_X,result_y,result_w,min_sample_n

#最优分裂方案选择
def choose_split(x,y,continuity_x,weight,criterion='gini',simplify=True):
    '''
    x:特征列，narray类型
    y:观测值列，narray类型
    weight: 权重，narray类型
    criterion:衡量标准，str类型，支持entropy/gini/sqr_err
    return
    0:最优分裂指标，float类型 
    1:最优分裂子集索引，narray类型 
    2:最优分裂子集标签，list(tuple)类型 
    '''
    #初始化变量
    if criterion=='entropy':
        bestCriterion=entropy(y,weight)
    elif criterion=='gini':
        bestCriterion=gini(y,weight)
    elif criterion=='sqr_err':
        bestCriterion=sqr_err(y,weight)
    else:
        raise ValueError('invalid criterion')
    bestSplitIdx,bestSplitLabel=None,None
    #获取所有分裂方案
    if continuity_x==True:
        #需要尝试的分裂点
        values,counts=dp.unique_count(x)
    else:
        #需要尝试的分裂组合
        values,counts=dp.unique_count(x)
        if len(values)>1:
            if criterion=='entropy':
                split_idx,split_label=pre_split(x,False,None)
                c=childs_entropy(y,split_idx,weight)
                return c,split_idx,split_label
            if simplify==False:
                combines=dp.combine_enum(values,split_mode=True)
            else:
                combines=np.eye(len(values),len(values))
        else:
            return 0.0,None,None
    #计算每个分裂方案的指标值，选出最优
    if continuity_x==True:
        m=len(values)-1
    else:
        m=len(combines) 
    for i in range(m):
        #获取分裂子集索引
        if continuity_x==True:
            #注：此处分裂阈值必须取中位数，直接取value[i]会影响到boost算法的集成效果
            key=(values[i]+values[i+1])/2.0
        else:
            key=[dp.combine_take(values,combines[i]),
                 dp.combine_take(values,1-combines[i])]
        split_idx,split_label=pre_split(x,continuity_x,key)
        #计算指标，选出最优
        if criterion=='entropy':
            c=childs_entropy(y,split_idx,weight)
        elif criterion=='gini':
            c=childs_gini(y,split_idx,weight)
        elif criterion=='sqr_err':
            c=childs_sqr_err(y,split_idx,weight)
        if c<bestCriterion:
            bestCriterion=c
            bestSplitIdx=split_idx
            bestSplitLabel=split_label
    return bestCriterion,bestSplitIdx,bestSplitLabel

#[构造类]

#决策树
class DecisionTree:
    '''\n  
    Note: 决策树，支持分类和回归，
          id3其实不太实用，但为了展示原始版本的决策树还是实现了，以后会移除
    
    Parameters
    ----------
    mode: 模式，分类->'c'，回归->'r'，默认'c'
    model_type : 模型算法类型，str类型(id3,c4.5,cart)，
                 'id3'->分类，离散特征+离散输出，
                 'c4.5'->分类，离散或连续特征+离散输出，
                 'cart'->分类或回归，离散或连续特征+离散或连续输出
                 默认值'cart'
    depth_max : 最大深度，int类型(>=1)，None表示无限制，默认值10
    split_sample_n : 分裂所需最少样本数，int类型(>=2)，默认值2
    leaf_sample_n : 叶节点所需最少样本数，int类型(>=1)，默认值1
    features_use : 每次使用的特征数量，str/float/int类型，默认值'all'
                  'all'->全量，
                  'sqrt'->总数量的平方根，
                  'log2'->总数量的2的对数，
                  float->总数量的相应比例，区间(0.0,1.0)，
                  int->固定数量，区间[1,feature_num]，
                  注意，即使使用'all'，生成的树依旧可能不同，
                  原因是在选取最优划分特征时，可能会出现多个特征都是最优划分的情况，
                  选用不同特征生成的树不同，准确率也不一样，
                  所以在遍历特征时保留了顺序上的随机性，以提供取到最优树的可能
    features_reuse : 是否允许一个特征重复使用，bool类型，默认值True
    ----------
    
    Attributes
    ----------
    tree: 拟合好的决策树，Tree类型
    time_cost: 耗时统计，Series类型
    ----------
    '''
    
    #构造函数，主要作用是校验和保存配置变量
    def __init__(self,mode='c',model_type='cart',
                 depth_max=None,split_sample_n=2,leaf_sample_n=1,
                 features_use='all',features_reuse=True):
        #校验参数类型和取值
        #check_type(变量名，变量类型，要求类型)
        #check_limit(变量名，限制条件，正确取值提示)
        check_type('mode',type(mode),type(''))
        mode_list=['c','r']
        mode=mode.lower()
        check_limit('mode',mode in mode_list,str(mode_list))
        
        check_type('model_type',type(model_type),type(''))
        type_list=['id3','c4.5','cart']
        model_type=model_type.lower()
        check_limit('model_type',model_type in type_list,str(type_list))
        if (mode=='r')&(model_type!='cart'):
            raise ValueError('only cart supports for r')
        
        if type(depth_max)!=type(None):
            check_type('depth_max',type(depth_max),type(0))
            check_limit('depth_max',depth_max>0,'value>0')
        
        check_type('split_sample_n',type(split_sample_n),type(0))
        check_limit('split_sample_n',split_sample_n>=2,'value>=2')
        check_type('leaf_sample_n',type(leaf_sample_n),type(0))
        check_limit('leaf_sample_n',leaf_sample_n>=1,'value>=1')
        check_type('features_reuse',type(features_reuse),type(True))
        
        check_type('features_use',type(features_use),[type(0),type(1.0),type('')])
        required="float(>0.0,<1.0),int(>=1,<=feature_n),str(['all','sqrt','log2'])"
        if type(features_use)==type(''):
            check_limit('features_use',features_use in ['all','sqrt','log2'],required)
        elif type(features_use)==type(0):
            check_limit('features_use',features_use>=1,required)
        elif type(features_use)==type(0.0):
            check_limit('features_use',(features_use>0.0)&(features_use<1.0),required)
        #保存参数
        self.mode=mode
        self.model_type=model_type
        self.depth_max=depth_max
        self.split_sample_n=split_sample_n
        self.leaf_sample_n=leaf_sample_n
        self.features_use=features_use
        self.features_reuse=features_reuse

    #最优特征选择(ID3)
    #选择依据：信息增益
    #信息增益=分裂前的熵-分裂后的熵
    #用于衡量基于某特征分裂后分类的不确定性降低了多少
    def choose_feature_by_id3_(self,X,y,columns,weight):
        '''
        X: 所有参与选择的特征列，narray类型
        y: 分类列，narray类型
        columns: 列索引，list(int)类型
        weight: 样本权重，narray类型
        return
        0: 最优分裂特征的索引 ，int类型
        1: 最优分裂子集索引，narray类型
        2: 最优分裂子集标签，list(tuple(str,float/int/list(int)))类型
        '''
        #随机抽取特征
        features_use_n=self.features_use_n
        feature_n=len(columns)
        if features_use_n>feature_n:
            features_use_n=feature_n
        features_idx=pd.Series(range(feature_n))
        features_idx=features_idx.sample(features_use_n)
        #计算分裂前的信息熵
        baseEntropy=entropy(y,weight)
        #初始化变量
        bestInfGain,bestFeatureIdx=0.0,-1
        bestSplitIdx,bestSplitLabel=None,None
        #逐个计算按不同特征分割后的信息增益并选出增益最大的一个特征
        for i in features_idx:
            x=X[:,i]
            #特征列值统一，无法用于分裂
            if len(np.unique(x))<=1:
                continue
            split_idx,split_label=pre_split(x)
            infGain=baseEntropy-childs_entropy(x,split_idx,weight)
            if infGain>bestInfGain:
                bestInfGain=infGain
                bestFeatureIdx=i
                bestSplitIdx=split_idx
                bestSplitLabel=split_label
        return bestFeatureIdx,bestSplitIdx,bestSplitLabel
    
    #最优特征选择(C4.5)
    #选择依据：信息增益比
    #信息增益比=信息增益/用于分裂的特征列的熵
    #避免因为特征取值多而导致信息增益偏大
    #C4.5增加了对连续数据的处理，连续特征根据信息增益选择最佳分裂点进行二分裂
    def choose_feature_by_c45_(self,X,y,columns,weight):
        '''
        X: 所有参与选择的特征列，narray类型
        y: 分类列，narray类型
        columns: 列索引，list(int)类型
        weight: 样本权重，narray类型
        return
        0: 最优分裂特征的索引 ，int类型
        1: 最优分裂子集索引，narray类型
        2: 最优分裂子集标签，list(tuple(str,float/int/list(int)))类型
        '''
        #随机抽取特征
        features_use_n=self.features_use_n
        feature_n=len(columns)
        if features_use_n>feature_n:
            features_use_n=feature_n
        features_idx=pd.Series(range(feature_n))
        features_idx=features_idx.sample(features_use_n)
        #计算分裂前的信息熵
        baseEntropy=entropy(y,weight)
        #初始化变量
        bestInfGainRatio=0.0
        bestFeatureIdx=-1
        bestSplitIdx,bestSplitLabel=None,None
        #逐个计算按不同特征分割后的信息增益率并选出增益率最大的一个特征
        for i in features_idx:
            x=X[:,i]
            #特征列值统一，无法用于分裂
            if len(np.unique(x))<=1:
                continue
            #选择分裂方案，计算信息增益率
            continuity_x=self.continuity_X[columns[i]]
            childsEntropy,splitIdx,splitLabel=\
                choose_split(x,y,continuity_x,weight,criterion='entropy')
            if type(splitIdx)==type(None):
                continue
            infGain=baseEntropy-childsEntropy
            splitFeatEntropy=entropy(splitIdx,weight)
            if splitFeatEntropy==0:
                continue
            infGainRatio=infGain/splitFeatEntropy
            if infGainRatio>bestInfGainRatio:
                bestInfGainRatio=infGainRatio
                bestFeatureIdx=i
                bestSplitIdx=splitIdx
                bestSplitLabel=splitLabel
        return bestFeatureIdx,bestSplitIdx,bestSplitLabel
       
    #最优特征选择(CART分类树)
    #选择依据：基尼指数
    #分类树输出离散标签
    #cart和c4.5采用同样的方式处理连续特征
    #同时cart对离散特征也进行二分处理，将离散值按不同组合分裂为两个子集，从中选择最优
    def choose_feature_by_cart_c_(self,X,y,columns,weight):
        '''
        X:所有参与选择的特征列，narray类型
        y:分类列，narray类型
        columns: 列索引，list(int)类型
        weight: 样本权重，narray类型
        return
        0: 最优分裂特征的索引 ，int类型
        1: 最优分裂子集索引，narray类型
        2: 最优分裂子集标签，list(tuple(str,float/int/list(int)))类型
        '''
        #随机抽取特征
        features_use_n=self.features_use_n
        feature_n=len(columns)
        if features_use_n>feature_n:
            features_use_n=feature_n
        features_idx=pd.Series(range(feature_n))
        features_idx=features_idx.sample(features_use_n)
        #初始化变量
        bestGini=gini(y,weight)
        bestFeatureIdx=-1
        bestSplitIdx,bestSplitLabel=None,None
        #逐个计算按不同特征分割后的基尼指数并选出指数最小的一个特征
        for i in features_idx:
            x=X[:,i]
            #特征列值统一，无法用于分裂
            if len(np.unique(x))<=1:
                continue
            #选择分裂方案，计算基尼指数
            continuity_x=self.continuity_X[columns[i]]
            childsGini,splitIdx,splitLabel=\
                choose_split(x,y,continuity_x,weight,criterion='gini')
            if type(splitIdx)==type(None):
                continue
            if childsGini<bestGini:
                bestGini=childsGini
                bestFeatureIdx=i
                bestSplitIdx=splitIdx
                bestSplitLabel=splitLabel
        return bestFeatureIdx,bestSplitIdx,bestSplitLabel
    
    #最优特征选择(CART回归树)
    #选择依据：平方误差
    #回归树输出连续数值
    def choose_feature_by_cart_r_(self,X,y,columns,weight):
        '''
        X:所有参与选择的特征列，narray类型
        y:回归值列，narray类型
        columns: 列索引，list(int)类型
        weight: 样本权重，narray类型
        return
        0: 最优分裂特征的索引 ，int类型
        1: 最优分裂子集索引，narray类型
        2: 最优分裂子集标签，list(tuple(str,float/int/list(int)))类型
        '''
        #随机抽取特征
        features_use_n=self.features_use_n
        feature_n=len(columns)
        if features_use_n>feature_n:
            features_use_n=feature_n
        features_idx=pd.Series(range(feature_n))
        features_idx=features_idx.sample(features_use_n)
        #初始化变量
        bestErr=sqr_err(y,weight)
        bestFeatureIdx=-1
        bestSplitIdx,bestSplitLabel=None,None
        #逐个计算按不同特征分割后的方差并选出方差最小的一个特征
        for i in features_idx:
            x=X[:,i]
            #特征列值统一，无法用于分裂
            if len(np.unique(x))<=1:
                continue
            #是否为连续特征
            #选择分裂方案，计算基尼指数
            continuity_x=self.continuity_X[columns[i]]
            childsSqrErr,splitIdx,splitLabel=\
                choose_split(x,y,continuity_x,weight,criterion='sqr_err')
            if type(splitIdx)==type(None):
                continue
            if childsSqrErr<bestErr:
                bestErr=childsSqrErr
                bestFeatureIdx=i
                bestSplitIdx=splitIdx
                bestSplitLabel=splitLabel
        return bestFeatureIdx,bestSplitIdx,bestSplitLabel
    
    #计算每个类的概率
    def compute_proba_(self,y,name,weight,return_counts=False):
        '''
        y: 分类列，narray类型
        return_count: 是否返回类和计数，bool类型
        return
        0: 分类概率，dict类型
        1: 类列表，narray类型
        2: 类计数，narray类型
        '''
        proba={}
        values,weights=dp.weight_sum(y,weight)
        p=weights/weight.sum()
        for i in range(len(values)):
            value=dp.index_to_label_(name,values[i],self.mapping_y)
            proba[value]=round(p[i],4)
        if return_counts==True:
            return proba,values,weights
        else:
            return proba
          
    #选择概率最高的类作为叶节点判定的类,用于预测
    def choose_class_(self,p_y_,classes,multi_output=False):
        '''
        p_y_: 预测的分类概率，DataFrame类型
        classes: 类标签，list(str)类型
        multi_output: 多列输出，bool类型，默认False
        return
        0: 选择的类，Series或DataFrame类型
        '''
        p_y_=(p_y_.T==p_y_.max(axis=1)).T.astype('int')
        if multi_output==True:
            p_y_.columns=classes
            return p_y_
        else:
            p_y=pd.Series(np.full(len(p_y_),''),index=p_y_.index)
            for i in range(len(classes)):
                p_y[p_y_.iloc[:,i]==1]=classes[i]
            return p_y
    
    #叶节点判断(仅先行判断的条件，非全部条件)
    def is_leaf_(self,X,y,depth):
        '''
        X: 当前所有的特征列，DataFrame类型
        y: 观测值列，Series类型
        depth: 当前深度，int类型
        return
        0: 是否叶节点，bool类型
        '''
        #超出高度上限，不继续分裂
        if self.depth_max!=None:
            if depth>=self.depth_max:
                if self.build_proc==True:
                    print('<LeafNode> reach maximum depth')
                return True
        #可用特征不足，不继续分裂
        if X.shape[1]==0:
            if self.build_proc==True:
                print('<LeafNode> lack of feature')
            return True
        #数据集过小，不继续分裂
        if len(X)<self.split_sample_n:
            if self.build_proc==True:
                print('<LeafNode> samples too small')
            return True
        #只有一个类，不继续分裂
        if len(np.unique(y))==1:
            if self.build_proc==True:
                print('<LeafNode> only one class')
            return True
        #特征向量统一，不继续分裂
        if len(np.unique(X))==1:
            if self.build_proc==True:
                print('<LeafNode> feature vector unification')
            return True
        return False
    
    #构建树
    def build_(self,X,y,weight):
        '''
        X: 当前所有的特征列，DataFrame类型
        y: 观测值列，Series类型
        weight: 样本权重，Series类型
        return
        0: 决策树，Tree类型
        '''
        #初始化决策树
        deciTree=Tree()
        #等待处理的数据队列：特征，观测值，连续性，父节点id，深度，
        #                   分裂特征名,约束方式，分裂值
        columns=list(range(len(X.columns)))
        queue=[(X.values,y.values,columns,weight.values,-1,0,None,None,None)]
        while len(queue)>0:
            start0=time.clock()
            X_,y_,columns_,weight_,parent,depth,feature,limit,value=queue.pop(0)
            self.time_cost['queue operate']+=time.clock()-start0
            if self.build_proc==True:
                if parent!=-1:
                    print('<Split> %s %s %s'%
                          (feature,limit,str(value)))
                print('Current dataset size: %d'%len(y_))
            start0=time.clock()
            is_leaf=self.is_leaf_(X_,y_,depth)
            self.time_cost['check input']+=time.clock()-start0
            #选择最优特征进行分裂，并记录结果
            if is_leaf==False:
                start0=time.clock()
                try:
                    if self.model_type=='id3':
                        bestFeatureIdx,bestSplitIdx,bestSplitLabel=\
                            self.choose_feature_by_id3_(X_,y_,columns_,weight_)
                    elif self.model_type=='c4.5':
                        bestFeatureIdx,bestSplitIdx,bestSplitLabel=\
                            self.choose_feature_by_c45_(X_,y_,columns_,weight_)
                    elif (self.model_type=='cart')&(self.mode=='c'):
                        bestFeatureIdx,bestSplitIdx,bestSplitLabel=\
                            self.choose_feature_by_cart_c_(X_,y_,columns_,weight_)
                    elif (self.model_type=='cart')&(self.mode=='r'):
                        bestFeatureIdx,bestSplitIdx,bestSplitLabel=\
                            self.choose_feature_by_cart_r_(X_,y_,columns_,weight_)
                    else:
                        raise TypeError('Unknown type')
                except:
                    self.err_X,self.err_y,self.err_col=X_,y_,columns_
                    print('subsets of data which cause error are saved as .err_X,.err_y,.err_col')
                    raise
                self.time_cost['compute best split']+=time.clock()-start0
                #未能成功选出可供分裂的特征
                if bestFeatureIdx==-1:
                    if self.build_proc==True:
                        print('<LeafNode> fail to choose a feature')
                    is_leaf=True
            #当前节点分类概率/回归值和节点误差计算
            start0=time.clock()
            if self.mode=='r':
                output=(weight_*y_).sum()/weight_.sum()
                error=((y_-y_.mean())**2*weight_).sum()
            else:
                output,values,weights=self.compute_proba_(y_,y.name,weight_,return_counts=True)
                error=weights.sum()-weights.max()
            self.time_cost['compute node attr']+=time.clock()-start0
            #将分裂条件中的离散值索引转换回标签
            if limit=='in':
                value=[dp.index_to_label_(feature,e,self.mapping_X) for e in value]
            elif limit=='=':
                value=dp.index_to_label_(feature,value,self.mapping_X)
            if is_leaf==True:
                #添加叶节点
                start0=time.clock()
                node=Node(parent=parent,sample_n=len(y_),error=error,is_leaf=True,
                          feature=feature,limit=limit,value=value,output=output)
                nodeId=deciTree.add_node(node)
                self.time_cost['add node']+=time.clock()-start0
            else:
                #获取最优分裂特征的相关信息
                bestFeatureLabel=X.columns[columns_[bestFeatureIdx]]
                #分裂数据集
                start0=time.clock()
                splited_X,splited_y,splited_weight,min_sample_n=split(X_,y_,weight_,bestSplitIdx)
                self.time_cost['split data']+=time.clock()-start0
                #下属叶节点样本数存在小于设定值的，将该节点设为叶节点，否则内节点
                if min_sample_n<self.leaf_sample_n:
                    start0=time.clock()
                    node=Node(parent=parent,sample_n=len(y_),error=error,is_leaf=True,
                    feature=feature,limit=limit,value=value,output=output)
                    nodeId=deciTree.add_node(node)
                    self.time_cost['add node']+=time.clock()-start0
                else:
                    start0=time.clock()
                    node=Node(parent=parent,sample_n=len(y_),error=error,is_leaf=False,
                              feature=feature,limit=limit,value=value,output=output)
                    nodeId=deciTree.add_node(node)
                    self.time_cost['add node']+=time.clock()-start0
                    if self.features_reuse==False:
                        columns__=columns_.copy()
                        columns__.pop(bestFeatureIdx)
                        #将分裂后的数据集加入队列继续处理
                        start0=time.clock()
                        for i in range(len(bestSplitLabel)):
                            queue.append((np.delete(splited_X[i],bestFeatureIdx,axis=1),
                                          splited_y[i],columns__,splited_weight[i],nodeId,depth+1,
                                          bestFeatureLabel,bestSplitLabel[i][0],bestSplitLabel[i][1]))
                        self.time_cost['queue operate']+=time.clock()-start0
                    else:
                        start0=time.clock()
                        for i in range(len(bestSplitLabel)):
                            queue.append((splited_X[i],splited_y[i],columns_,splited_weight[i],nodeId,depth+1,
                                          bestFeatureLabel,bestSplitLabel[i][0],bestSplitLabel[i][1]))
                        self.time_cost['queue operate']+=time.clock()-start0
        return deciTree
    
    #X输入校验
    def check_input_X_(self,X,name='X',mode=None,model_type=None,
                       to_index=False,return_source=False):
        if type(mode)==type(None):
            mode=self.mode
        if type(model_type)==type(None):
            model_type=self.model_type
        #类型校验
        check_type(name,type(X),type(pd.DataFrame()))
        #连续性判断
        continuity_X=dp.get_continuity(X,name)
        if type(continuity_X)==type(True):
            continuity_X=[continuity_X]
        #ID3不支持连续特征
        if model_type=='id3':
            if True in continuity_X:
                raise TypeError('ID3 does not support continuous features')
        #视情况将X转换为数值索引(numba的需要)
        if to_index==True:
            if return_source==True:
                X0=X.copy()
            if False not in continuity_X:
                mapping_X=None
            else:
                cols=[]
                for i in range(len(continuity_X)):
                    if continuity_X[i]==False:
                        cols.append(X.columns[i])
                X,mapping_X=dp.label_to_index(X,cols)
            if return_source==True:
                return X,continuity_X,mapping_X,X0
            else:
                return X,continuity_X,mapping_X
        else:
            return X,continuity_X
        
    #y输入校验
    def check_input_y_(self,y,name='y',mode=None,model_type=None,
                       to_index=False,return_source=False):
        if type(mode)==type(None):
            mode=self.mode
        if type(model_type)==type(None):
            model_type=self.model_type
        #类型校验
        check_type(name,type(y),type(pd.Series()))
        #连续性判断
        continuity_y=dp.get_continuity(y,name)
        #回归模式的y必须是数值型,分类模式下y转换为str类型
        if mode=='r':
            if continuity_y==False:
                raise TypeError('Regressor only support %s for numeric'%name)
        else:
            y=y.astype('str')
        #视情况将y转换为数值索引(numba的需要)
        if to_index==True:
            if return_source==True:
                y0=y.copy()
            if mode=='r':
                mapping_y=None
            else:
                y,mapping_y=dp.label_to_index(y,[y.name])
            if return_source==True:
                return y,continuity_y,mapping_y,y0
            else:
                return y,continuity_y,mapping_y
        else:
            return y,continuity_y
    
    #计算每次分裂使用的特征数量上限
    def compute_features_use_n_(self,feature_n,features_use):
        if type(features_use)==type(''):
            if features_use=='all':
                features_use_n=feature_n
            elif features_use=='sqrt':
                features_use_n=np.sqrt(feature_n)
            elif features_use=='log2':
                features_use_n=np.log2(feature_n)
        elif type(features_use)==type(0):
            if features_use>feature_n:
                features_use_n=feature_n
            else:
                features_use_n=features_use
        elif type(features_use)==type(0.0):
            features_use_n=features_use*feature_n
        return int(features_use_n)
    
    #拟合
    def fit(self,X,y,sample_weight=None,output=False,show_time=False,
            build_proc=False,check_input=True):
        '''\n
        Function: 使用输入数据拟合决策树
        
        Note: 数据列的连续性会进行自动判断，不被支持的类型需要预处理
              (int64,float64)->连续
              (bool,category,object)->离散
              所有离散数据会强制转换为str标签

        Parameters
        ----------
        X: 所有的特征列，DataFrame类型
        y: 观测值列，Series类型
        sample_weight: 样本权重，Series类型
        output: 是否输出拟合好的树，bool类型，默认值False
        show_time: 是否显示耗时，bool类型，默认值False
        build_proc: 反馈构建过程，bool类型，默认值False
        check_input: 是否进行输入校验，bool类型，默认值True
        ----------
        
        Returns
        -------
        0: 决策树，Tree类型
        -------
        '''
        start = time.clock()
        #输入校验
        check_type('check_input',type(check_input),type(True))
        if type(sample_weight)==type(None):
            temp=np.ones(len(X))
            sample_weight=pd.Series(temp/temp.sum(),index=X.index)
        if check_input==True:
            check_type('output',type(output),type(True))
            check_type('show_time',type(show_time),type(True))
            check_type('build_proc',type(build_proc),type(True))
            #校验X,y输入
            X,self.continuity_X,self.mapping_X=self.check_input_X_(X,to_index=True)
            y,self.continuity_y,self.mapping_y=self.check_input_y_(y,to_index=True)
            #校验X,y输入是否匹配
            check_index_match(X,y,'X','y')
            #权重向量校验
            check_type('sample_weight',type(sample_weight),type(pd.Series()))
            check_type('elements in sample_weight',sample_weight.dtype,np.float64)
            check_index_match(X,sample_weight,'X','sample_weight')
        #计算每次分裂使用的特征数量上限
        self.features_use_n=self.compute_features_use_n_(len(X.columns),self.features_use)
        self.time_cost=pd.Series(np.zeros(7),name='time cost',
                index=['total cost','queue operate','check input',
                       'compute best split','compute node attr',
                       'split data','add node'])
        self.build_proc=build_proc
        #构建树
        self.tree=self.build_(X,y,sample_weight)
        end = time.clock()
        self.time_cost['total cost']=end-start
        if show_time==True:
            print('\ntime used for trainning: %f'%(end-start))
        if output==True:
            return self.tree
        
    #决策路径
    def decition_path(self,dr,tree=None,return_path=False,check_input=True):
        '''\n
        Function: 获取单行数据的决策路径
        
        Note: 只能一次处理一行数据，主要用于校对，
        需要批量获取数据的决策路径可以使用predict，设置参数return_paths=True
        
        Parameters
        ----------
        dr: 数据行，Series类型
        tree: 决策树，Tree类型，默认调用内部缓存的树
        return_path: 是否仅以返回值形式给到路径，bool类型，默认值False，
                     此模式只返回节点id列表，不包含完整信息
        check_input: 是否进行输入校验，bool类型，默认值True
        ----------
        
        Print
        -----
        0: 流经节点的信息
        -----
        
        Returns
        -----
        0: 流经节点，list(int)类型
        -----
        '''
        if type(tree)==type(None):
            tree=self.tree     
        #校验输入
        check_type('check_input',type(check_input),type(True))
        if check_input==True:
            check_type('tree',type(tree),type(Tree()))
            check_type('return_path',type(return_path),type(True))
            check_type('dr',type(dr),type(pd.Series()))
            check_feats_match(dr.index,tree.features,'dr','tree',mode='right')
        #应用树，获取流至节点，再获取流经节点
        reach=tree.go(dr)
        path=tree.get_path(reach)
        if return_path==True:
            return path
        else:
            iloc=-1
            for node_id in path:
                node,iloc=tree.find_node(node_id,start=iloc+1,return_iloc=True)
                print(node.info_to_str())
    
    #流至节点
    def flow_to_node(self,X,node_id,tree=None,check_input=True):
        '''\n
        Function: 获取数据集流至指定节点处的子集
        
        Parameters
        ----------
        X: 所有的特征列，DataFrame类型
        node_id: 节点Id，int类型
        tree: 决策树，Tree类型，默认调用内部缓存的树
        check_input: 是否进行输入校验，bool类型，默认值True
        ----------
        
        Returns
        -------
        0:流至该节点的数据，DataFrame类型
        -------
        '''
        if type(tree)==type(None):
            tree=self.tree
        #校验输入
        check_type('check_input',type(check_input),type(True))
        if check_input==True:
            check_type('tree',type(tree),type(Tree()))
            check_type('node_id',type(node_id),type(0))
            check_type('X',type(X),type(pd.DataFrame()))
            check_feats_match(X.columns,tree.features,'X','tree',mode='right')
        path_nodes=tree.get_path(node_id,return_nodes=True)
        for node in path_nodes:
            if node.nid==0:
                continue
            if node.limit=='<=':
                X=X[X[node.feature]<=node.value]
            elif node.limit=='>':
                X=X[X[node.feature]>node.value]
            elif node.limit=='in':
                X=X[X[node.feature].isin(node.value)]
            else:
                X=X[X[node.feature]==node.value]
        return X
    
    #预测
    def predict(self,X,tree=None,return_proba=False,return_paths=False,
                show_time=False,check_input=True):
        '''\n
        Function: 使用输入数据和树进行预测，没有输入树时使用内部缓存的树
        
        Parameters
        ----------
        X: 所有特征列，DataFrame类型
        tree: 决策树，Tree类型，默认调用内部缓存的树
        return_proba: 是否返回分类概率，分类模式下有效，bool类型，默认值False
        return_paths: 是否返回决策路径，bool类型，默认值False
                     （路径信息以str类型返回，可转换为list使用）
        show_time: 是否显示耗时，bool类型，默认值False
        check_input: 是否进行输入校验，bool类型，默认值True
        ----------
        
        Returns
        -------
        0: 预测的分类/分类概率/回归值，Series/DataFrame类型
        1: 所有数据最终抵达的节点和决策路径，DataFrame类型
        -------
        '''
        start = time.clock()
        if type(tree)==type(None):
            tree=self.tree
        #校验参数
        check_type('check_input',type(check_input),type(True))
        if check_input==True:
            check_type('tree',type(tree),type(Tree()))
            check_type('return_proba',type(return_proba),type(True))
            check_type('return_paths',type(return_paths),type(True))
            check_type('show_time',type(show_time),type(True))
            X,continuity_X=self.check_input_X_(X)
            check_feats_match(X.columns,tree.features,'X','tree',mode='right')
        #数据集大小
        n=len(X)
        #数据流，记录已达到节点
        #flow=np.zeros(n)
        #分类模式先求分类概率，回归模式直接求回归值
        if tree.mode=='c':
            #定义存放分类结果的DataFrame
            p_y=pd.DataFrame(
                    np.zeros((n,len(tree.classes))),
                    index=X.index,columns=tree.classes)
        else:
            #定义存放回归值的Series
            p_y=pd.Series(np.zeros(n),index=X.index)
        #对每个叶节点进行数据流筛选
        flow=pd.Series(np.zeros(len(X)).astype('int'),index=X.index)
        for node in tree.nodes:
            if node.is_leaf==True:
                X_=self.flow_to_node(X,node.nid,check_input=False)
                if tree.mode=='c':
                    p_y_=pd.DataFrame(node.output,index=X_.index,columns=tree.classes)
                else:
                    p_y_=pd.Series(node.output,index=X_.index)
                p_y.update(p_y_)
        #分类模式下可以返回分类概率或唯一分类
        if (tree.mode=='c')&(return_proba==False):
            p_y=self.choose_class_(p_y,tree.classes)
        #是否返回决策路径
        if return_paths==False:
            end = time.clock()
            if show_time==True:
                print('\ntime used for predict:%f'%(end-start))
            return p_y
        else:
            paths=pd.DataFrame()
            paths['reach']=flow
            paths['path']=''
            reach_nodes=flow.drop_duplicates().sort_values().astype('int').tolist()
            for nodeId in reach_nodes:
                path=tree.get_path(nodeId)
                paths.loc[paths['reach']==nodeId,'path']=str(path)
            end = time.clock()
            if show_time==True:
                print('\ntime used for predict: %f'%(end-start))
            return p_y,paths
    
    #评估
    def assess(self,y,p_y,mode=None,check_input=True):
        '''\n
        Function: 使用输入的观测值和预测值进行模型评估
        
        Notes: 注意数据集的数据类型，分类首选类型str，回归首选类型float64，
               拟合时数据集采用非首选类型可能会导致此处类型不匹配，建议提前转换
        
        Parameters
        ----------
        y:观测值，Series类型
        p_y:预测值，Series类型
        mode:模式，str类型，默认使用内部缓存树的属性，
             'c'->分类，'r'->回归
        check_input: 是否进行输入校验，bool类型，默认值True
        ----------
        
        Returns
        -------
        0: 分类->准确率，回归->R方，float类型
        -------
        '''
        if type(mode)==type(None):
            mode=self.tree.mode
        #校验输入
        check_type('check_input',type(check_input),type(True))
        if check_input==True:
            check_type('mode',type(mode),type(''))
            mode_list=['c','r']
            check_limit('mode',mode in mode_list,str(mode_list))
            y,continuity_y=self.check_input_y_(y,name='y')
            p_y,continuity_p_y=self.check_input_y_(p_y,name='p_y')
            check_index_match(y,p_y,'y','p_y')
        #分类模式求准确率，回归模式求R2
        if mode=='c':
            return stats.accuracy(y,p_y)
        elif mode=='r':
            return stats.r_sqr(y,p_y)
            
    #误差代价err_cost_=sum(E_i)+a*leafNum
    #E_i为下属各个子节点上的误差个数或方差，leafNum为下属叶节点总数，
    #a为平衡参数，用于平衡误差和复杂度，a越大越倾向于选择简单的模型，a为0则只考虑误差
    def err_cost_(self,a=0.0,start_id=0,tree=None,after_prun=False):
        '''
        a: 平衡参数，float类型
        start_id: 起始节点id，int类型
        tree: 决策树，Tree类型
        after_prun: 是否计算剪枝后的误差代价，bool类型
        return
        0: 误差代价
        '''
        #未从外部传入树时使用内部存储的树
        if type(tree)==type(None):
            tree=self.tree
        node,iloc=tree.find_node(start_id,return_iloc=True)
        if after_prun==False:
            leafNum=tree.get_leaf_num(start_id)
            e,queue=leafNum*a,[(node,iloc)]
            while len(queue)>0:
                node,iloc=queue.pop(0)
                if node.is_leaf==True:
                    e+=node.error
                else:
                    for childId in node.childs:
                        child,iloc=tree.find_node(childId,start=iloc+1,return_iloc=True)
                        queue.append((child,iloc))
        else:
            e=a+node.error
        return e
    
    #降低错误率剪枝 Reduced-Error Pruning
    def pruning_rep_(self,test_X,test_y,tree=None,return_subtrees=False):
        '''
        test_X: 测试数据集全部特征列，DataFrame类型
        test_y: 测试数据集全部观测值列，Series类型
        tree: 决策树，Tree类型
        return_subtrees: 是否返回剪枝过程的子树序列，bool类型
        return
        0: 剪枝后的决策树，Tree类型/剪枝过程的子树序列+准确率+节点id，list(Tree,int,int)类型
        '''
        if type(tree)==type(None):
            tree=self.tree.copy()
        else:
            tree=tree.copy()
        #计算完全树在测试集上的准确率
        p_y=self.predict(test_X,tree,show_time=False)
        best_score=self.assess(test_y,p_y,tree.mode)
        subtrees,scores,cut_ids=[tree.copy()],[best_score],[-1]
        #从下至上遍历每个节点
        for i in range(len(tree.nodes)-1,0,-1):
            node=tree.nodes[i]
            #非叶节点才可剪枝
            if node.is_leaf==False:
                #复制树并试剪枝
                tree_=tree.copy()
                tree_.cut(node.nid)
                #计算剪枝后在测试集上的准确率
                p_y=self.predict(test_X,tree_,show_time=False)
                score=self.assess(test_y,p_y,tree.mode)
                #若准确率没有降低，保留剪枝后的树
                if score>=best_score:
                    tree,best_score=tree_,score
                    subtrees.append(tree.copy())
                    scores.append(score)
                    cut_ids.append(node.nid)
        if return_subtrees==True:
            return subtrees,scores,cut_ids
        else:
            return tree
    
    #悲观剪枝 Pessimistic Error Pruning
    #注：暂未找到pep应用于回归树的参考案例，所以暂不支持回归树剪枝
    def pruning_pep_(self,tree=None,return_subtrees=False):
        '''
        tree:决策树，Tree类型
        return_subtrees:是否返回剪枝过程的子树序列，bool类型
        return 
        0:剪枝后的决策树，Tree类型/剪枝过程的子树序列+节点id，list(Tree,int)类型
        '''    
        if type(tree)==type(None):
            tree=self.tree.copy()
        else:
            tree=tree.copy()
        subtrees,cut_ids=[tree.copy()],[-1]
        #自上而下遍历每个节点
        for i in range(1,len(tree.nodes)):
            if i>=len(tree.nodes):
                break
            node=tree.nodes[i]
            if node.is_leaf==False:
                #计算剪枝前的误差均值和标准差
                E=self.err_cost_(a=0.5,start_id=node.nid,tree=tree)
                S_E=np.sqrt(E*(node.sample_n-E)/node.sample_n)
                #计算剪枝后的误差均值
                E_=self.err_cost_(a=0.5,start_id=node.nid,tree=tree,after_prun=True)
                #剪枝后的误差均值变化在一个标准差内
                if E_<=E+S_E:
                    tree.cut(node.nid)
                    subtrees.append(tree.copy())
                    cut_ids.append(node.nid)
        if return_subtrees==True:
            return subtrees,cut_ids
        else:
            return tree
        
    #代价复杂度剪枝 Cost-Complexity Pruning  
    def pruning_ccp_(self,test_X,test_y,tree=None,return_subtrees=False):
        '''
        test_X: 测试数据集全部特征列，DataFrame类型
        test_y: 测试数据集全部观测值列，Series类型
        tree: 决策树，Tree类型
        return_subtrees: 是否返回剪枝过程的子树序列，bool类型
        return
        0: 剪枝后的决策树，Tree类型/剪枝过程的子树序列+准确率+节点id，list(Tree,int,int)类型
        '''     
        if type(tree)==type(None):
            tree=self.tree.copy()
        else:
            tree=tree.copy()
        #计算完全树在测试集上的准确率
        p_y=self.predict(test_X,tree,show_time=False)
        best_score=self.assess(test_y,p_y,tree.mode)
        best_tree=tree.copy()
        subtrees,scores,cut_ids=[tree.copy()],[best_score],[-1]
        #从完全树开始迭代剪枝至只剩根节点
        while len(tree.nodes)>1:
            a_min,best_cut,leaf_reduce=np.inf,0,0
            #每次迭代选择表面误差增益值最小的节点剪枝,将剪枝后的子树用于下一次迭代
            for i in range(0,len(tree.nodes)):
                node=tree.nodes[i]
                if node.is_leaf==False:
                    leafNum=tree.get_leaf_num(node.nid)
                    #此处采用最简单的误差代价，也可以使用sum(sample_n*gini or entropy)
                    cost=self.err_cost_(start_id=node.nid,tree=tree)
                    cost_=self.err_cost_(start_id=node.nid,tree=tree,after_prun=True)
                    #表面误差率增益值a=误差代价增加量/叶节点数减少量
                    #注：分子要小(误差增加少)，分母要大(复杂度减少多)，即a要最小
                    a=(cost_-cost)/(leafNum-1)
                    if a<a_min:
                        a_min,best_cut,leaf_reduce=a,node.nid,leafNum-1
                    elif a==a_min:
                        if leafNum-1>leaf_reduce:
                            best_cut,leaf_reduce=node.nid,leafNum-1
            #当前迭代最优子树
            tree.cut(best_cut)
            p_y=self.predict(test_X,tree,show_time=False)
            score=self.assess(test_y,p_y,tree.mode)
            subtrees.append(tree.copy())
            scores.append(score)
            cut_ids.append(best_cut)
            #选择在测试集上表现最好的子树
            if score>=best_score:
                best_score=score
                best_tree=tree.copy()
        if return_subtrees==True:
            return subtrees,scores,cut_ids
        else:
            return best_tree
                
    #剪枝
    def pruning(self,test_X=None,test_y=None,tree=None,mode='ccp',
                return_tree=False,show_time=False,check_input=True):
        '''\n
        Function: 对输入树进行剪枝
        
        Note: 部分算法需要输入测试数据
        
        Parameters
        ----------
        test_X: 测试数据集全部特征列，DataFrame类型，默认值None
        test_y: 测试数据集全部观测值列，Series类型，默认值None
        tree: 决策树，Tree类型，默认调用内部缓存的树
        mode: 模式，str类型，默认值'ccp'，
              'rep'->降低错误率剪枝
              'pep'->悲观剪枝
              'ccp'->代价复杂度剪枝
        return_tree: 是否直接返回树而不替换内部缓存的树，bool类型，默认值False
        show_time: 是否显示耗时，bool类型，默认值False
        check_input: 是否进行输入校验，bool类型，默认值True
        ----------
        
        Returns
        -------
        0: 剪枝后的决策树，Tree类型
        -------
        '''
        start = time.clock()
        if type(tree)==type(None):
            tree=self.tree
        #参数校验
        check_type('check_input',type(check_input),type(True))
        if check_input==True:
            check_type('tree',type(tree),type(Tree()))
            mode_list=['rep','pep','ccp']
            check_type('mode',type(mode),type(''))
            mode=mode.lower()
            check_limit('mode',mode in mode_list,str(mode_list))
            if mode in ['rep','ccp']:
                #校验输入
                test_X,continuity_X=self.check_input_X_(test_X,'test_X')
                test_y,continuity_y=self.check_input_y_(test_y,'test_y')
                #校验X,y输入是否匹配
                check_index_match(test_X,test_y,'test_X','test_y')
                check_feats_match(test_X.columns,tree.features,'test_X','tree',mode='right')
        #剪枝
        if mode=='rep':
            tree_=self.pruning_rep_(test_X,test_y,tree)
        elif mode=='pep':
            tree_=self.pruning_pep_(tree)
        elif mode=='ccp':
            tree_=self.pruning_ccp_(test_X,test_y,tree)
        end = time.clock()
        if show_time==True:
            print('\ntime used for pruning:%f'%(end-start))
        #是否以返回值形式给到剪枝后的树    
        if return_tree==False:
            self.tree=tree_
        else:
            return tree_
    
    #打印结点信息
    def print_nodes(self,tree=None):
        '''\n
        Function: 打印结点信息
                
        Parameters
        ----------
        tree: 决策树，Tree类型，默认调用内部缓存的树
        ----------
        
        Print
        -------
        0: 结点信息
        -------
        '''
        if type(tree)==type(None):
            tree=self.tree
        else:
            check_type('tree',type(tree),type(Tree()))
        print('\n[Nodes Info]')
        tree.print_nodes()
    
    #保存树
    def save_tree(self,file_path,tree=None):
        '''\n
        Function: 保存树

        Parameters
        ----------
        tree: 决策树，Tree类型
        file_path: 保存文件的路径，str类型
        ----------
        '''
        if type(tree)==type(None):
            tree=self.tree
        else:
            check_type('tree',type(tree),type(Tree()))
        check_type('file_path',type(file_path),type(''))
        tree.to_dataframe().to_csv(file_path,encoding='utf-8',index=False)
    
    #读取树
    def read_tree(self,file_path,output=False):
        '''\n
        Function: 读取树结构
                
        Parameters
        ----------
        file_path: 文件的路径，str类型
        output: 是否返回读取到的树，bool类型，默认值False
        ----------
        
        Returns
        -------
        0: 读取到的树，Tree类型
        -------
        '''
        check_type('file_path',type(file_path),type(''))
        df=pd.read_csv(file_path,encoding='utf-8')
        if output==True:
            return Tree(df)
        else:
            self.tree=Tree(df)
        
    #计算树的叶节点数
    def get_leaf_num(self,start_id=0,tree=None):
        '''\n
        Function: 计算树的叶节点数
                
        Parameters
        ----------
        start_id: 起始的节点id，int类型，默认值0
        tree: 决策树，Tree类型，默认调用内部缓存的树
        ----------
        
        Returns
        -------
        0: 叶节点数量，int类型
        -------
        '''
        if type(tree)==type(None):
            tree=self.tree
        else:
            check_type('tree',type(tree),type(Tree()))
        check_type('start_id',type(start_id),type(0))
        leafNum=tree.get_leaf_num(start_id)
        return leafNum
    
    #计算树的深度
    def get_depth(self,start_id=0,tree=None):
        '''\n
        Function: 计算树的深度
                
        Parameters
        ----------
        start_id: 起始的节点id，int类型，默认值0
        tree: 决策树，Tree类型，默认调用内部缓存的树
        ----------
        
        Returns
        -------
        0: 树的深度，int类型
        -------
        '''
        if type(tree)==type(None):
            tree=self.tree
        else:
            check_type('tree',type(tree),type(Tree()))
        check_type('start_id',type(start_id),type(0))
        depth_max=tree.get_depth(start_id)
        return depth_max
    
    #注：可视化用于展示复杂的树会看不清
    #定义可视化格式
    style_inNode = dict(boxstyle="round4", color='#3366FF')  # 定义中间判断结点形态
    style_leafNode = dict(boxstyle="circle", color='#FF6633')  # 定义叶结点形态
    style_arrow_args = dict(arrowstyle="<-", color='g')  # 定义箭头
        
    #选择概率最高的类作为叶节点判定的类
    def choose_class__(self,proba_dict):
        '''
        proba_dict: 分类概率，dict类型
        return
        0: 选择的类，str类型
        '''
        class_,proba_max='',0.0
        for key in proba_dict.keys():
            if proba_dict[key]>proba_max:
                proba_max=proba_dict[key]
                class_=key
        return class_
    
    #绘制带箭头的注释
    def plot_node_(self,node_text,location,p_location,node_type,first):
        '''
        node_text: 节点上的文字，str类型
        location: 中心点坐标，tuple(float)类型
        p_location: 父节点坐标，tuple(float)类型
        node_type: 节点类型，预定义的dict类型
        first: 根节点标志位，bool类型
        '''
        if first==True:
            self.ax1.annotate(node_text,xy=p_location,xycoords='axes fraction',
                     xytext=location,textcoords='axes fraction',
                     va="center",ha="center",bbox=node_type)
        else:
            self.ax1.annotate(node_text,xy=p_location, xycoords='axes fraction',
                     xytext=location,textcoords='axes fraction',
                     va="center",ha="center",bbox=node_type, 
                     arrowprops=self.style_arrow_args)
    
    #在父子结点间填充文本信息
    def plot_mid_text_(self,location,p_location,text):
        '''
        location: 中心点坐标，tuple(float)类型
        p_location: 父节点坐标，tuple(float)类型
        text: 文本，str类型
        '''
        xMid=(p_location[0]-location[0])/2.0+location[0]
        yMid=(p_location[1]-location[1])/2.0+location[1]
        self.ax1.text(xMid,yMid,text,va="center",ha="center",rotation=30)
    
    #绘制树_遍历节点
    def plot_(self,start_id,tree):
        '''
        start_id: 开始节点，int类型
        print_loc: 打印节点位置信息，bool类型
        tree: 决策树，Tree类型
        '''
        node,iloc=tree.find_node(start_id,return_iloc=True)
        #总宽度，总高度，x偏移，y偏移，当前深度
        totalW=float(self.get_leaf_num(start_id,tree=tree))-1
        totalD=float(self.get_depth(start_id,tree=tree))+0.1
        if totalW==0:
            if tree.mode=='c':
                output=self.choose_class__(node.output)
            else:
                output=str(node.output)
            self.plot_node_(output,(0.5,0.5),(0.5,0.5),self.style_leafNode,True)
            return
        xOff,yOff=-1.0/totalW,1.0
        depth=tree.nodes[start_id].depth
        pLocation_=(xOff,yOff)
        #队列：待处理的节点，父节点坐标，父节点下属叶节点数
        queue=[(node,(xOff,yOff),0)]
        while len(queue)>0:
            node,pLocation,pLeafNum=queue.pop(0)
            #获取当前节点下属叶节点数
            leafNum=self.get_leaf_num(node.nid,tree=tree)
            #绘制方式是逐层绘制，深度变化时调整y偏移，父节点变化时根据父节点调整x偏移
            if node.depth>depth:
                depth=node.depth
                yOff=yOff-1.0/totalD
            if pLocation!=pLocation_:
                pLocation_=pLocation
                xOff=pLocation[0]-(1.0+float(pLeafNum))/2.0/totalW
            #首个节点不需要绘制箭头
            if node.nid==start_id:
                first,mid_text=True,''
            else:
                first=False
                mid_text=node.limit+' '+str(node.value)
            #叶节点/内结点绘制
            if node.is_leaf==True:
                #调整x偏移（每个叶节点的偏移量统一）
                xOff=xOff+1.0/totalW
                #选择概率最大的分类/显示回归值
                if tree.mode=='c':
                    output=self.choose_class__(node.output)
                else:
                    int_len=len(str(int(node.output)))
                    dec_len=3-int_len
                    if dec_len<0:
                        dec_len=0
                    output=str(round(node.output,dec_len))
                #绘制当前节点和指向其的箭头
                self.plot_node_(output,(xOff, yOff), 
                                pLocation,self.style_leafNode,False)
                #显示箭头上的文字
                self.plot_mid_text_((xOff,yOff),pLocation,mid_text)
                #打印节点坐标
                '''
                print('<leafNode id=%d depth=%d text=%s> x:%f y:%f'%
                      (node.nid,depth,class_,xOff,yOff))
                '''
            else:
                #内节点显示分裂特征名
                bestFeature=node.childs_feature
                #调整x偏移（每个内结点偏移量由下属叶节点数决定）
                xOff=xOff+(1.0+float(leafNum))/2.0/totalW
                location=(xOff,yOff)
                #绘制节点和箭头
                self.plot_node_(bestFeature,location,pLocation, 
                                self.style_inNode,first)
                self.plot_mid_text_(location,pLocation,mid_text)
                #打印节点坐标
                '''
                print('<inNode id=%d depth=%d text=%s> x:%f y:%f'%
                      (node.nid,depth,bestFeature,xOff,yOff))
                '''
                #再次调整x偏移（上一次偏移得到的是中心位置）
                xOff=xOff+((1.0+float(leafNum))/2.0-1.0)/totalW
                #子结点加入队列
                for childId in node.childs: 
                    child,iloc=tree.find_node(childId,start=iloc+1,return_iloc=True)
                    queue.append((child,location,leafNum))
    
    #绘制树
    def plot(self,start_id=0,tree=None):
        '''\n
        Function: 绘制树
   
        Parameters
        ----------
        start_id: 开始节点id，int类型，默认值0
        tree: 决策树，Tree类型，默认调用内部缓存的树
        ----------
        
        Print
        -------
        基于matplotlib的决策树绘图
        -------
        '''
        if type(tree)==type(None):
            tree=self.tree
        else:
            check_type('tree',type(tree),type(Tree()))
        check_type('start_id',type(start_id),type(0))
        plt.rcParams['font.sans-serif']=['SimHei']
        plt.rcParams['axes.unicode_minus']=False
        fig=plt.figure(1,facecolor='white')
        fig.clf()
        plt.suptitle("[Decision Tree Plot]")
        axprops=dict(xticks=[], yticks=[])
        self.ax1=plt.subplot(111,frameon=False,**axprops)
        self.plot_(start_id,tree)
        plt.show()  