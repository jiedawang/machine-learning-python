# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import numba as nb

#[数据结构]

#节点
class Node:
    #可以通过第一个参数传入series初始化（优先），也可以单独传入各个属性
    '''
    parent:父节点索引，int类型
    sample_n:流至该节点的训练样本数，int类型
    is_leaf:是否是叶节点，bool类型
    feature:内节点参数，用于分裂的特征，str类型
    limit:内节点参数，限制类型（=，<=,>,in），str类型
    value:内节点参数，用于分裂的值，float/str/list(str)类型
    classify:叶节点参数，分类结果，dict(str->float)类型
    '''
    def __init__(self,data=None,parent=-1,sample_n=0,is_leaf=False,feature=None,
                 limit=None,value=None,classify=None):
        if limit not in ['<=','>','=','in',None]:
            raise TypeError('unknown limit')
        if type(data)!=type(None):
            self.load_series(data)
        else:
            self.load(parent,sample_n,is_leaf,feature,limit,value,classify)
     
    #用于设置属性
    '''
    （以下几个属性不在构造节点时，而在构造树时赋值）
    childs:子节点索引列表，list(int)类型
    childs_feature:分裂特征，str类型
    depth:深度，int类型
    idx:该节点索引，在构造树时自动赋值，int类型
    '''
    def load(self,parent,sample_n,is_leaf,feature,limit,value,classify):
        self.parent=parent
        self.childs=[]
        self.childs_feature=None
        self.sample_n=sample_n
        self.is_leaf=is_leaf
        self.feature=feature
        self.limit=limit
        self.value=value
        self.classify=classify
        self.depth=0
        self.idx=0
    
    #通过series加载
    '''
    data:节点信息，Series类型
    '''
    def load_series(self,data):
        #判断输入格式是否正确
        if type(data)!=type(pd.Series()):
            raise TypeError('The input should be series')
        #标签是否对应
        label=Node.info_label()
        if data.index.tolist()!=label:
            raise TypeError('The index do not meet the requirements')
        #加载
        if type(data['classify'])==type(''):
            classify=eval(data['classify'])
        else:
            classify=data['classify']
        self.load(data['parent'],data['sample_n'],data['is_leaf'],
                  data['feature'],data['limit'],data['value'],classify)
     
    #添加子节点信息
    '''
    child:子节点，Node类型
    '''
    def add_child(self,child):
        self.childs.append(child)
      
    #将节点属性转换为字符串
    '''
    return> 0:节点信息，str类型
    '''
    def info_to_str(self):
        if self.is_leaf==False:
            if self.parent==-1:
                return '<inNode Id=%d pId=%d depth=%d> sample_n:%d'\
                    %(self.idx,self.parent,self.depth,self.sample_n)
            else:
                return '|-- %s %s %s'%(self.feature,self.limit,self.value)+\
                    '\n<inNode Id=%d pId=%d depth=%d> sample_n:%d'\
                    %(self.idx,self.parent,self.depth,self.sample_n)
        else:
            return '|-- %s %s %s'%(self.feature,self.limit,self.value)+\
                '\n<leafNode Id=%d pId=%d depth=%d> sample_n:%d classify:%s'\
                %(self.idx,self.parent,self.depth,self.sample_n,str(self.classify))
    
    #将节点属性转换为列表
    '''
    return> 0:节点属性列表，list类型
    '''
    def info_to_list(self):
        return [self.idx,self.depth,self.parent,self.childs,self.sample_n,
                self.is_leaf,self.feature,self.limit,self.value,self.classify]
    
    #节点属性标签
    '''
    return> 0:节点属性标签列表，list(str)类型
    '''
    def info_label():
        return ['idx','depth','parent','childs','sample_n',
                'is_leaf','feature','limit','value','classify']
        
#树
class Tree:
    #可传入dataframe初始化，也可不传入生成一个空的树
    '''
    data:完整树节点信息，DataFrame类型
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
    
    #添加节点，返回新添加节点的索引
    '''
    node:新节点，Node类型
    return> 0:新增节点的索引，int类型
    '''
    def add_node(self,node):
        idx=self.node_count
        node.idx=idx
        if node.parent!=-1:
            node.depth=self.nodes[node.parent].depth+1
            self.nodes[node.parent].add_child(idx)
            self.nodes[node.parent].childs_feature=node.feature
        if node.depth>self.depth:
            self.depth=node.depth
        self.nodes.append(node)
        self.node_count+=1
        return idx
    
    #查找节点
    '''
    node_:节点标识，int/float/Node类型
    return> 0:查找到的节点，Node类型
    '''
    def find_node(self,node_):
        if type(node_)==type(0):
            nodeIdx=node_
        elif type(node_)==type(0.0):
            nodeIdx=int(node_)
        elif type(node_)==type(Node()):
            nodeIdx=node_.idx
        else:
            raise TypeError('Unrecognized dtype')
        if (nodeIdx<0)|(nodeIdx>=len(self.nodes)):
            raise IndexError('Fail to find node')
        return self.nodes[nodeIdx]
    
    #查找某节点的父节点
    '''
    node_:节点标识，int/float/Node类型
    return> 0:父节点，Node类型
    '''
    def get_parent(self,node_):
        node=self.find_node(node_)
        return self.nodes[node.parent]
    
    #查找某节点的所有子节点
    '''
    node_:节点标识，int/float/Node类型
    return> 0:子节点列表，list(Node)类型
    '''
    def get_childs(self,node_):
        node=self.find_node(node_)
        childs=[]
        for idx in node.childs:
            childs.append(self.nodes[idx])
        return childs
    
    #将树结构转化为dataframe
    '''
    return> 0:全部树节点信息，DataFrame类型
    '''
    def to_dataframe(self):
        label=Node.info_label()
        nodes=[]
        for n in self.nodes:
            nodes.append(n.info_to_list())
        df_tree=pd.DataFrame(nodes,columns=label)
        return df_tree
    
    #将dataframe还原为树结构
    '''
    data:全部树节点信息，DataFrame类型
    '''
    def load_dataframe(self,data):
        if type(data)!=type(pd.DataFrame()):
            raise TypeError('The input should be dataframe')
        label=Node.info_label()
        if data.columns.tolist()!=label:
            raise TypeError('The columns do not meet the requirements')
        self.reset()
        for i in range(len(data)):
            dr=data.iloc[i,:]
            self.add_node(Node(dr))
    
    #获取流至该节点的路径 
    '''
    node_:节点标识，int/float/Node类型
    return> 0:流经节点索引列表，list(int)类型
    '''       
    def get_path(self,node_):
        node=self.find_node(node_)
        path=[node.idx]
        while node.idx>0:
            node=self.get_parent(node)
            path.append(node.idx)
        path.reverse()
        return path
    
    #打印全部节点信息
    def print_nodes(self):
        for i in range(self.node_count):
            print(self.nodes[i].info_to_str())     
            
    #计算树的叶节点数
    '''
    start_id:起始的节点id，int类型
    return> 0:叶节点数量，int类型
    '''
    def get_leaf_num(self,start_id=0):
        leafNum=0
        queue=[self.nodes[start_id]]
        while len(queue)>0:
            node=queue.pop(0)
            if node.is_leaf==True:
                leafNum+=1
            else:
                for childIdx in node.childs:
                    queue.append(self.nodes[childIdx])
        return leafNum
    
    #计算树的深度
    '''
    start_id:起始的节点id，int类型
    return> 0:树的深度，int类型
    '''
    def get_depth(self,start_id=0):
        depth_max=0
        depth_start=self.nodes[start_id].depth
        queue=[self.nodes[start_id]]
        while len(queue)>0:
            node=queue.pop(0)
            if node.is_leaf==True:
                depth=node.depth-depth_start
                if depth>depth_max:
                    depth_max=depth
            else:
                for childIdx in node.childs:
                    queue.append(self.nodes[childIdx])
        return depth_max

#[运算函数]

#使用numba加速运算，第一次运行时需要一些时间编译，
#且只能接收Numpy数组，对于pandas的数据对象可通过values属性获取

#去重统计
'''
array:需要去重的数据列，narray类型
return> 0:去重后的数据列，narray类型
        1:数量统计，narray类型
'''
@nb.jit(nopython=True,cache=True)
def unique_count(array):
    #排序
    array_=np.sort(array)
    #初始化值变量：取值列表/当前取值/计数列表/当前计数
    values,value=[array_[0]],array_[0]
    counts,count=[],0
    #遍历数据列，对每个取值计数
    for i in range(len(array_)):
        if array_[i]==value:
            count+=1
        else:
            values.append(array_[i])
            counts.append(count)
            value,count=array_[i],1
    counts.append(count)
    return np.array(values),np.array(counts)

#列表元素查找
'''
array:需要查找的数据列，narray类型
values:查找的元素，list类型
return> 0:布尔索引，narray类型
'''
@nb.jit(nopython=True,cache=True)
def isin(array,values):
    #数据列长度，取值列表长度
    n,m=len(array),len(values)
    #用于过滤结果的布尔索引
    filterIdx=np.array([False for i in range(n)])
    #遍历数据列，将匹配到的行的布尔索引更改
    for i in range(n):
        for j in range(m):
            if array[i]==values[j]:
                filterIdx[i]=True
    return filterIdx

#组合枚举(numba的nopython不支持递归和嵌套list）
#注：剔除了空集和全集
'''
values:离散取值，list类型
split_mode:分割模式，求离散取值分割到两个子集中的所有组合，只返回一个子集的结果，bool类型
return> 0:组合的结果，二维narray类型，01表示是否取，行对应每种组合，列对应各个value，narray类型
'''
@nb.jit(nopython=True,cache=True)
def combine_enum(values,split_mode=True):
    #取值列表长度
    vl_count=len(values)
    #取值少于2没有除空集和全集以外的组合
    if vl_count<=1:
        return None
    #将每个取值是否取用以01标志位表示，每一种组合为一行
    #结果矩阵行数对应组合种数，等于2的vl_count次方-2，列数等于vl_count
    #二分裂模式只需要考虑前面一半的组合，后半部分与前面是对称的
    if split_mode==True:
        cb_count=2**(vl_count-1)-1
    else:
        cb_count=2**vl_count-2
    result=np.zeros((cb_count,vl_count))
    #用二进制数遍历的方式获得全部01组合
    for i in range(1,cb_count+1):
        byte_size,j=0,i
        while j>1:
            if j%2==1:
                result[i-1][byte_size]=1
            j=j//2
            byte_size+=1
        if j==1:
            result[i-1][byte_size]=1
    return result

#组合抽取
'''
values:离散取值，list类型
take_array:取值标识，narray类型，可对上面一个方法的返回值进行切片得到
return> 0:组合值列表，list类型
'''
@nb.jit(nopython=True,cache=True)
def combine_take(values,take_array):
    combine=[]
    for i in range(len(take_array)):
        if take_array[i]==1:
            combine.append(values[i])
    return combine

'''
#递归+嵌套list，不能用numba,性能大概是上面的1/30
def combine_enum(values,split_mode=True,n=-1):
    if n==-1:
        if split_mode==True:
            n=len(values)//2+len(values)%2
        else:
            n=len(values)
    result=[]
    for i in range(len(values)):
        result.append([values[i]])
        if n>1:
            values_=values[i+1:].copy()
            combine_=combine_enum(values_,split_mode,n-1)
            for j in range(len(combine_)):
                result.append([values[i]]+combine_[j])
    return result
'''

#信息熵etp=sum(p*log2(p))
#p=每个取值value的占比
'''
array:需要求熵的数据列，narray类型
continuous:连续性，bool类型
value:分裂点，float类型，只有在处理连续数据时有意义
return> 0:熵，float类型
'''
@nb.jit(nopython=True,cache=True)
def entropy(array,continuous=False,value=0):
    #数据集大小,初始化熵
    n,etp=len(array),0.0
    #是否是连续数据,连续数据按指定阈值分裂成两部分,离散数据按每个取值分裂
    if continuous==True:
        #统计左半部分大小，由于只有两个子集，另一半不需要另外统计
        count=0       
        for i in range(n):
            if array[i]<=value:
                count+=1
        p=count/n
        etp=-p*np.log2(p)-(1-p)*np.log2(1-p)
    else:
        #去重并统计
        values,counts=unique_count(array)
        #遍历每一个值计算
        for i in range(len(values)):
            p=counts[i]/n
            etp-=p*np.log2(p)
    return etp

#条件熵con_etp=sum(p*etp)
#p,etp=x的各个取值的数量占比以及按x的值分裂后每个子集y的熵
'''
x:用于分裂的特征列，narray类型
y:分类列，narray类型
continuous:连续性，bool类型
value:分裂点，float类型,只有在处理连续数据时有意义
return> 0:条件熵，float类型
'''
@nb.jit(nopython=True,cache=True)
def con_entropy(x,y,continuous=False,value=0):
    n=len(x)
    #连续特征和离散特征采用不同计算方式
    if continuous==True:
        boolIdx=(x<=value)
        p=len(x[boolIdx])/n
        con_ent=p*entropy(y[boolIdx])\
            +(1-p)*entropy(y[~boolIdx])
    else:
        #去重并统计
        values,counts=unique_count(x)
        con_ent=0.0
        #遍历每个取值
        for i in range(len(values)):
            y_=y[x==values[i]]
            p=counts[i]/n
            con_ent+=p*entropy(y_)
    return con_ent

#最优分裂点选择
'''
x:用于分裂的特征列，连续变量，narray类型
y:分类列，narray类型
return> 0:最优分裂点，float类型 
        1:所有可能分裂点的数量，int类型
'''
@nb.jit(nopython=True,cache=True)
def choose_split_value(x,y):
    #计算分裂前的信息熵
    baseEntropy=entropy(y)
    #需要尝试的分裂点
    values,n=filter_split_values(x,y)
    #初始化变量
    bestInfGain=0.0
    bestSplitValue=values[0]
    #逐个计算所有可能分裂点的条件熵
    for j in range(len(values)-1):
        split_value=values[j]
        infGain=baseEntropy-con_entropy(x,y,True,split_value)
        if infGain>bestInfGain:
            bestInfGain=infGain
            bestSplitValue=split_value
    return bestSplitValue,n

#筛选分裂点
'''
x:用于分裂的特征列，narray类型
y:分类列，narray类型
return> 0:经过筛选的分裂点集，list(float)类型
        1:所有可能分裂点的数量，int类型
'''
@nb.jit(nopython=True,cache=True)
def filter_split_values(x,y):
    n=len(x)
    #将x,y按x升序排序
    sortIdx=np.argsort(x)
    x,y=x[sortIdx],y[sortIdx]
    #需要选取的点的布尔索引(因为不能直接初始化bool类型所以用int代替)
    filterIdx=np.zeros(n)
    filterIdx[0],filterIdx[n-1]=1,1
    #将分类结果y有变化的位置选取出来
    for i in range(n-1):
        if y[i]!=y[i+1]:
            filterIdx[i],filterIdx[i+1]=1,1
    values,counts=unique_count(x[filterIdx==1])
    return values,n

#基尼指数gini=1-sum(p*p)
#p=每个取值value的占比
#基尼指数的计算一般只针对离散分类，不用于特征
'''
array:需要求基尼指数的数据列,narray类型
return> 0:基尼指数，float类型
'''
@nb.jit(nopython=True,cache=True)
def gini(array):
    #数据集大小,初始化基尼指数
    n,g=len(array),1.0
    #先排序和初始化变量
    values,counts=unique_count(array)
    #遍历每一个元素
    for i in range(len(values)):
        p=counts[i]/n
        g-=p*p
    return g

#条件基尼
'''
array:需要求基尼指数的数据列,narray类型
return> 0:基尼指数，float类型
'''
@nb.jit(nopython=True,cache=True)
def con_gini(x,y,continuous=False,value=[]):
    n,con_gini=len(x),0.0
    #连续特征和离散特征采用不同计算方式
    if continuous==True:
        boolIdx=(x<=value[0])
    else:
        boolIdx=isin(x,value)
    p=len(x[boolIdx])/n
    con_gini=p*gini(y[boolIdx])\
        +(1-p)*gini(y[~boolIdx])
    return con_gini

#最优离散值组合选择
'''
x:用于分裂的特征列,Series类型
y:分类列,Series类型
return> 0:最优分裂组合左子集，list类型 
        1:右子集，list类型
'''
@nb.jit(nopython=True,cache=True)
def choose_value_combine(x,y):
    #需要尝试的分裂组合
    values,counts=unique_count(x)
    if len(values)>1:
        combines=combine_enum(values)
        #初始化变量
        minGini=1.0
        bestCombineIdx=-1
        #逐个计算所有可能分裂方式的基尼系数
        for i in range(len(combines)):
            combine=combine_take(values,combines[i])
            gini=con_gini(x,y,False,combine)
            if gini<minGini:
                minGini=gini
                bestCombineIdx=i
        #选择失败
        if bestCombineIdx==-1:
            return None
        #生成左右子集取值列表
        bestCombine=combines[bestCombineIdx]
        bestCombineLeft,bestCombineRight=[],[]
        for j in range(len(values)):
            if bestCombine[j]==1:
                bestCombineLeft.append(values[j])
            else:
                bestCombineRight.append(values[j])
        return bestCombineLeft,bestCombineRight
    else:
        return [0],[0]

#[构造类]

class DecisionTree:
    
    #信息熵,可以用于求类别的熵，也可以用于求特征的熵,只能计算单列
    #表示随机变量不确定性的度量，范围0~log2(n)，数值越大不确定性越大,n为离散值种类数
    #0log0=0 ；当对数的底为2时，熵的单位为bit；为e时，单位为nat。
    '''
    def entropy(self,info,continuous=False,value=0):
        n=len(info)
        if continuous==True:
            #计算值的概率分布
            p=len(info[info<=value])/n
            #计算信息熵
            etp=-p*np.log2(p)-(1-p)*np.log2(1-p)
        else:
            #计算值的概率分布
            values_count=info.groupby(info).count()
            p=values_count/n
            #计算信息熵
            etp=-np.sum(p*np.log2(p))
        return etp
    '''
    
    #条件熵
    #在x中第i个随机变量确定的情况下，随机变量y的不确定性
    #即按第i个特征划分数据后的信息熵
    '''
    def con_entropy(self,x,y,continuous=False,value=0):
        n=len(x)
        #计算条件熵
        con_ent=0.0
        if continuous==True:
            boolIdx=(x<=value)
            p=len(x[boolIdx])/n
            con_ent+=p*self.entropy(y[boolIdx])
            con_ent+=(1-p)*self.entropy(y[~boolIdx])
        else:
            values=x.drop_duplicates().tolist()
            for i in range(len(values)):
                boolIdx=(x==values[i])
                p=len(x[boolIdx])/n
                con_ent+=p*self.entropy(y[boolIdx])
        return con_ent
        
    (另一种写法,使用pandas进行多级聚合统计)
    def con_entropy(self,x,y,continuous=False,value=0):
        #如果x是连续值，将x转化为关于分裂点的布尔索引
        if continuous==True:
            x=(x<=value)
        #根据划分特征和分类统计数量
        values_count=y.groupby([x,y]).size()
        #单独提出划分特征的取值计数
        split_values_count=values_count.sum(level=x.name)
        #每个特征取值中各个分类出现的概率
        p_y=values_count/split_values_count
        #统计每个特征取值下子集的熵
        etp=(-p_y*np.log2(p_y)).sum(level=x.name)
        #统计不同特征取值出现的概率
        p_x=split_values_count/split_values_count.sum()
        #计算条件熵
        con_ent=np.dot(p_x.T,etp)
        return con_ent
    '''
    
    #筛选分裂点，取分类结果有变化的点
    '''
    def filterSplitValues(self,x,y):
        #重整源数据
        data=pd.DataFrame()
        data['x'],data['y']=x,y
        data=data.sort_values('x')
        data.index=np.linspace(0,len(data)-1,len(data)).astype(np.int64)
        #提取首末行
        head,foot=data.iloc[0:1,:],data.iloc[len(data)-1:,:]
        #复制一份作为后续数
        subsequent=data.copy()
        subsequent.index=subsequent.index-1
        subsequent.columns=subsequent.columns+'_next'  
        #每个数与自己的后续数匹配并找出变化位置
        data=data.join(subsequent,how='inner')   
        change_points=data[data['y']!=data['y_next']]  
        #提取变化位置前后的点 
        change_f=change_points.iloc[:,0:2]
        change_b=change_points.iloc[:,2:]
        change_b.columns=change_f.columns
        #将所有的检查点整合并去重排序
        check_points=pd.concat([head,change_f,change_b,foot])
        check_points=check_points.drop_duplicates().sort_index().iloc[:,0].tolist()
        return check_points
    '''
       
    #最优特征选择(ID3)
    #选择依据：信息增益
    #=划分前类别的信息熵-划分后类别的条件熵
    #用于衡量经过某特征的划分后分类的不确定性降低了多少
    '''
    X:所有参与选择的特征列，DataFrame类型
    y:分类列，Series类型
    return> 0:最优分裂特征的索引，int类型
    '''
    def choose_feature_by_id3(self,X,y):
        #计算分割前的信息熵
        baseEntropy=entropy(y.values)
        #初始化变量
        bestInfGain=0.0
        bestFeatureIdx=-1
        #逐个计算按不同特征分割后的信息增益并选出增益最大的一个特征
        for i in range(len(X.columns)):
            infGain=baseEntropy-con_entropy(X.iloc[:,i].values,y.values)
            if infGain>bestInfGain:
                bestInfGain=infGain
                bestFeatureIdx=i
        return bestFeatureIdx
    
    #最优特征选择(C4.5)
    #选择依据：信息增益比
    #=信息增益/划分特征的信息熵
    #避免因为特征取值多而导致信息增益偏大
    #C4.5增加了对连续数据的处理，连续特征根据信息增益选择最佳分裂点转换为离散值
    '''
    X:所有参与选择的特征列，DataFrame类型
    y:分类列，Series类型
    continuous:连续性，list(bool)类型
    return> 0:最优分裂特征的索引 ，int类型
            1：最优分裂值，float类型
    '''
    def choose_feature_by_c45(self,X,y,continuous):
        #计算分割前的信息熵
        baseEntropy=entropy(y.values)
        #初始化变量
        bestInfGainRatio=0.0
        bestFeatureIdx=-1
        bestSplitValue=0.0
        #逐个计算按不同特征分割后的信息增益并选出增益最大的一个特征
        for i in range(len(X.columns)):
            x=X.iloc[:,i]
            #是否为连续特征
            if continuous[i]==True:
                splitValue,n=choose_split_value(x.values,y.values)
                splitFeatEntropy=entropy(x.values,True,splitValue)
                infGain=baseEntropy\
                    -con_entropy(x.values,y.values,True,splitValue)
                    #-np.log2(n-1)/len(x)
            else:
                splitValue=0.0
                splitFeatEntropy=entropy(x.values)
                infGain=baseEntropy-con_entropy(x.values,y.values)
            if splitFeatEntropy==0:
                continue
            infGainRatio=infGain/splitFeatEntropy
            if infGainRatio>bestInfGainRatio:
                bestInfGainRatio=infGainRatio
                bestFeatureIdx=i
                bestSplitValue=splitValue
        return bestFeatureIdx,bestSplitValue
       
    #最优特征选择(CART分类树)
    #选择依据：基尼指数
    #分类树输出离散数值
    #cart和c4.5采用同样的方式处理连续特征
    #同时cart对离散特征也进行二分处理，将离散值划分到两个子集里，选择最优的一种分裂
    '''
    X:所有参与选择的特征列，DataFrame类型
    y:分类列，Series类型
    continuous:连续性，list(bool)类型
    return> 0:最优分裂特征的索引，int类型 
            1:最优分裂方案，float或list(list(int))类型
    '''
    def choose_feature_by_cart_c(self,X,y,continuous):
        #初始化变量
        bestGini=1.0
        bestFeatureIdx=-1
        bestSplit=None
        #逐个计算按不同特征分割后的信息增益并选出增益最大的一个特征
        for i in range(len(X.columns)):
            x=X.iloc[:,i]
            #是否为连续特征
            if continuous[i]==True:
                splitValue,n=choose_split_value(x.values,y.values)
                gini=con_gini(x.values,y.values,True,[splitValue])
                temp=splitValue
            else:
                combineLeft,combineRight=choose_value_combine(x.values,y.values)
                if combineLeft!=combineRight:
                    gini=con_gini(x.values,y.values,False,combineLeft)
                    temp=[combineLeft,combineRight]
                else:
                    continue
            if gini<bestGini:
                bestGini=gini
                bestFeatureIdx=i
                bestSplit=temp
        return bestFeatureIdx,bestSplit
    
    #计算每个类的概率
    '''
    y:分类列，Series类型
    return> 0:分类概率，dict类型
    '''
    def compute_proba(self,y):
        proba={}
        values_count=y.groupby(y).count()
        total_count=values_count.sum()
        for i in range(len(values_count)):
            p=values_count.iloc[i]/total_count
            value=self.get_ylabel(values_count.index[i])
            proba[value]=p
        return proba   
          
    #选择概率最高的类作为叶节点判定的类,用于预测
    '''
    p_y_:预测的分类概率，DataFrame类型
    return> 0:选择的类，Series类型
    '''
    def choose_class(self,p_y_):
        classify=self.classify
        p_max=p_y_.max(axis=1)
        p_y=pd.Series(np.full(len(p_y_),''),index=p_y_.index)
        for i in range(len(classify)):
            p_y[p_y_.iloc[:,i]==p_max]=classify[i]
        #按类别分布情况加权随机填充未能分类的记录
        nullIdx=(p_y=='')
        n=p_y[nullIdx].count()
        if n>0:
            p_y.loc[nullIdx]=p_y[~nullIdx].sample(n=n,replace=True).tolist()
        return p_y
    
    #校验输入数据类型并返回X的离散性默认判定
    '''
    X:所有的特征列，DataFrame类型
    y:分类列，Series类型
    model_type:模型算法类型，str类型,id3/c4.5/cart三种
    return> 0:连续性判定，list(bool)类型
    '''
    def check_input(self,X,y,model_type):
        type_list=('id3','c4.5','cart')
        if model_type.lower() not in type_list:
            print('model_type should in:')
            print(type_list)
            raise TypeError('Unknown type')
        if type(X)!=type(pd.DataFrame()):
            raise TypeError('X should be dataframe')
        if type(y)!=type(pd.Series()):
            raise TypeError('y should be series')
        #离散性默认设置
        continuous=[]
        for dtype in X.dtypes:
            if str(dtype) in ['object','category','bool']:
                continuous.append(False)
            else:
                continuous.append(True)
        #ID3不支持连续特征
        if model_type=='ID3':
            if True in continuous:
                raise TypeError('ID3 does not support continuous features')
        return continuous
    
    #将离散变量转化为数值型以支持numba运行
    # 注：离散变量会统一转换为str识别
    '''
    X:所有的特征列，DataFrame类型
    continuous:连续性，list(bool)类型
    return> 0:转化后的X，DataFrame类型 
            1:映射关系，DataFrame类型
    '''
    def format_X(self,X,continuous):  
        mapping_list=[]
        X_=X.copy()
        for i in range(len(continuous)):
            if continuous[i]==False:
                feature_label=X.columns[i]
                values=X.iloc[:,i].sort_values().drop_duplicates().astype('str')
                mapping={label:idx for idx,label in enumerate(values)}
                X_.iloc[:,i]=X.iloc[:,i].astype('str').map(mapping)
                mapping_list+=[[feature_label,idx,label] for idx,label in enumerate(values)]
        return X_,pd.DataFrame(mapping_list,columns=['feature','valueId','label'])
    '''
    y:分类列，Series类型
    return> 0:转化后的y，Series类型
            1:映射关系，DataFrame类型
    '''
    def format_y(self,y):
        y_=y.copy()
        values=y.sort_values().drop_duplicates().astype('str')
        mapping={label:idx for idx,label in enumerate(values)}
        y_=y.astype('str').map(mapping)
        mapping_list=[[idx,label] for idx,label in enumerate(values)]
        return y_,pd.DataFrame(mapping_list,columns=['valueId','label'])
    
    #转换回原来的标签
    '''
    feature: 特征名，即列名，str类型
    valueId: 数值型标签,int类型
    return> 0:原标签，str类型
    '''
    def get_xlabel(self,feature,valueId):
        mapping=self.mapping_X
        boolIdx=(mapping['feature']==feature)&(mapping['valueId']==valueId)
        return mapping['label'][boolIdx].values[0]
    '''
    valueId: 数值型标签，int类型
    return> 0:原标签，str类型
    '''
    def get_ylabel(self,valueId):
        mapping=self.mapping_y
        boolIdx=(mapping['valueId']==valueId)
        return mapping['label'][boolIdx].values[0]
    
    #根据第i列特征分裂数据集
    '''
    X:当前所有的特征列，DataFrame类型
    y:分类列，Series类型
    continuous:连续性，bool类型
    value:分裂依据，二分裂时使用，连续数据->float类型/离散数据->list类型
    return> 0:X分裂后的集合，list(DataFrame) 
            1:y分裂后的集合，list(Series) 
            2:分裂值列表，list
    '''
    def split(self,X,y,i,continuous=False,value=None):
        #抽取第i列特征
        x=X.iloc[:,i]
        featLabel=X.columns[i]
        #连续特征和离散特征采用不同的处理方式
        if continuous==True:
            if type(value)==type(None):
                raise TypeError('must provide split value for continuous feature')
            #根据分裂点将数据集拆分
            values=[('<=',value),('>',value)]
            boolIdx=(x<=value)
            result_X=[X[boolIdx],X[~boolIdx]]
            result_y=[y[boolIdx],y[~boolIdx]]
        else:
            if type(value)==type([]):
                boolIdx=(x.isin(value[0]))
                result_X=[X[boolIdx],X[~boolIdx]]
                result_y=[y[boolIdx],y[~boolIdx]]
                values=[('in',[self.get_xlabel(featLabel,m) for m in value[0]]),
                        ('in',[self.get_xlabel(featLabel,n) for n in value[1]])]              
            else:
                #去重得到特征值列表
                values=x.sort_values().drop_duplicates().tolist()
                #根据不同的特征值进行分割
                result_X,result_y=[],[]
                for j in range(len(values)):
                    result_X.append(X[x==values[j]])
                    result_y.append(y[x==values[j]])
                    values[j]=('=',self.get_xlabel(featLabel,values[j]))
        return result_X,result_y,values
    
    #拟合
    #注：分类会转换为str处理
    '''
    X:所有的特征列，DataFrame类型
    y:分类列，Series类型
    continuous:连续性，list(bool)类型
    model_type:模型算法类型，str类型，id3/c4.5/cart三种,
    depth_max:最大深度，int类型
    output:是否输出拟合好的树，bool类型
    return> 0:决策树，Tree类型
    '''
    def fit(self,X,y,continuous=[],model_type='c4.5',depth_max=10,output=False):
        start = time.clock()
        temp=self.check_input(X,y,model_type)
        self.model_type=model_type.lower()
        self.depth_max=depth_max
        if continuous==[]:
            continuous=temp
        self.continuous=continuous
        self.features=X.columns.tolist()
        X,self.mapping_X=self.format_X(X,continuous)
        self.classify=y.drop_duplicates().astype('str').tolist()
        y,self.mapping_y=self.format_y(y)
        self.tree=self.build(X,y,continuous)
        end = time.clock()
        print('\ntime used for trainning:%f'%(end-start))
        if output==True:
            return self.tree
    
    #构建树
    '''
    X:当前所有的特征列，DataFrame类型
    y:分类列，Series类型
    continuous:连续性，list(bool)类型
    return> 0:决策树，Tree类型
    print> 分裂过程
    '''
    def build(self,X,y,continuous):
        model_type=self.model_type
        deciTree=Tree()
        #等待处理的数据队列：特征，分类，连续性，父节点id，深度，
        #                   分裂特征名,约束方式，分裂值
        queue=[(X,y,continuous,-1,0,None,None,None)]
        while len(queue)>0:
            X_,y_,continuous_,parent,depth,feature,limit,value=queue.pop(0)
            if parent!=-1:
                print('<Split> %s %s %s'%
                     (feature,limit,str(value)))
            print('Current dataset size: %d'%len(y_))
            is_leaf=self.is_leaf_(X_,y_,depth)
            #选择最优特征进行分裂，并记录结果
            if is_leaf==False:
                if model_type=='id3':
                    bestFeatureIdx,bestSplit=self.choose_feature_by_id3(X_,y_),None
                elif model_type=='c4.5':
                    bestFeatureIdx,bestSplit=self.choose_feature_by_c45(X_,y_,continuous)
                elif model_type=='cart':
                    bestFeatureIdx,bestSplit=self.choose_feature_by_cart_c(X_,y_,continuous)
                else:
                    raise TypeError('Unknown type')
                #未能成功选出可供分裂的特征
                if bestFeatureIdx==-1:
                    print('<LeafNode> fail to choose a feature')
                    is_leaf=True
            if is_leaf==True:
                #添加叶节点
                node=Node(parent=parent,sample_n=len(y_),is_leaf=True,feature=feature,
                    limit=limit,value=value,classify=self.compute_proba(y_))
                nodeId=deciTree.add_node(node)
            else:
                #添加内节点
                node=Node(parent=parent,sample_n=len(y_),is_leaf=False,feature=feature,
                     limit=limit,value=value,classify=None)
                nodeId=deciTree.add_node(node)
                #获取最优分裂特征的相关信息
                bestFeatureLabel=X_.columns[bestFeatureIdx]
                #分裂数据集
                splited_X,splited_y,split_values=self.split(X_,y_,
                                                            bestFeatureIdx,
                                                            continuous_[bestFeatureIdx],
                                                            bestSplit)
                continuous__=continuous_.copy()
                continuous__.pop(bestFeatureIdx)
                #将分裂后的数据集加入队列继续处理
                for i in range(len(split_values)):
                    queue.append((splited_X[i].drop(bestFeatureLabel,axis=1),
                                  splited_y[i],continuous__,nodeId,depth+1,
                                  bestFeatureLabel,split_values[i][0],split_values[i][1]))
        return deciTree
    
    #叶节点判断
    '''
    X:当前所有的特征列，DataFrame类型
    y:分类列，Series类型
    depth:当前深度，int类型
    return> 0:是否叶节点，bool类型
    '''
    def is_leaf_(self,X,y,depth):
        #超出高度上限，不继续分裂
        if depth>self.depth_max:
            print('<LeafNode> reach maximum depth')
            return True
        #可用特征不足，不继续分裂
        if len(X.columns)==0:
            print('<LeafNode> lack of feature')
            return True
        #数据集过小，不继续分裂
        if len(X)<2:
            print('<LeafNode> samples too small')
            return True
        #只有一个类，不继续分裂
        if len(y.drop_duplicates())==1:
            print('<LeafNode> only one class')
            return True
        #特征向量统一，不继续分裂
        if len(X.drop_duplicates())==1:
            print('<LeafNode> feature vector unification')
            return True
        return False
    
    #决策路径，只能一次处理一行数据，主要用于校对
    '''
    dr:数据行，Series类型
    tree:决策树，Tree类型
    print> 流经的节点信息
    '''
    def decition_path(self,dr,tree=None):
        if type(tree)==type(None):
            tree=self.tree
        #准备访问的节点
        queue=[tree.nodes[0]]
        print(tree.nodes[0].info_to_str())
        #直到没有等待访问的节点时结束
        while len(queue)>0:
            node=queue.pop(0)
            bestFeature=node.childs_feature
            match_flag=False
            for childIdx in node.childs:
                child=tree.nodes[childIdx]
                if child.limit=='<=':
                    if dr[bestFeature]<=child.value:
                        match_flag=True
                elif child.limit=='>':
                    if dr[bestFeature]>child.value:
                        match_flag=True
                elif child.limit=='in':
                    if dr[bestFeature] in child.value:
                        match_flag=True
                else:
                    if dr[bestFeature]==child.value:
                        match_flag=True
                #打印匹配上的节点信息，如果是内节点再加入待访问队列
                if match_flag==True:
                    print(child.info_to_str())
                    if child.is_leaf==False:
                        queue.append(child)
                    break
        return
    
    #预测
    '''
    X:所有特征列，DataFrame类型
    tree:决策树，Tree类型
    return_proba:是否返回分类概率，bool类型
    return_paths:是否返回决策路径，bool类型
    return> 0:预测的分类/分类概率，Series/DataFrame类型
            1:所有数据最终抵达的节点和决策路径，DataFrame类型
    '''
    def predict(self,X,tree=None,return_proba=False,return_paths=False):
        if type(tree)==type(None):
            tree=self.tree
        start = time.clock()
        n=len(X)
        classify=self.classify
        #数据流，记录已达到节点
        flow=pd.Series(np.zeros(n),index=X.index)
        #定义存放分类结果的series
        p_y_=pd.DataFrame(
                np.zeros(n*len(classify)).reshape(n,len(classify)),
                index=X.index,columns=classify)
        #准备访问的节点
        queue=[tree.nodes[0]]
        #直到没有等待访问的节点时结束
        while len(queue)>0:
            node=queue.pop(0)
            #筛选出流至该节点的数据
            boolIdx=(flow==node.idx)
            X_=X[boolIdx]
            flow_=flow[boolIdx]
            #当前节点是叶节点，返回分类
            if node.is_leaf==True:
                p_y_.update(pd.DataFrame(node.classify,columns=classify,index=X_.index))
            else:
                #当前节点是内节点，遍历每个子节点
                for childIdx in node.childs:
                    child=tree.nodes[childIdx]
                    #不同限制条件采用不同方式处理，更新数据流所到达的节点
                    if child.limit=='<=':
                        flow_[X_[child.feature]<=child.value]=childIdx
                    elif child.limit=='>':
                        flow_[X_[child.feature]>child.value]=childIdx
                    elif child.limit=='in':
                        flow_[X_[child.feature].isin(child.value)]=childIdx
                    else:
                        flow_[X_[child.feature]==child.value]=childIdx
                    flow.update(flow_)
                    #将子节点放入准备访问的队列
                    queue.append(child)
        if return_proba==False:
            p_y_=self.choose_class(p_y_)        
        if return_paths==False:
            end = time.clock()
            print('\ntime used for predict:%f'%(end-start))
            return p_y_
        else:
            paths=pd.DataFrame()
            paths['reach']=flow
            paths['path']=''
            reach_nodes=flow.sort_values().drop_duplicates().astype('int').tolist()
            for nodeIdx in reach_nodes:
                path=tree.get_path(nodeIdx)
                paths.loc[paths['reach']==nodeIdx,'path']=str(path)
            end = time.clock()
            print('\ntime used for predict:%f'%(end-start))
            return p_y_,paths
    
    #评估
    #注：y/p_y的数据类型会转换为str后比较
    '''
    y:实际的分类，Series类型
    p_y:预测的分类，Series类型
    return> 0:准确率，float类型
    '''
    def assess(self,y,p_y):
        #p_y.index=y.index
        cp=pd.DataFrame()
        cp['y'],cp['p']=y.astype('str'),p_y.astype('str')
        accuracy=len(cp[cp['y']==cp['p']])*1.0/len(y)
        return accuracy
    
    #打印结点信息
    '''
    tree:决策树，Tree类型
    print> 节点信息
    '''
    def print_nodes(self,tree=None):
        if type(tree)==type(None):
            tree=self.tree
        print('\n[Nodes Info]')
        self.tree.print_nodes()
    
    #保存树结构
    '''
    tree:决策树，Tree类型
    file_path:保存文件的路径，str类型
    '''
    def save_tree(self,file_path,tree=None):
        if type(tree)==type(None):
            tree=self.tree
        tree.to_dataframe().to_csv(file_path,encoding='utf-8',index=False)
    
    #读取树结构    
    '''
    file_path:文件的路径，str类型
    output:是否返回读取的树，bool类型
    return> 0:决策树，Tree类型
    '''
    def read_tree(self,file_path,output=False):
        df=pd.read_csv(file_path,encoding='utf-8')
        self.tree=Tree(df)
        if output==True:
            return self.tree
        
    #计算树的叶节点数
    '''
    start_id:起始的节点id，int类型
    tree:决策树，Tree类型
    return> 0:叶节点数量，int类型
    '''
    def get_leaf_num(self,start_id=0,tree=None):
        if type(tree)==type(None):
            tree=self.tree
        leafNum=tree.get_leaf_num(start_id)
        return leafNum
    
    #计算树的深度
    '''
    start_id:起始的节点id，int类型
    tree:决策树，Tree类型
    return> 0:树的深度，int类型
    '''
    def get_depth(self,start_id=0,tree=None):
        if type(tree)==type(None):
            tree=self.tree
        depth_max=tree.get_depth(start_id)
        return depth_max

    
    #注：可视化用于展示复杂的树会看不清
    #定义可视化格式
    style_inNode = dict(boxstyle="round4", color='#3366FF')  # 定义中间判断结点形态
    style_leafNode = dict(boxstyle="circle", color='#FF6633')  # 定义叶结点形态
    style_arrow_args = dict(arrowstyle="<-", color='g')  # 定义箭头
        
    #选择概率最高的类作为叶节点判定的类，用于观察树结构
    '''
    proba_dict:分类概率，dict类型
    return> 0:选择的类，str类型
    '''
    def choose_class_(self,proba_dict):
        class_,proba_max='',0.0
        for key in proba_dict.keys():
            if proba_dict[key]>proba_max:
                proba_max=proba_dict[key]
                class_=key
        return class_
    
    #绘制带箭头的注释
    '''
    node_text:节点上的文字，str类型
    location:中心点坐标，tuple(float)类型
    p_location:父节点坐标，tuple(float)类型
    node_type:节点类型，预定义的dict类型
    first:根节点标志位，bool类型
    '''
    def plot_node_(self,node_text,location,p_location,node_type,first):
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
    '''
    location:中心点坐标，tuple(float)类型
    p_location:父节点坐标，tuple(float)类型
    text:文本，str类型
    '''
    def plot_mid_text_(self,location,p_location,text):
        xMid=(p_location[0]-location[0])/2.0+location[0]
        yMid=(p_location[1]-location[1])/2.0+location[1]
        self.ax1.text(xMid,yMid,text,va="center",ha="center",rotation=30)
    
    #绘制树_遍历节点
    '''
    start_id:开始节点，int类型
    print_loc：打印节点位置信息，bool类型
    tree:决策树，Tree类型
    '''
    def plot_(self,start_id,print_loc,tree):
        #总宽度，总高度，x偏移，y偏移，当前深度
        totalW=float(self.get_leaf_num(start_id))-1
        totalD=float(self.get_depth(start_id))+0.1
        xOff,yOff=-1.0/totalW,1.0
        depth=tree.nodes[start_id].depth
        pLocation_=(xOff,yOff)
        #队列：待处理的节点，父节点坐标，父节点下属叶节点数
        queue=[(tree.nodes[start_id],(xOff,yOff),0)]
        while len(queue)>0:
            node,pLocation,pLeafNum=queue.pop(0)
            #获取当前节点下属叶节点数
            leafNum=self.get_leaf_num(node.idx)
            #绘制方式是逐层绘制，深度变化时调整y偏移，父节点变化时根据父节点调整x偏移
            if node.depth>depth:
                depth=node.depth
                yOff=yOff-1.0/totalD
            if pLocation!=pLocation_:
                pLocation_=pLocation
                xOff=pLocation[0]-(1.0+float(pLeafNum))/2.0/totalW
            #首个节点不需要绘制箭头
            if node.idx==start_id:
                first,mid_text=True,''
            else:
                first=False
                mid_text=node.limit+str(node.value)
            #叶节点/内结点绘制
            if node.is_leaf==True:
                #调整x偏移（每个叶节点的偏移量统一）
                xOff=xOff+1.0/totalW
                #选择概率最大的分类
                class_=self.choose_class_(node.classify)
                #绘制当前节点和指向其的箭头
                self.plot_node_(class_,(xOff, yOff), 
                                pLocation,self.style_leafNode,False)
                #显示箭头上的文字
                self.plot_mid_text_((xOff,yOff),pLocation,mid_text)
                #打印节点坐标
                if print_loc==True:
                    print('<leafNode id=%d depth=%d text=%s> x:%f y:%f'%
                          (node.idx,depth,class_,xOff,yOff))
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
                if print_loc==True:
                    print('<inNode id=%d depth=%d text=%s> x:%f y:%f'%
                          (node.idx,depth,bestFeature,xOff,yOff))
                #再次调整x偏移（上一次偏移得到的是中心位置）
                xOff=xOff+((1.0+float(leafNum))/2.0-1.0)/totalW
                #子结点加入队列
                for childIdx in node.childs: 
                    queue.append((tree.nodes[childIdx],location,leafNum))
    
    #绘制树
    '''
    start_id:开始节点，int类型
    print_loc：打印节点位置信息，bool类型
    tree:决策树，Tree类型
    '''
    def plot(self,start_id=0,print_loc=False,tree=None):
        if type(tree)==type(None):
            tree=self.tree
        if (start_id<0)|(start_id>=len(tree.nodes)):
            TypeError('Index out of bounds')
        print('\n[Tree Plot]')
        plt.rcParams['font.sans-serif']=['SimHei']
        plt.rcParams['axes.unicode_minus']=False
        fig=plt.figure(1,facecolor='white')
        fig.clf()
        axprops=dict(xticks=[], yticks=[])
        self.ax1=plt.subplot(111,frameon=False,**axprops)
        self.plot_(start_id,print_loc,tree)
        plt.show()  
    
    
    
    
    
            
            
            