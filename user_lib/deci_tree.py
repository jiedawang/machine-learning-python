# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

class DecisionTree:
  
    #根据第i列特征分割数据集
    def split(self,data,i,continuous=False,value=0):
        #抽取第i列特征
        x=data.iloc[:,i]
        #连续特征和离散特征采用不同的处理方式
        if continuous==True:
            #根据分裂点将数据集拆分
            values=['<=%s'%str(value),'>%s'%str(value)]
            result=[data[x<=value],data[x>value]]
        else:
            #去重得到特征值列表
            values=x.drop_duplicates().sort_values().tolist()
            #根据不同的特征值进行分割
            result=[]
            for i in range(len(values)):
                result.append(data[x==values[i]])
        return result,values
    
    #信息熵,可以用于求类别的熵，也可以用于求特征的熵,只能计算单列
    #表示随机变量不确定性的度量，范围0~log2(n)，数值越大不确定性越大,n为离散值种类数
    #0log0=0 ；当对数的底为2时，熵的单位为bit；为e时，单位为nat。
    def getEntropy(self,info,continuous=False,value=0):
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
    
    #条件熵
    #在x中第i个随机变量确定的情况下，随机变量y的不确定性
    #即按第i个特征划分数据后的信息熵
    def getCondiEntropy(self,x,y,continuous=False,value=0):
        n=len(x)
        #计算条件熵
        con_ent=0.0
        if continuous==True:
            boolIdx=(x<=value)
            p=len(x[boolIdx])/n
            con_ent+=p*self.getEntropy(y[boolIdx])
            con_ent+=(1-p)*self.getEntropy(y[~boolIdx])
        else:
            values=x.drop_duplicates().tolist()
            for i in range(len(values)):
                boolIdx=(x==values[i])
                p=len(x[boolIdx])/n
                con_ent+=p*self.getEntropy(y[boolIdx])
        '''(另一种写法)
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
        '''
        return con_ent

    
    #最优特征选择(ID3)
    #选择依据：信息增益
    #=划分前类别的信息熵-划分后类别的条件熵
    #用于衡量经过某特征的划分后分类的不确定性降低了多少
    def chooseFeatureByID3(self,data):
        #计算分割前的信息熵
        y=data.iloc[:,len(data.columns)-1]
        baseEntropy=self.getEntropy(y)
        #初始化变量
        bestInfGain=0.0
        bestFeatureIdx=-1
        #逐个计算按不同特征分割后的信息增益并选出增益最大的一个特征
        for i in range(len(data.columns)-1):
            infGain=baseEntropy-self.getCondiEntropy(data.iloc[:,i],y)
            if infGain>bestInfGain:
                bestInfGain=infGain
                bestFeatureIdx=i
        return bestFeatureIdx
    
    #最优特征选择(C4.5)
    #选择依据：信息增益比
    #=信息增益/划分特征的信息熵
    #避免因为特征取值多而导致信息增益偏大
    #C4.5增加了对连续数据的处理，连续特征根据信息增益选择最佳分裂点转换为离散值
    def chooseFeatureByC4_5(self,data,continuous):
        #计算分割前的信息熵
        y=data.iloc[:,len(data.columns)-1]
        baseEntropy=self.getEntropy(y)
        #初始化变量
        bestInfGainRatio=0.0
        bestFeatureIdx=-1
        bestSplitValue=0.0
        #逐个计算按不同特征分割后的信息增益并选出增益最大的一个特征
        for i in range(len(data.columns)-1):
            x=data.iloc[:,i]
            #是否为连续特征
            if continuous[i]==True:
                splitValue,n=self.chooseSplitValue(x,y)
                splitFeatEntropy=self.getEntropy(x,True,splitValue)
                infGain=baseEntropy\
                    -self.getCondiEntropy(x,y,True,splitValue)\
                    -np.log2(n-1)/len(data)
            else:
                splitValue=0.0
                splitFeatEntropy=self.getEntropy(x)
                infGain=baseEntropy-self.getCondiEntropy(x,y)
            if splitFeatEntropy==0:
                continue
            infGainRatio=infGain/splitFeatEntropy
            if infGainRatio>bestInfGainRatio:
                bestInfGainRatio=infGainRatio
                bestFeatureIdx=i
                bestSplitValue=splitValue
        return bestFeatureIdx,bestSplitValue
    
    #最优分裂点选择
    def chooseSplitValue(self,x,y):
        #计算分裂前的信息熵
        baseEntropy=self.getEntropy(y)
        #需要尝试的分裂点
        values=self.screeningSplitValues(x,y)
        #初始化变量
        bestInfGain=0.0
        bestSplitValue=values[0]
        #逐个计算所有可能分裂点的条件熵
        for j in range(len(values)-1):
            split_value=values[j]
            infGain=baseEntropy-self.getCondiEntropy(x,y,True,split_value)
            if infGain>bestInfGain:
                bestInfGain=infGain
                bestSplitValue=split_value
        return bestSplitValue,len(values)
    
    #筛选分裂点，取分类结果有变化的点
    def screeningSplitValues(self,x,y):
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
    
    #选择出现频数最高的类作为叶节点判定的类
    def chooseClass(self,y):
        values=y.groupby(y).count()
        return values[values==values.max()].index.values[0]
    
    #生成树，结果以字典形式返回
    def createTree(self,data,model_type='C4.5',h_max=10,h=1):
        start = time.clock()
        type_list=('ID3','C4.5','CART')
        if model_type not in type_list:
            print('model_type should in:')
            print(type_list)
            raise TypeError('Unknown type')
        #离散性默认设置
        continuous=[]
        for dtype in data.dtypes.iloc[:(len(data.dtypes)-1)]:
            if str(dtype) in ['object','category','bool']:
                continuous.append(False)
            else:
                continuous.append(True)
        #ID3不支持连续特征
        if model_type=='ID3':
            if True in continuous:
                print(continuous)
                raise TypeError('ID3 does not support continuous features') 
        #类别数据
        y=data.iloc[:,len(data.columns)-1]
        #数据集只有一个类时返回这个类名
        if len(y.groupby(y).count())==1:
            return y[y.index[0]]
        #可用特征不足，返回出现频数最高的类名
        if len(data.columns)==1:
            return self.chooseClass(y)
        #超出高度上限，返回出现频数最高的类名
        if h>h_max:
            return self.chooseClass(y)
        #选择最优特征进行分割，并以字典形式记录结果
        #格式：{特征名：{特征值（中间结点）：{...},特征值（叶结点）：类名}}
        if model_type=='ID3':
            bestFeatureIdx=self.chooseFeatureByID3(data)
            bestSplitValue=0.0
        elif model_type=='C4.5':
            bestFeatureIdx,bestSplitValue=self.chooseFeatureByC4_5(data,continuous)
        #特征值统一，无法继续分割
        if bestFeatureIdx==-1:
            return self.chooseClass(y)
        #获取最优划分特征的相关信息
        bestFeatureLabel=data.columns[bestFeatureIdx]
        bestFeatureContinuous=continuous[bestFeatureIdx]
        #定义树
        deciTree={bestFeatureLabel:{}}
        #分割数据集
        split_set,split_values=self.split(data,bestFeatureIdx,
                                          bestFeatureContinuous,
                                          bestSplitValue)
        #对各个结点进行递归生成剩余部分
        for i in range(len(split_values)):
            deciTree[bestFeatureLabel][split_values[i]]=self.createTree(
                    split_set[i].drop(bestFeatureLabel,axis=1),
                    model_type,h_max,h+1)
        end = time.clock()
        if h==1:
            print('\ntime used for trainning:%f'%(end-start))
        return deciTree
    
    #预测
    def predict(self,tree,x,fill_empty=True,first=True):
        start = time.clock()
        #定义存放分类结果的series
        p_y=pd.Series(np.full(len(x),''),index=x.index,name='classify')
        #获取首个结点在dict中对应的key，即最优划分特征
        bestFeature=list(tree.keys())[0]
        #截取该节点下方的分支树
        childDict=tree[bestFeature]
        #遍历每个分支
        for key in childDict.keys():
            #连续特征和离散特征采用不同方式处理
            if type(key)==type(''):
                if key.find('<=')>=0:
                    #注：暂不支持时间类型的分割，所以直接转为float
                    splitValue=float(key.replace('<=',''))
                    boolIdx=(x[bestFeature]<=splitValue)
                elif key.find('>')>=0:
                    splitValue=float(key.replace('>',''))
                    boolIdx=(x[bestFeature]>splitValue)
                else:
                    boolIdx=(x[bestFeature]==key)
            else:
                boolIdx=(x[bestFeature]==key)
            #如果是中间结点就拆分数据并继续递归，如果是叶结点则返回类名
            if type(childDict[key])==type({}):
                p_y.update(self.predict(
                    childDict[key],x[boolIdx],first=False))
            else:
                #注：直接对x[bool_index]更新是错误做法
                p_y.loc[boolIdx]=childDict[key]
        #按类别分布情况加权随机填充未能分类的记录
        if (first==True)&(fill_empty==True):
            nullIdx=(p_y=='')
            n=p_y[nullIdx].count()
            p_y.loc[nullIdx]=p_y[~nullIdx].sample(n=n).tolist()
        end = time.clock()
        if first==True:
            print('\ntime used for predict:%f'%(end-start))
        return p_y
    
    #评估
    def assessment(self,y,p_y):
        p_y.index=y.index
        cp=pd.DataFrame()
        cp['y'],cp['p']=y,p_y
        accuracy=len(cp[cp['y']==cp['p']])*1.0/len(y)
        return accuracy
    
    #打印结点信息
    def printNodes(self,tree,nodeId='0',h=0):
        firstKey=list(tree.keys())[0]
        childDict=tree[firstKey]
        childNodeId=0
        if h==0:
            print('\n[Nodes Info]')
        print('<inNode Id=%s pId=%s h=%d> bestFeature:%s'
              %(nodeId,nodeId[:-1],h,firstKey))
        for key in childDict.keys():
            if type(childDict[key])==type({}):
                print('|--%s=%s'%(firstKey,str(key)))
                self.printNodes(childDict[key],nodeId+str(childNodeId),h+1)
            else:
                print('|--%s=%s'%(firstKey,str(key)))
                print('<leafNode Id=%s pId=%s h=%d> class:%s'
                      %(nodeId+str(childNodeId),nodeId,h+1,str(childDict[key])))
            childNodeId+=1
    
    #保存树结构
    def saveTree(self,tree,file_path):
        treeStr=str(tree)
        treeStr=treeStr.replace('Interval','pd.Interval')
        file=open(file_path,'w')
        file.write(treeStr)
        file.close()
    
    #读取树结构    
    def readTree(self,file_path):
        file=open(file_path,'r')
        treeStr=file.read()
        file.close()
        return eval(treeStr)
    
    #计算树的叶节点数
    def getLeafNum(self,tree):
        leafNum=0
        firstKey=list(tree.keys())[0]
        childDict=tree[firstKey]
        for key in childDict.keys():
            #如果是中间结点就加上分支树的叶结点数，如果是叶结点则数量加1
            if type(childDict[key])==type({}):
                leafNum+=self.getLeafNum(childDict[key])
            else:
                leafNum+=1
        return leafNum
    
    #计算树的高度
    def getTreeHeight(self,tree):
        height_max=0
        firstKey=list(tree.keys())[0]
        childDict=tree[firstKey]
        for key in childDict.keys():
            #如果是中间结点就在分支树的高度上加1，如果是叶结点算作1高度
            if type(childDict[key])==type({}):
                height=1+self.getTreeHeight(childDict[key])
            else:
                height=1
            #保留分支能抵达的最大的一个高度
            if height>height_max:
                height_max=height
        return height_max
    
    #注：目前可视化不能用于展示复杂的树，看不清
    #定义可视化格式
    inNode = dict(boxstyle="round4", color='#3366FF')  # 定义中间判断结点形态
    leafNode = dict(boxstyle="circle", color='#FF6633')  # 定义叶结点形态
    arrow_args = dict(arrowstyle="<-", color='g')  # 定义箭头
    
     # 绘制带箭头的注释
    def plotNode(self,nodeTxt, centerPt, parentPt, nodeType):
        self.ax1.annotate(nodeTxt, xy=parentPt,  xycoords='axes fraction',
                 xytext=centerPt, textcoords='axes fraction',
                 va="center", ha="center", bbox=nodeType, arrowprops=self.arrow_args )
    
    # 在父子结点间填充文本信息  
    def plotMidText(self,cntrPt, parentPt, txtString):
        xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]
        yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
        self.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)
    
    #绘制当前结点
    def plotTree(self,tree, parentPt, nodeTxt):
        leafNum = self.getLeafNum(tree)
        #depth = self.getTreeHeight(tree)
        firstStr = list(tree.keys())[0]      
        cntrPt = (self.xOff + (1.0 + float(leafNum))/2.0/self.totalW, self.yOff)
        #绘制中间结点并标记对应的划分属性值
        self.plotMidText(cntrPt, parentPt, nodeTxt)
        self.plotNode(firstStr, cntrPt, parentPt, self.inNode)
        secondDict = tree[firstStr]
        #减少y偏移
        self.yOff = self.yOff - 1.0/self.totalD 
        for key in secondDict.keys():
            #中间结点继续调用该方法绘制
            if type(secondDict[key]).__name__=='dict':   
                self.plotTree(secondDict[key],cntrPt,str(key))
            #绘制叶结点
            else:
                self.xOff = self.xOff + 1.0/self.totalW
                self.plotNode(secondDict[key], (self.xOff, self.yOff), cntrPt, self.leafNode)
                self.plotMidText((self.xOff, self.yOff), cntrPt, str(key))
        self.yOff = self.yOff + 1.0/self.totalD
    
    #绘制树
    def createPlot(self,tree):
        print('\n[Tree Plot]')
        fig = plt.figure(1, facecolor='white')
        fig.clf()
        axprops = dict(xticks=[], yticks=[])
        self.ax1 = plt.subplot(111, frameon=False, **axprops)
        self.totalW = float(self.getLeafNum(tree))
        self.totalD = float(self.getTreeHeight(tree))
        self.xOff = -0.5/self.totalW
        self.yOff = 1.0
        self.plotTree(tree, (0.5,1.0), '')
        plt.show()
    
    
    
    
    
    
    
            
            
            