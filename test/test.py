# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

#通过list列表或array数组创建，见图1
#注：这两种类型都是用中括号标识
s=pd.Series([1,2,3,4])

#可接收嵌套list和多维array，见图2
#但这样一来Series中存放的就是list和array而不是数值了，通常不会这么做
s=pd.Series([[1,2],[3,4]])

#通过tuple元组创建，和list、array差不多
s=pd.Series((1,2,3,4))

#通过dict字典创建，见图3
#字典的key会作为索引，value会作为值
s=pd.Series({'name':'jack','age':20})

#通过其他参数设置可以定义列名和索引列，见图4
s=pd.Series([1,2,3,4],name='level',index=['a','b','c','d'])

#借助一些现成的函数生成特殊的Series，见图5
s=pd.Series(np.ones(4))