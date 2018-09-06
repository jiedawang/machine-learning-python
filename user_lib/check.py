# -*- coding: utf-8 -*-

#(统一异常信息和简化代码,一部分功能其实在传入参数时完成)

#参数类型校验
def check_type(name,var_type,req_type):
    '''
    name:变量名，str类型
    var_type:变量的类型，type类型，可用type([参数])获取到
    req_type:要求的类型，type或list(type)类型，可用type([正确类型示例])获取到
    '''
    if type(req_type)==type([]):
        if var_type in req_type: return
    else:        
        if var_type==req_type: return
    var_type_str=str(var_type).replace("<class '","").replace("'>","")
    req_type_str=str(req_type).replace("<class '","").replace("'>","")
    raise TypeError('wrong type of %s\nunsupported -> %s\nrequired -> %s'
                    %(name,var_type_str,req_type_str))
  
#参数取值校验        
def check_limit(name,condition,required):
    '''
    name:变量名，str类型
    condition:限制条件，bool类型，直接在传入时写布尔表达式就行了
    required:正确取值提示，str类型
    '''  
    if condition==False:
        required_str=required.replace("<class '","").replace("'>","")
        raise ValueError('the value of %s does not meet the requirements'%name+
                        '\nrequired -> %s'%required_str)

#index匹配性校验 
def check_index_match(a,b,a_name,b_name,only_len=False):
    '''
    a,b: 数据集，Series或DataFrame类型
    a_name,b_name: 数据集名称，str类型
    '''
    if len(a)!=len(b):
        raise ValueError('the lengths of %s and %s do not match'%(a_name,b_name))
    if only_len==False:
        if (a.index==b.index).all()==False:
            raise ValueError('the indexs of %s and %s do not match'%(a_name,b_name))

#序列元素匹配校验
def check_items_match(a,b,a_name,b_name,item_name,mode='len'):
    '''
    a,b: 两个序列，list或ndarray类型
    a_name,b_name: 序列名称，str类型
    item_name: 匹配要素名称，str类型
    mode: 匹配模式，str类型，
          'len'->仅校验长度
          'left'->左列表优先，右列表必须包含全部左列表的值
          'right'->右列表优先，左列表必须包含全部右列表的值
          'equal'->两个列表必须完全相等
    '''
    if mode=='len':
        if len(a)!=len(b):
            raise ValueError('the %s of %s and %s do not match'%(item_name,a_name,b_name))
    elif mode=='left':
        for feature in a:
            if feature not in b:
                raise ValueError('the %s of %s do not match to the %s'%(item_name,b_name,a_name))
    elif mode=='right':
        for feature in b:
            if feature not in a:
                raise ValueError('the %s of %s do not match to the %s'%(item_name,a_name,b_name))
    elif mode=='equal':
        if set(a)!=set(b):
            raise ValueError('the %s of %s and %s do not match'%(item_name,a_name,b_name))
    else:
        raise ValueError('unknown mode')