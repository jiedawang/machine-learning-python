# -*- coding: UTF-8 -*-

import pyodbc

class MSSQL_Manager:

    #构造函数
    def __init__(self,host,user,pwd,db):
        self.host=host
        self.user=user
        self.pwd=pwd
        self.db=db

    #私有方法：获取数据库连接
    def __GetConnection(self):
        """
        得到连接信息
        返回: conn.cursor()
        """
        #参数校验
        if not self.db:
            raise NameError("没有设置数据库")
        if not self.user:
            raise NameError("没有设置用户名")
        if not self.pwd:
            raise NameError("没有设置密码")
        #连接数据库
        self.conn = pyodbc.connect('DRIVER={SQL Server Native Client 10.0};SERVER='
                                       +self.host+';DATABASE='+self.db+';UID='
                                       +self.user+';PWD='+self.pwd)
        cur = self.conn.cursor()
        #判断连接是否有效
        if not cur:
            raise NameError("连接数据库失败")
        else:
            return cur

    #自定义数据查询（有返回值）
    def ExecQuery(self,sql):
        """
        执行查询语句
        返回的是一个包含tuple的list，list的元素是记录行，tuple的元素是每行记录的字段

        调用示例：
                ms = MSSQL(host="localhost",user="sa",pwd="123456",db="PythonWeiboStatistics")
                resList = ms.ExecQuery("SELECT id,NickName FROM WeiBoUser")
                for (id,NickName) in resList:
                    print str(id),NickName
        """
        try:
            cur = self.__GetConnection()
            cur.execute(sql)
            buffer = cur.fetchall()
            resList=[]
            for i in buffer:
                resList.append(list(i))
            defList=[]
            for i in cur.description:
                defList.append(i[0])
            #查询完毕后必须关闭连接
            self.conn.close()
            return resList,defList
        except:
            raise

    # 自定义数据操作（无返回值）
    def ExecNonQuery(self,sql):
        """
        执行非查询语句

        调用示例：
            cur = self.__GetConnect()
            cur.execute(sql)
            self.conn.commit()
            self.conn.close()
        """
        try:
            cur = self.__GetConnection()
            cur.execute(sql)
            self.conn.commit()
            self.conn.close()
        except:
            raise

    # 获取全部数据库信息
    def GetDatabases(self):
        return self.ExecQuery("SELECT * FROM sys.databases")

    # 获取全部表信息
    def GetTables(self):
        return self.ExecQuery("SELECT * FROM sys.objects WHERE type='U'")

    # 获取表的列信息
    def GetColumns(self,table):
        if type(table)==type(1):
            return self.ExecQuery("SELECT * FROM sys.columns WHERE object_id= %d" % (table))
        elif type(table)==type("a"):
            return self.ExecQuery('''
            SELECT col.* FROM sys.columns col 
            JOIN sys.objects obj ON col.object_id=obj.object_id
            AND obj.name='%s'
            ''' % (table))
        else:
            raise NameError("无效的参数类型")
