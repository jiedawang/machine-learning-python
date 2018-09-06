# -*- coding: utf-8 -*-
from PyQt5.QtWidgets import (QWidget, QToolTip, 
    QPushButton, QApplication, QMainWindow,QGraphicsScene,
    QMenu, QVBoxLayout, QSizePolicy, QMessageBox)
from PyQt5.uic import loadUi
from PyQt5.QtGui import QFont
from PyQt5.QtCore import QCoreApplication,QTimer

import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

import numpy as np
import sys
import random
import pandas as pd
import user_lib.svm as svm

#非线性可分简单数据集
f = open('D:\\training_data\\used\\simple_data2.txt')
buf = pd.read_table(f,header=None,sep=',')
buf.columns=['x1','x2','y']
describe=buf.describe()

X=buf.iloc[:,:2].copy()
y=buf.iloc[:,2].copy()
#X2_=dp.minmax_scaler(X2)
y[y==0]=-1

x0=np.array([0.125,0.125])
#x0=np.array([[-0.6,0.5],[0.4,-0.5],[0.9,0.4]])

#通过继承FigureCanvas类，使得该类既是一个PyQt5的Qwidget，
#又是一个matplotlib的FigureCanvas，这是连接pyqt5与matplotlib的关键
class QtFigureCanvas(FigureCanvas):
    
    def __init__(self,parent=None,width=12,height=8,dpi=100):
        self.fig=plt.figure(figsize=(width,height),dpi=dpi)
        #调用figure下面的add_subplot方法，类似于matplotlib.pyplot下面的subplot方法
        self.axes=self.fig.add_subplot(111, projection='3d')
        #切换matplotlib后端，关闭ipython输出
        #plt.switch_backend('Agg')
        #关闭绘图窗口
        #plt.close() 
        #初始化图像
        self.initial_figure()
        #初始化父类
        FigureCanvas.__init__(self,self.fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
    
    def initial_figure(self):
        pass

#动态画布：每秒自动更新
class DynamicCanvas(QtFigureCanvas):
    
    def __init__(self,*args,**kwargs):
        QtFigureCanvas.__init__(self,*args,**kwargs)
        self.timer=QTimer(self)
        self.timer.timeout.connect(self.update_figure)
        self.i=0
        self.step=15
        self.k_type='lin'
    
    #初始化图像  
    def initial_figure(self):
        self.k_plot(x0,X,y,k_type=None,scalar=0.)
    
    #核函数图像
    def k_plot(self,x0,X,y,k_type=None,scalar=1.):
        svm0=svm.SupportVectorMachine()
        if k_type=='lin':
            self.axes.set_title('Linear')
            z=svm0.lin_kernel_(X.values,x0)#.sum(axis=1)
        elif k_type=='rbf':
            self.axes.set_title('RBF')
            z=svm0.rbf_kernel_(X.values,x0,1.)#.sum(axis=1)
        else:
            k_type='lin'
            z=svm0.lin_kernel_(X.values,x0)#.sum(axis=1)
        self.axes.scatter(X['x1'][y==1],X['x2'][y==1],scalar*z[y==1],c='b',marker='o')
        self.axes.scatter(X['x1'][y==-1],X['x2'][y==-1],scalar*z[y==-1],c='r',marker='o')
        self.axes.set_xlabel('x1')
        self.axes.set_ylabel('x2')
        self.axes.set_zlabel('k(x0,x)')
        self.axes.set_zlim(scalar*z.min(),z.max())
    
    #更新图像 
    def update_figure(self):
        #清除原来的图像并重新绘制
        self.axes.cla()
        self.k_plot(x0,X,y,k_type=self.k_type,scalar=self.i/self.step)
        self.draw()
        self.i+=1
        if self.i>self.step:
            self.timer.stop()
    
    #启停自动更新
    def show_kernel1(self):
        if self.i>self.step:
            self.i=0
        self.k_type='lin'
        self.timer.start(100)
        
    def show_kernel2(self):
        if self.i>self.step:
            self.i=0
        self.k_type='rbf'
        self.timer.start(100)
        
           
class MainWindow(QMainWindow):
    
    def __init__(self, parent = None):
        super(MainWindow, self).__init__(parent)
        #加载UI文件
        loadUi('../ui/kernel_effect.ui', self)
        
        #设置窗体标题
        self.setWindowTitle('Kernel effect')
        #创建画布并加入布局
        self.dynamicCanvas=DynamicCanvas(self.centralwidget)
        self.verticalLayout2.addWidget(self.dynamicCanvas)
        #设置按钮点击事件
        self.btnKernel1.clicked.connect(self.dynamicCanvas.show_kernel1)
        self.btnKernel2.clicked.connect(self.dynamicCanvas.show_kernel2)
        #设置第一层布局为主布局
        self.centralwidget.setLayout(self.verticalLayout)
        #获取焦点
        self.centralwidget.setFocus()
    
    #关闭窗体时弹出确认窗口    
    def closeEvent(self,event):
        reply=QMessageBox.question(self,'Message',
                          'Are you sure to quit?',
                          QMessageBox.Yes|QMessageBox.No,
                          QMessageBox.No)
        if reply==QMessageBox.Yes:
            plt.close()
            event.accept()
        else:
            event.ignore()
    
if ( __name__ == '__main__' ):
    app = None
    if ( not app ):
        app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    if ( app ):
        app.exec_()