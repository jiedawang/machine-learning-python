# -*- coding: utf-8 -*-
from PyQt5.QtWidgets import (QWidget, QToolTip, 
    QPushButton, QApplication, QMainWindow,QGraphicsScene,
    QMenu, QVBoxLayout, QSizePolicy, QMessageBox)
from PyQt5.uic import loadUi
from PyQt5.QtGui import QFont
from PyQt5.QtCore import QCoreApplication,QTimer
import matplotlib
#matplotlib.use("Qt5Agg")  # 声明使用QT5
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import sys
import random

#通过继承FigureCanvas类，使得该类既是一个PyQt5的Qwidget，
#又是一个matplotlib的FigureCanvas，这是连接pyqt5与matplotlib的关键
class QtFigureCanvas(FigureCanvas):
    
    def __init__(self,parent=None,width=10,height=6,dpi=100):
        self.fig=Figure(figsize=(width,height),dpi=dpi)
        #调用figure下面的add_subplot方法，类似于matplotlib.pyplot下面的subplot方法
        self.axes=self.fig.add_subplot(111)
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
        self.timer.start(1000)
    
    #初始化图像  
    def initial_figure(self):
        self.axes.plot([0,1,2,3],[1,2,0,4],'r')
    
    #更新图像 
    def update_figure(self):
        #随机生成0~10范围内的4个数
        y=[random.randint(0,10) for i in range(4)]
        #清除原来的图像并重新绘制
        self.axes.cla()
        self.axes.plot([0,1,2,3],y,'r')
        self.draw()
    
    #启停自动更新
    def start_update(self):
        self.timer.start(1000)
        
    def stop_update(self):
        self.timer.stop()
        
           
class MainWindow(QMainWindow):
    
    def __init__(self, parent = None):
        super(MainWindow, self).__init__(parent)
        #加载UI文件
        loadUi('../ui/mainwindow.ui', self)
        
        #设置窗体标题
        self.setWindowTitle('Test')
        #创建画布并加入布局
        self.dynamicCanvas=DynamicCanvas(self.centralwidget)
        self.verticalLayout2.addWidget(self.dynamicCanvas)
        #设置按钮点击事件
        self.btnStart.clicked.connect(self.dynamicCanvas.start_update)
        self.btnStop.clicked.connect(self.dynamicCanvas.stop_update)
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
        

''' 
class SimpleForm(QWidget):
    
    def __init__(self):
        super().__init__()
        self.initUI()
   
    def initUI(self):
        #为工具提示设置字体
        QToolTip.setFont(QFont('SansSerif', 10))
        #为窗口创建一个提示信息
        self.setToolTip('This is a application for <b>test</b>')
        #创建按钮并设置提示
        btn1=QPushButton('Show Text',self)
        btn2=QPushButton('Quit',self)
        #btn.sizeHint()显示默认尺寸
        btn1.resize(btn1.sizeHint())
        btn2.resize(btn2.sizeHint())
        #移动控件位置
        btn1.move(40,230)
        btn2.move(170,230)
        #绑定事件
        btn2.clicked.connect(QCoreApplication.instance().quit)
        #显示窗体
        self.setGeometry(300,300,300,300)
        self.setWindowTitle('Test')
        self.show()
    
if __name__=='__main__':
    #应用程序对象
    app=QApplication(sys.argv)
    app.aboutToQuit.connect(app.deleteLater)
    #用户界面对象
    ex=SimpleForm()    
    #系统exit()方法确保应用程序干净的退出
    #的exec_()方法有下划线。因为exec是一个Python关键词。因此，用exec_()代替
    sys.exit(app.exec_())
'''
