# -*- coding: utf-8 -*-
#GUI
from PyQt5.QtWidgets import (QWidget,QToolTip,QComboBox,QPushButton,QLabel,
                             QApplication,QMainWindow,QMenu,QGridLayout,QStyleOption,QStyle,
                             QSizePolicy,QMessageBox,QFileDialog,QGraphicsScene)
from PyQt5.uic import loadUi
from PyQt5.QtGui import QFont,QPixmap,QImage,QColor,QPainter,QPen
from PyQt5.QtCore import QCoreApplication,QTimer,QThread,pyqtSignal,QSize,QPoint
from PyQt5.QtCore import Qt

#绘图
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

#模型和数据处理
import numpy as np
import sys
import user_lib.neural_network as nn
import time
from PIL import Image,ImageQt

#数据集
from sklearn.datasets import make_moons,make_circles
from user_lib.image_data import MnistManager,CifarManager

#模型管理
class ModelManager:
    
    def __init__(self):
        #数据集设置
        self.dataset_type=None
        self.dataset_path=None
        self.noise=None
        self.train_X,self.train_y=None,None
        self.test_X,self.test_y=None,None
        self.test_X2,self.test_y2=None,None
        self.data_reload=True
        #模型设置
        self.hidden_layers=None
        self.hidden_activation=None
        self.output_activation=None
        self.learning_rate=None
        self.iter_max=None
        self.optimizer=None
        self.batch_size=None
        self.l2_alpha=None
        self.dropout=None
        self.model=None
        self.model_rebuild=True
        #拟合设置
        self.external_monitor=None
        
    #数据集设置
    def dataset_config(self,dataset_type,dataset_path=None,noise=0.0):
        if self.dataset_type!=dataset_type:
            self.dataset_type=dataset_type
            self.data_reload=True
        if self.dataset_path!=dataset_path:
            self.dataset_path=dataset_path
            self.data_reload=True
        if self.noise!=noise:
            self.noise=noise
            self.data_reload=True
    
    #载入数据集
    def load_dataset(self):
        if self.data_reload==True:
            if self.dataset_type=='圆形':
                self.train_X,self.train_y=make_circles(n_samples=500,
                                                       noise=self.noise)
                self.test_X,self.test_y=None,None
                message='生成数据集完毕'
            elif self.dataset_type=='月牙形':
                self.train_X,self.train_y=make_moons(n_samples=500,
                                                     noise=self.noise)
                self.test_X,self.test_y=None,None
                message='生成数据集完毕'
            elif self.dataset_type=='mnist':
                message=''
                if type(self.dataset_path)==type(None):
                    message+='未指定数据集路径,已使用默认路径\n'
                    datafile_dir=None
                else:
                    datafile_dir=self.dataset_path
                mnist=MnistManager(datafile_dir,self.external_monitor)
                self.train_X,self.train_y,self.test_X,self.test_y=mnist.read_as_array()
                message+='载入Mnist数据集完毕'
            elif self.dataset_type=='cifar10':
                message=''
                if type(self.dataset_path)==type(None):
                    message+='未指定数据集路径,已使用默认路径\n'
                    datafile_dir=None
                else:
                    datafile_dir=self.dataset_path
                cifar=CifarManager(datafile_dir,self.external_monitor)
                self.train_X,self.train_y,self.test_X,self.test_y=cifar.read_as_array(chinese_label=True)
                message+='载入cifar10数据集完毕'
            else:
                raise ValueError('无法识别的数据集类型-%s'%self.dataset_type)
            self.data_reload=False
            self.model_rebuild=True
            return message
        else:
            return '无需重复加载，已跳过'
    
    #加载单张图片    
    def load_one_image(self,file_path):
        pixmap=QPixmap(file_path)
        try:
            label=file_path.split("/")[-1].split('.',1)[0].split('_',1)[1]
        except:
            print('\n注意：图片名字格式为[id]_[label].png时才能正确载入标签\n')
            label='?'
        pixmap,image_=self.format_image(pixmap)
        self.test_X2=image_
        self.test_y2=str(label)
        return pixmap
    
    #加载单张qt的pixmap    
    def load_one_pixmap(self,pixmap,transform_gray=True):
        pixmap,image_=self.format_image(pixmap)
        self.test_X2=image_
        self.test_y2=str('?')
        return pixmap
    
    #格式化图片
    def format_image(self,pixmap):
        #适配mnist数据集
        if self.dataset_type=='mnist':
            pixmap=pixmap.scaled(QSize(28,28))
            image=Image.fromqpixmap(pixmap)
            image_=np.asarray(image).astype('float64')
            #将彩色图转换为灰度图
            if len(image_.shape)==3:
                image_=(11*image_[:,:,0]+16*image_[:,:,1]+5*image_[:,:,2])/32
                image=Image.fromarray(np.uint8(image_))
            pixmap=ImageQt.toqpixmap(image)
        #适配cifar数据集
        elif self.dataset_type=='cifar10':
            pixmap=pixmap.scaled(QSize(32,32))
            image=Image.fromqpixmap(pixmap)
            image_=np.asarray(image).astype('float64')
            #将灰度图转换为彩色图
            if len(image_.shape)==2:
                image_=image_.repeat(3).reshape(image_.shape+(3,))
                image=Image.fromarray(np.uint8(image_))
            pixmap=ImageQt.toqpixmap(image)
        else:
            raise ValueError('需要先创建模型')
        return pixmap,image_
    
    #导出图片    
    def export_images(self):
        if self.dataset_type=='mnist':
            message=''
            if type(self.dataset_path)==type(None):
                message+='未指定数据集路径,已使用默认路径\n'
                datafile_dir=None
            else:
                datafile_dir=self.dataset_path
            mnist=MnistManager(datafile_dir,self.external_monitor)
            mnist.to_images(rewrite=True)
            message+='导出图片完毕'
            return message
        elif self.dataset_type=='cifar10':
            message=''
            if type(self.dataset_path)==type(None):
                message+='未指定数据集路径,已使用默认路径\n'
                datafile_dir=None
            else:
                datafile_dir=self.dataset_path
            cifar=CifarManager(datafile_dir,self.external_monitor)
            cifar.to_images(rewrite=True,chinese_label=True)
            message+='导出图片完毕'
            return message
        else:
            raise ValueError('无法识别的数据集类型')
    
    #模型设置    
    def model_config(self,hidden_layers=[],
                     hidden_activation='relu',output_activation='sigmoid',
                     learning_rate=0.1,iter_max=10,optimizer='Nesterov',
                     batch_size=100,l2_alpha=0.0,dropout='关闭'):
        if self.hidden_layers!=hidden_layers:
            self.hidden_layers=hidden_layers.copy()
            self.model_rebuild=True
        if self.hidden_activation!=hidden_activation:   
            self.hidden_activation=hidden_activation
            self.model_rebuild=True
        if self.output_activation!=output_activation:
            self.output_activation=output_activation
            self.model_rebuild=True
        self.learning_rate=learning_rate
        self.iter_max=iter_max
        self.optimizer=optimizer
        self.batch_size=batch_size
        self.l2_alpha=l2_alpha
        self.dropout=dropout
    
    #重置模型
    def reset_model(self):
        self.model_rebuild=True
        del self.model
        self.model=None
    
    #新建模型    
    def create_model(self):
        hidden_layers=tuple(self.hidden_layers) 
        if self.hidden_activation=='sigmoid':
            activation='sigm'
        elif self.hidden_activation in ['tanh','relu']:
            activation=self.hidden_activation
        else:
            raise ValueError('无法识别的隐含层激活函数类型')
        if self.output_activation=='sigmoid':
            softmax=False
        elif self.output_activation=='softmax':
            softmax=True
        else:
            raise ValueError('未知的softmax设置')
        learning_rate=self.learning_rate
        iter_max=self.iter_max
        if self.optimizer=='Nesterov':
            optimizer='nagd'
        elif self.optimizer=='SGD':
            optimizer='sgd'
        elif self.optimizer=='Momentum':
            optimizer='magd'
        elif self.optimizer=='Adam':
            optimizer='adam'
        else:
            raise ValueError('未知的优化器')
        batch_size=self.batch_size
        l2_alpha=self.l2_alpha
        if self.dropout=='关闭':
            dropout_p=0.0
        elif self.dropout=='开启':
            dropout_p=0.5
        else:
            raise ValueError('未知的弃权设置')
        if self.model_rebuild==True:
            self.model=nn.MultilayerPerceptron(
                    hidden_layers=hidden_layers,activation=activation,
                    softmax=softmax,optimizer=optimizer,batch_size=batch_size,
                    iter_max=iter_max,learning_rate=learning_rate,
                    l2_alpha=l2_alpha,dropout_p=dropout_p) 
            self.model_rebuild=False
            return '创建模型成功'
        else:
            self.model.bind_func_(activation,softmax,optimizer,'c')
            self.model.batch_size=batch_size
            self.model.iter_max=iter_max
            self.model.learning_rate=learning_rate
            self.model.l2_alpha=l2_alpha
            self.model.dropout_p=dropout_p
            return '无需重建模型，已更新设置'
    
    #设置拟合任务的外部监视方法
    def set_external_monitor(self,external_monitor):
        self.external_monitor=external_monitor
    
    #拟合
    def fit(self):
        start=time.clock()
        self.model.set_external_monitor(self.external_monitor)
        if self.dataset_type in ['mnist','cifar10']:
            self.model.fit(self.train_X/255,self.train_y,
                           self.test_X/255,self.test_y,
                           monitor_cost=True,monitor_score=True)
        else:
            self.model.fit(self.train_X,self.train_y,
                           self.test_X,self.test_y,
                           monitor_cost=True,monitor_score=True)
        return '\n训练用时: %f s\n'%(time.clock()-start)
    
    #预测
    def predict(self):
        if type(self.model)==type(None):
            raise ValueError('尚未构建模型，不能进行预测')
        samples_n,correct_n=0,0
        for i in range(self.test_y.shape[0]):
            prob_y=self.model.predict(self.test_X[i]/255,return_a=True)
            pred_y=self.model.prob_to_label_(prob_y,self.model.classes)
            message,correct_n,samples_n=self.predict_info(
                prob_y,pred_y,self.test_y[i],correct_n,samples_n
                )
            image=Image.fromarray(np.uint8(self.test_X[i]))
            pixmap=ImageQt.toqpixmap(image)
            message=(pixmap,message)
            if type(self.external_monitor)!=type(None):
                self.external_monitor(message)
                time.sleep(0.05)
        return '测试集验证结束\n'
    
    #单图预测            
    def predict_one(self):
        if type(self.model)==type(None):
            raise ValueError('尚未构建模型，不能进行预测')
        if type(self.test_X2)==type(None):
            raise ValueError('尚未选择图片，不能进行预测')
        if self.test_X2.shape!=self.model.input_shape:
            raise ValueError('当前图片数据格式与训练数据已不匹配，请重新选择加载')
        prob_y=self.model.predict(self.test_X2/255,return_a=True)
        pred_y=self.model.prob_to_label_(prob_y,self.model.classes)
        message,correct_n,samples_n=self.predict_info(
                prob_y,pred_y,self.test_y2,0,0
                )
        return message
    
    #预测信息
    def predict_info(self,prob_y,pred_y,y,correct_n,samples_n):
        message=pred_y+'|'
        prob_sort=np.argsort(-1.*prob_y)
        for j in range(prob_y.shape[0]):
            class_idx=prob_sort[j]
            message+=' %s : %.4f%%'%(self.model.classes[class_idx],
                                     prob_y[class_idx]*100)
            if j<prob_y.shape[0]-1:
                message+='<br>'
            else:
                message+='|'
        if pred_y==y:
            message+='正确|'
            correct_n+=1
        else:
            if y!='?':
                message+='错误|'
            else:
                message+='???|'
        samples_n+=1
        message+='目标分类 %s<br>命中数 %d/%d<br>准确率 %.4f%%'%(
                y,correct_n,samples_n,
                correct_n/samples_n*100)
        return message,correct_n,samples_n
    
    #绘制网络
    def plot_network(self):
        if type(self.model)==type(None):
            raise ValueError('尚未构建模型，不能进行绘制')
        fig,axes=self.model.plot_network(return_fig=True)
        return (fig,axes)

'''
mm=ModelManager()
mm.dataset_config('圆形')
mm.model_config(hidden_layers=[100,])
mm.load_dataset()
mm.create_model()
fig,axes=mm.plot_network()
mm.tranform_data()
mm.fit()
mm.predict()
'''

#生成网格数据
def generate_grid_data(x,y,broaden=0.2,steps=51):
    steps=complex(0,steps)
        
    #X值域
    x_min,x_max=x.min(),x.max()
    y_min,y_max=y.min(),y.max()
        
    #生成网格数据，用于绘制等高线图或曲面图
    x0,y0=np.mgrid[
            x_min-broaden:x_max+broaden:steps,
            y_min-broaden:y_max+broaden:steps
            ]
        
    return x0,y0
    
#网格数据预测
def grid_predict(x0,y0,model):
    
    xy0=np.c_[x0.reshape(x0.shape[0]*x0.shape[1]),
              y0.reshape(y0.shape[0]*y0.shape[1])]
        
    #模型预测等高线图
    z0=model.predict(xy0,return_a=True,check_input=False)
    z0=z0.reshape(x0.shape[0],x0.shape[1],2)[:,:,1]

    return z0

#网格数据预测
def grid_predict2(layer_id,neuron_id,x0,y0,model):

    xy0=np.c_[x0.reshape(x0.shape[0]*x0.shape[1]),
              y0.reshape(y0.shape[0]*y0.shape[1])]
        
    #模型预测等高线图
    al,al_grad=model.forward_prop_(xy0,return_al=True)
    z0=al[layer_id][:,neuron_id]
    z0=z0.reshape(x0.shape)

    return z0

#通过继承FigureCanvas类，使得该类既是一个PyQt5的Qwidget，
#又是一个matplotlib的FigureCanvas，这是连接pyqt5与matplotlib的关键

#拟合结果图
class FittingResultFigure(FigureCanvas):
    
    def __init__(self,parent=None,width=6,height=6,dpi=80):
        #新建图像和轴
        self.fig=plt.figure(figsize=(width,height),dpi=dpi)
        self.axes=self.fig.add_subplot(111)
        #切换matplotlib后端，关闭ipython输出
        #plt.switch_backend('Agg')
        #关闭绘图窗口
        plt.close() 
        #初始化父类
        FigureCanvas.__init__(self,self.fig)
        self.setParent(parent)
        '''
        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Preferred,
                                   QSizePolicy.Preferred)
        FigureCanvas.updateGeometry(self)
        '''
        #初始化图像
        self.initial_figure()

    #初始化图像  
    def initial_figure(self):
        self.axes.cla()
        self.fig.subplots_adjust(top=0.98,bottom=0.11,right=0.96,left=0.14)
        self.axes.set_xlabel('x1')
        self.axes.set_ylabel('x2')
        self.draw()
    
    #更新图像 
    def update_figure(self,x,y,z,x0,y0,z0):

        self.axes.cla()
        self.axes.contourf(x0,y0,z0,cmap=plt.cm.RdYlBu) 
        
        #数据集散点图
        self.axes.scatter(x[z==1],y[z==1],c='b',edgecolors='black',label='y=1')
        self.axes.scatter(x[z==0],y[z==0],c='r',edgecolors='black',label='y=0')
        self.axes.legend()
        
        self.axes.set_xlabel('x1')
        self.axes.set_ylabel('x2')
        #self.draw()
        
#训练评分视图
class TrainingViewFigure(FigureCanvas):
    
    def __init__(self,ylabel,parent=None,width=6,height=6,dpi=80):
        #新建图像和轴
        self.fig=plt.figure(figsize=(width,height),dpi=dpi)
        self.axes=self.fig.add_subplot(111)
        #关闭绘图窗口
        plt.close() 
        #初始化父类
        FigureCanvas.__init__(self,self.fig)
        self.setParent(parent)
        #初始化图像
        self.ylabel=ylabel
        self.initial_figure()

    #初始化图像  
    def initial_figure(self):
        self.axes.cla()
        self.fig.subplots_adjust(top=0.95,bottom=0.15,right=0.97,left=0.15)
        self.axes.set_xlabel('iter')
        self.axes.set_ylabel(self.ylabel)
        self.draw()
    
    #更新图像 
    def update_figure(self,x,y1,y2):
        self.axes.cla()
        #数据集散点图
        self.axes.plot(x,y1,c='deepskyblue',label='train')
        if type(y2)!=type(None):
            self.axes.plot(x,y2,c='orange',label='test')
        self.axes.legend()
        
        self.axes.set_xlabel('iter')
        self.axes.set_ylabel(self.ylabel)
        #self.draw()

#网络结构图
class StructureViewFigure(FigureCanvas):
    
    def __init__(self,parent=None,width=6,height=6,dpi=80):
        #新建图像和轴
        self.fig=plt.figure(figsize=(width,height),dpi=dpi)
        self.axes=self.fig.add_subplot(111,frameon=False)
        #关闭绘图窗口
        plt.close() 
        #初始化父类
        FigureCanvas.__init__(self,self.fig)
        self.setParent(parent)
        #初始化图像
        self.initial_figure()

    #初始化图像  
    def initial_figure(self):
        self.axes.cla()
        self.axes.set_xticks([])
        self.axes.set_yticks([])
        self.draw()
    
    #更新图像 
    def update_figure(self,model):
        self.axes.cla()
        self.axes.set_xticks([])
        self.axes.set_yticks([])
        layers=model.layers
        layers_n=len(layers)
        height=max(layers)
        layer_offset=0.19
        neuron_offset=0.165
        connect_offset=0.04
        show_max=6
        for i in range(len(model.weights)):
            if i==0:
                model.plot_layer_(self.axes,model.weights[i],model.biases[i],i+4-0.5*layers_n,height,show_max,
                                  layer_offset,neuron_offset,connect_offset,first=True)
            else:
                model.plot_layer_(self.axes,model.weights[i],model.biases[i],i+4-0.5*layers_n,height,show_max,
                                  layer_offset,neuron_offset,connect_offset,first=False)
        #self.fig.savefig('D://temp.jpg')
        #self.draw()
        
#神经元作用图
class NeuronActionFigure(FigureCanvas):
    
    def __init__(self,parent=None,width=6,height=6,dpi=80):
        #新建图像和轴
        self.fig=plt.figure(figsize=(width,height),dpi=dpi)
        self.axes=self.fig.add_subplot(111,projection='3d')
        #关闭绘图窗口
        plt.close() 
        #初始化父类
        FigureCanvas.__init__(self,self.fig)
        self.setParent(parent)
        #初始化图像
        self.initial_figure()

    #初始化图像  
    def initial_figure(self):
        self.axes.cla()
        self.axes.set_xlabel('x1')
        self.axes.set_ylabel('x2')
        self.axes.set_zlabel('a')
        self.draw()
    
    #更新图像 
    def update_figure(self,x,y,z):
        self.axes.cla()
        self.axes.set_xlabel('x1')
        self.axes.set_ylabel('x2')
        self.axes.set_zlabel('a')
        surf=self.axes.plot_surface(x,y,z,color='lightgrey',alpha=1.0)
        surf._facecolors2d=surf._facecolors3d
        surf._edgecolors2d=surf._edgecolors3d
        self.axes.view_init(elev=20, azim=40)
        #self.draw()

'''
fig=plt.figure(figsize=(4,4),dpi=100)
axes=fig.add_subplot(111)
fig.subplots_adjust(top=0.9,bottom=0.2,right=0.9,left=0.2)

plt.subplots_adjust(top=0.9, right=0.85)

test=FittingResultFigure(parent=None,width=10,height=10,dpi=100)
size_policy=QSizePolicy()
size_policy.setVerticalStretch(0)
size_policy.setHorizontalStretch(0)
size_policy.setVerticalPolicy(QSizePolicy.Preferred)
size_policy.setHorizontalPolicy(QSizePolicy.Preferred)
test.setSizePolicy(size_policy)
'''

#画板窗体
class PaintBoard(QMainWindow):
    
    return_trigger=pyqtSignal(QPixmap)
    
    def __init__(self,parent=None,base_image=None,mode='L'):
       super(PaintBoard,self).__init__(parent)
       #加载UI文件
       loadUi('../ui/paint_board.ui',self)
       #初始化画板
       self.initial_board(mode)
       #显示设置
       self.setFixedSize(QSize(629,523))
       #绑定事件
       self.comboBox_board_color.currentIndexChanged.connect(self.update_board_color)
       self.horizontalSlider_color_red.valueChanged.connect(self.update_painter_color_r)
       self.horizontalSlider_color_green.valueChanged.connect(self.update_painter_color_g)
       self.horizontalSlider_color_blue.valueChanged.connect(self.update_painter_color_b)
       self.comboBox_painter_type.currentIndexChanged.connect(self.update_painter_type)
       self.horizontalSlider_painter_thickness.valueChanged.connect(self.update_painter_type)
       self.pushButton_reset.clicked.connect(self.clear)
       self.pushButton_undo.clicked.connect(self.rollback)
       self.pushButton_complete.clicked.connect(self.return_pixmap)
       self.action.triggered.connect(self.open_image)
       self.action_2.triggered.connect(self.save_image)

    #初始化画板
    def initial_board(self,mode):
        self.mode=mode
        #初始化画板属性
        self.__is_empty=True
        self.__base_image=None
        self.__board_history=[]
        if mode=='L':
            self.comboBox_board_color.setCurrentIndex(1)
            self.__board_color=QColor("black")
        else:
            self.comboBox_board_color.setCurrentIndex(0)
            self.__board_color=QColor("white")
        #新建QPixmap作为画板显示源，用底色填充
        self.__board_size=QSize(470,470)
        self.__board=QPixmap(self.__board_size)
        self.__board.fill(self.__board_color)
        self.label_board.setPixmap(self.__board)
        #鼠标位置,上一次和当前位置
        self.__last_pos = QPoint(0,0)
        self.__current_pos = QPoint(0,0)
        #初始化绘图工具及其属性
        self.__painter=QPainter()
        self.update_painter_type()
        if mode=='L':
            self.set_painter_color_rgb(255,255,255)
        else:
            self.set_painter_color_rgb(0,0,0)

    #更新画板背景色
    def update_board_color(self):
        last_board_color=self.__board_color
        #更新画板颜色，如果画笔颜色与画板正好相反，同时更新画笔颜色
        if self.comboBox_board_color.currentText()=='白色':
            self.__board_color=QColor("white")
            if self.__painter_color.name()=='#ffffff':
                self.set_painter_color_rgb(0,0,0)
        else:
            self.__board_color=QColor("black")
            if self.__painter_color.name()=='#000000':
                self.set_painter_color_rgb(255,255,255)
        #画板为空时直接填充背景色
        if (self.__is_empty==True)&(type(self.__base_image)==type(None)):
            self.__board.fill(self.__board_color)
        #画板不为空时替换所有与背景色一致的颜色
        else:
            mask=self.__board.createMaskFromColor(last_board_color, Qt.MaskOutColor)
            self.__painter.begin(self.__board)
            self.__painter.setPen(self.__board_color)
            self.__painter.drawPixmap(self.__board.rect(),mask,mask.rect())
            self.__painter.end()
        self.label_board.setPixmap(self.__board)
    
    #更新画笔颜色
    def set_painter_color_rgb(self,r,g,b):
        self.horizontalSlider_color_red.setValue(r)
        self.horizontalSlider_color_green.setValue(g)
        self.horizontalSlider_color_blue.setValue(b)
        self.__painter_color=QColor.fromRgb(r,g,b)
        self.label_color_show.setStyleSheet("background-color: rgb(%d,%d,%d);"%(r,g,b))
    
    def update_painter_color_r(self):
        r=self.horizontalSlider_color_red.value()
        if self.mode=='L':
            g,b=r,r
            self.horizontalSlider_color_green.setValue(g)
            self.horizontalSlider_color_blue.setValue(b)
        else:
            g=self.horizontalSlider_color_green.value()
            b=self.horizontalSlider_color_blue.value()
        self.__painter_color=QColor.fromRgb(r,g,b)
        self.label_color_show.setStyleSheet("background-color: rgb(%d,%d,%d);"%(r,g,b))
        
    def update_painter_color_g(self):
        g=self.horizontalSlider_color_green.value()
        if self.mode=='L':
            r,b=g,g
            self.horizontalSlider_color_red.setValue(r)
            self.horizontalSlider_color_blue.setValue(b)
        else:
            r=self.horizontalSlider_color_red.value()
            b=self.horizontalSlider_color_blue.value()
        self.__painter_color=QColor.fromRgb(r,g,b)
        self.label_color_show.setStyleSheet("background-color: rgb(%d,%d,%d);"%(r,g,b))
        
    def update_painter_color_b(self):
        b=self.horizontalSlider_color_blue.value()
        if self.mode=='L':
            r,g=b,b
            self.horizontalSlider_color_red.setValue(r)
            self.horizontalSlider_color_green.setValue(g)
        else:
            r=self.horizontalSlider_color_red.value()
            g=self.horizontalSlider_color_green.value()
        self.__painter_color=QColor.fromRgb(r,g,b)
        self.label_color_show.setStyleSheet("background-color: rgb(%d,%d,%d);"%(r,g,b))
    
    #更新画笔类型
    def update_painter_type(self):
        thickness_=self.horizontalSlider_painter_thickness.value()
        self.__painter_thickness=10+3*thickness_
        self.__painter_type=self.comboBox_painter_type.currentText()
        if self.__painter_type=='橡皮':
            self.__eraser_mode=True
        else:
            self.__eraser_mode=False
        
    #清空画板    
    def clear(self):
        if type(self.__base_image)==type(None):
            self.__board.fill(self.__board_color)
        else:
            self.__board=self.__base_image.copy()
        self.label_board.setPixmap(self.__board)    
        self.__is_empty=True
    
    #回滚绘图操作    
    def rollback(self):
        if len(self.__board_history)>0:
           self.__board=self.__board_history.pop(-1)
           self.label_board.setPixmap(self.__board)
    
    #返回pixmap到父窗体       
    def return_pixmap(self):
        self.return_trigger.emit(self.__board)
        self.close()
    
    #鼠标按下事件    
    def mousePressEvent(self,e):
        #在状态栏显示鼠标位置
        self.show_status(e.x(),e.y())
        #记录鼠标位置
        self.__current_pos=e.pos()
        #更新上一次鼠标位置
        self.__last_pos=self.__current_pos
        #保存旧图像
        if len(self.__board_history)>=10:
            trash=self.__board_history.pop(0)
        self.__board_history.append(self.__board.copy())
        
    #鼠标移动事件
    def mouseMoveEvent(self,e):
        #在状态栏显示鼠标位置
        self.show_status(e.x(),e.y())
        #记录鼠标位置
        self.__current_pos=e.pos()
        #绘图
        self.__painter.begin(self.__board)
        #设置画笔
        #橡皮擦模式下颜色为背景色
        if self.__eraser_mode==False:
            self.__painter.setPen(QPen(self.__painter_color,
                                       self.__painter_thickness,
                                       cap=Qt.RoundCap))
        else:
            self.__painter.setPen(QPen(self.__board_color,
                                       self.__painter_thickness,
                                       cap=Qt.RoundCap))
        #画线    
        self.__painter.drawLine(self.__last_pos, self.__current_pos)
        self.__painter.end()
        #更新上一次鼠标位置
        self.__last_pos=self.__current_pos
        #更新显示
        self.label_board.setPixmap(self.__board)
    
    #鼠标释放事件    
    def mouseReleaseEvent(self,e):
        self.__is_empty=False
    
    #显示状态    
    def show_status(self,x,y):
        self.statusbar.showMessage('location : ( %d , %d ) ; thickness : %d'%(x,y,self.__painter_thickness))

    #保存图片
    def save_image(self):
        file_path,file_type=QFileDialog.getSaveFileName(
                self,"保存图片","C:\\temp.png" ,"PNG Image (*.png);;JPG Image (*.jpg)")
        if file_path!='':
            self.__board.save(file_path)
    
    #打开图片        
    def open_image(self):
        file_path,file_type=QFileDialog.getOpenFileName(
                self,"选取文件","C:\\","PNG Image (*.png);;JPG Image (*.jpg)") 
        if file_path!='':
            image=QImage(file_path)
            pixmap=QPixmap(image)
            pixmap=pixmap.scaled(QSize(428,428))
            self.__is_empty=True
            self.__base_image=pixmap
            self.__board_history=[]
            self.__board=pixmap.copy()
            self.label_board.setPixmap(self.__board)

#主窗体
class MainWindow(QMainWindow):
    
    def __init__(self,parent=None):
        super(MainWindow,self).__init__(parent)
        #加载UI文件
        loadUi('../ui/nn_show.ui',self)
        
        #设置窗体标题
        self.setWindowTitle('神经网络演示')
        
        #创建画布并加入布局
        size_policy=QSizePolicy()
        size_policy.setVerticalStretch(0)
        size_policy.setHorizontalStretch(0)
        size_policy.setVerticalPolicy(QSizePolicy.Preferred)
        size_policy.setHorizontalPolicy(QSizePolicy.Preferred)
        
        self.structure_view=StructureViewFigure(
                width=5.8,height=3.2,dpi=60)
        self.structure_view.setSizePolicy(size_policy)
        scene=QGraphicsScene()
        scene.addWidget(self.structure_view)
        self.graphicsView_structure1.setScene(scene)
        
        self.neuron_action=NeuronActionFigure(
                width=5.8,height=3.2,dpi=60)
        self.neuron_action.setSizePolicy(size_policy)
        self.scrollArea_neurons_action1.setWidget(self.neuron_action)

        self.fitting_result=FittingResultFigure(
                width=4,height=4)
        self.fitting_result.setSizePolicy(size_policy)
        self.gridLayout_7.addWidget(self.fitting_result)

        self.training_score_view1=TrainingViewFigure(
                ylabel='score',width=4,height=4)
        self.training_score_view1.setSizePolicy(size_policy)
        self.tabWidget_monitor1.addTab(self.training_score_view1, "训练评分")
        self.training_cost_view1=TrainingViewFigure(
                ylabel='cost',width=4,height=4)
        self.training_cost_view1.setSizePolicy(size_policy)
        self.tabWidget_monitor1.addTab(self.training_cost_view1, "训练代价")
        
        self.training_score_view2=TrainingViewFigure(
                ylabel='score',width=4,height=4)
        self.training_score_view2.setSizePolicy(size_policy)
        self.tabWidget_monitor2.addTab(self.training_score_view2, "训练评分")
        self.training_cost_view2=TrainingViewFigure(
                ylabel='cost',width=4,height=4)
        self.training_cost_view2.setSizePolicy(size_policy)
        self.tabWidget_monitor2.addTab(self.training_cost_view2, "训练代价")
        
        #绑定事件
        self.pushButton_structure1.clicked.connect(self.draw_structure1)
        self.pushButton_neurons_action1.clicked.connect(self.draw_neuron_action1)
        self.pushButton_add_layer1.clicked.connect(self.add_layer1)
        self.pushButton_remove_layer1.clicked.connect(self.remove_layer1)
        self.pushButton_train1.clicked.connect(self.start_train1)
        self.pushButton_reset1.clicked.connect(self.reset_model1)
        self.comboBox_layer1.currentIndexChanged.connect(self.update_combobox_neurons)
        
        self.pushButton_add_layer2.clicked.connect(self.add_layer2)
        self.pushButton_remove_layer2.clicked.connect(self.remove_layer2)
        self.pushButton_train2.clicked.connect(self.start_train2)
        self.pushButton_reset2.clicked.connect(self.reset_model2)
        self.pushButton_dataset_path2.clicked.connect(self.set_dataset_path2)
        self.pushButton_select_image2.clicked.connect(self.select_image2)
        self.pushButton_export_images2.clicked.connect(self.export_images2)
        self.pushButton_test2.clicked.connect(self.verify2)
        self.pushButton_test_one2.clicked.connect(self.verify_one2)
        self.pushButton_draw_image2.clicked.connect(self.open_paintboard)
        
        self.comboBox_output_activation2.currentIndexChanged.connect(self.match_cost)
        self.comboBox_cost2.currentIndexChanged.connect(self.match_output)

        #初始化
        self.modelManager1=ModelManager()
        self.hidden_layers1=[]
        self.modelManager2=ModelManager()
        self.hidden_layers2=[]
        self.hidden_layers_info2()
        self.background_status='free'
        self.dataset_path=None
        self.comboBox_layer1.addItem('输出层')
        self.update_combobox_neurons()
        self.open_neuron_action_view=False
        
        #获取焦点
        self.centralWidget.setFocus()

    #关闭窗体时弹出确认窗口    
    def closeEvent(self,event):
        reply=QMessageBox.question(self,'提示',
                          '确认退出?',
                          QMessageBox.Yes|QMessageBox.No,
                          QMessageBox.No)
        if reply==QMessageBox.Yes:
            #plt.close()
            event.accept()
        else:
            event.ignore()
            
    #更新后台任务状态
    def update_background_status(self,status):
        #重置按钮/停止按钮转换
        if status=='busy':
            self.pushButton_reset1.setText('停止')
            self.pushButton_reset1.setStyleSheet('background-color: rgb(255,165,0);')
            self.pushButton_reset2.setText('停止')
            self.pushButton_reset2.setStyleSheet('background-color: rgb(255,165,0);')
        elif status=='free':
            self.pushButton_reset1.setText('重置')
            self.pushButton_reset1.setStyleSheet('background-color: rgb(224,255,255);')
            self.pushButton_reset2.setText('重置')
            self.pushButton_reset2.setStyleSheet('background-color: rgb(224,255,255);')
        else:
            raise ValueError('未定义的后台任务状态')
        self.background_status=status
    
    #强制停止后台任务    
    def background_stop(self):
        reply=QMessageBox.question(self,'提示',
                                   '强制停止任务可能发生不可预知的错误，是否停止？',
                                   QMessageBox.Yes|QMessageBox.No,
                                   QMessageBox.No)
        if reply==QMessageBox.Yes:
            self.background.terminate()
            self.background.quit()
            self.update_background_status('free')
            self.monitor_append2('\n已强制终止后台任务\n')
            
    #代价函数和输出层激活函数绑定，同时变更
    def match_cost(self):
        if self.comboBox_output_activation2.currentIndex()==1:
            self.comboBox_cost2.setCurrentIndex(1)
        else:
            self.comboBox_cost2.setCurrentIndex(0)
            
    def match_output(self):
        if self.comboBox_cost2.currentIndex()==1:
            self.comboBox_output_activation2.setCurrentIndex(1)
        else:
            self.comboBox_output_activation2.setCurrentIndex(0)
            
    #添加层
    def add_layer1(self):
        neurons_n=self.spinBox_neurons_n1.value()
        self.hidden_layers1.append(neurons_n)
        self.hidden_layers_info1()
        hidden_layers_n=len(self.hidden_layers1)
        self.comboBox_layer1.addItem('隐含层%d'%(hidden_layers_n))
        self.update_combobox_neurons()
        
    def add_layer2(self):
        neurons_n=self.spinBox_neurons_n2.value()
        self.hidden_layers2.append(neurons_n)
        self.hidden_layers_info2()
        
    #移除层
    def remove_layer1(self):
        if len(self.hidden_layers1)>0:
            self.hidden_layers1.pop(-1)
            self.hidden_layers_info1()
            items_n=self.comboBox_layer1.count()
            self.comboBox_layer1.removeItem(items_n-1)
            self.update_combobox_neurons()
            
    def remove_layer2(self):
        if len(self.hidden_layers2)>0:
            self.hidden_layers2.pop(-1)
            self.hidden_layers_info2()

    #显示隐含层结构信息
    def hidden_layers_info1(self):
        text=''
        for i in range(len(self.hidden_layers1)):
            text+=str(self.hidden_layers1[i])
            if i<len(self.hidden_layers1)-1:
                text+=','
        self.textBrowser_hidden_layers1.setHtml(text)
        
    def hidden_layers_info2(self):
        text=''
        for i in range(len(self.hidden_layers2)):
            text+=str(self.hidden_layers2[i])
            if i<len(self.hidden_layers2)-1:
                text+=','
        self.textBrowser_hidden_layers2.setHtml(text)
        
    #更新神经元选择下拉列表
    def update_combobox_neurons(self):
        if self.comboBox_layer1.currentIndex()==0:
            self.comboBox_neurons1.clear()
            for i in range(2):
                self.comboBox_neurons1.addItem('神经元%d'%(i+1))
        else:
            layer_id=self.comboBox_layer1.currentIndex()
            neurons_n=self.modelManager1.model.layers[layer_id]
            self.comboBox_neurons1.clear()
            for i in range(neurons_n):
                self.comboBox_neurons1.addItem('神经元%d'%(i+1))
        
    #训练
    def start_train1(self):
        #后台线程忙碌时进行阻断
        if self.background_status!='free':
            self.monitor_append1('\n后台有任务正在运行，请等待\n')
            return
        #清空监控窗口
        self.monitor_clear1()
        #载入设置
        self.modelManager1.dataset_config(
                dataset_type=self.comboBox_dataset_type1.currentText(),
                noise=self.doubleSpinBox_noise1.value()
                )
        self.modelManager1.model_config(
                hidden_layers=self.hidden_layers1,
                hidden_activation=self.comboBox_activation1.currentText(),
                learning_rate=self.doubleSpinBox_learning_rate1.value(),
                iter_max=self.spinBox_iter_max1.value()
                )
        #后台任务队列
        tasks,tasks_name=[],[]
        tasks.append(self.modelManager1.load_dataset)
        tasks.append(self.modelManager1.create_model)
        tasks.append(self.modelManager1.fit)
        tasks_name.append('加载%s数据集'%self.modelManager1.dataset_type)
        tasks_name.append('创建模型')
        tasks_name.append('训练模型')
        #创建后台任务
        self.background=WorkThread(tasks,tasks_name)
        #将信号发射方法和其他要执行的任务封装，设置为模型管理器的外部监视方法，
        #这样长耗时任务可以反馈进度(需要调度的对象提供该参数支持)
        #注：此方法会在子线程中运行，所以不能进行UI更新操作，会奔溃
        def external_monitor(messages):
            #拟合结果图
            x=self.modelManager1.train_X[:,0]
            y=self.modelManager1.train_X[:,1]
            z=self.modelManager1.train_y
            x0,y0=generate_grid_data(x,y,broaden=0.2,steps=51)
            z0=grid_predict(x0,y0,self.modelManager1.model)
            self.fitting_result.update_figure(x,y,z,x0,y0,z0) 
            #训练监控曲线图
            message,score_h,cost_h=messages[0],messages[1],messages[2]
            x=range(len(score_h))
            y=np.array(score_h)
            self.training_score_view1.update_figure(x,y,None)
            y=np.array(cost_h)
            self.training_cost_view1.update_figure(x,y,None)
            self.background.task_trigger4.emit((message,))
            time.sleep(0.05)
        self.modelManager1.set_external_monitor(external_monitor)
        #绑定信号槽以更新监控窗口
        self.background.task_trigger0.connect(self.monitor_append1)
        self.background.task_trigger4.connect(self.train_monitor1)
        self.background.thread_trigger.connect(self.update_background_status)
        #启动后台任务
        self.background.start()
        
    def start_train2(self):
        #后台线程忙碌时进行阻断
        if self.background_status!='free':
            self.monitor_append2('\n后台有任务正在运行，请等待\n')
            return
        #清空监控窗口
        self.monitor_clear2()
        #载入设置
        self.modelManager2.dataset_config(
                dataset_type=self.comboBox_dataset_type2.currentText(),
                dataset_path=self.dataset_path
                )
        self.modelManager2.model_config(
                hidden_layers=self.hidden_layers2,
                hidden_activation=self.comboBox_activation2.currentText(),
                output_activation=self.comboBox_output_activation2.currentText(),
                learning_rate=self.doubleSpinBox_learning_rate2.value(),
                iter_max=self.spinBox_iter_max2.value(),
                optimizer=self.comboBox_optimizer2.currentText(),
                batch_size=self.spinBox_batch_size2.value(),
                l2_alpha=self.doubleSpinBox_l2_alpha2.value(),
                dropout=self.comboBox_dropout2.currentText()
                )
        #后台任务队列
        tasks,tasks_name=[],[]
        def dataset_info():
            classes_n=len(np.unique(self.modelManager2.train_y))
            message='''
                <div class="text" style=" text-align:center;">
                训练集<br>%d<br>测试集<br>%d<br>
                特征数<br>%d<br>分类数<br>%d</div>
                '''%(
                self.modelManager2.train_y.shape[0],
                self.modelManager2.test_y.shape[0],
                int(np.array(self.modelManager2.train_X.shape[1:]).prod()),
                classes_n
                )
            return message
        tasks.append(self.modelManager2.load_dataset)
        tasks.append(dataset_info)
        tasks.append(self.modelManager2.create_model)
        tasks.append(self.modelManager2.fit)
        tasks_name.append('加载%s数据集'%self.modelManager2.dataset_type)
        tasks_name.append('')
        tasks_name.append('创建模型')
        tasks_name.append('训练模型')
        tasks_feedback_trigger=[0,1,0,0]
        #创建后台任务
        self.background=WorkThread(
                tasks,tasks_name,tasks_feedback_trigger=tasks_feedback_trigger
                )
        #将信号发射方法和其他要执行的任务封装，设置为模型管理器的外部监视方法，
        #这样长耗时任务可以反馈进度(需要调度的对象提供该参数支持)
        #注：此方法会在子线程中运行，所以不能进行UI更新操作，会奔溃
        def external_monitor(messages):
            if type(messages)==type(''):
                self.background.task_trigger0.emit(messages)
            elif len(messages)==1:
                self.background.task_trigger0.emit(messages[0])
            else:
                message,score_h,cost_h=messages[0],messages[1],messages[2]
                x=range(len(score_h))
                y=np.array(score_h)
                self.training_score_view2.update_figure(x,y[:,0],y[:,1])
                y=np.array(cost_h)
                self.training_cost_view2.update_figure(x,y[:,0],y[:,1])
                self.background.task_trigger4.emit((message,))
        self.modelManager2.set_external_monitor(external_monitor)
        #绑定信号槽以更新监控窗口
        self.background.task_trigger0.connect(self.monitor_append2)
        self.background.task_trigger1.connect(self.update_dataset_info2)
        self.background.task_trigger4.connect(self.train_monitor2)
        self.background.thread_trigger.connect(self.update_background_status)
        #启动后台任务
        self.background.start()
    
    #重置模型
    def reset_model1(self):
        if self.background_status=='free':
            self.modelManager1.reset_model()
            self.monitor_clear1()
            self.fitting_result.initial_figure()
            self.neuron_action.initial_figure()
            self.training_score_view1.initial_figure()
            self.training_cost_view1.initial_figure()
        else:
            self.background_stop()
            
    def reset_model2(self):
        if self.background_status=='free':
            self.modelManager2.reset_model()
            self.monitor_clear2()
            self.training_score_view2.initial_figure()
            self.training_cost_view2.initial_figure()
        else:
            self.background_stop()
            
    #绘制结构图     
    def draw_structure1(self):
        #后台线程忙碌时进行阻断
        if self.background_status!='free':
            self.monitor_append1('\n后台有任务正在运行，请等待\n')
            return
        #后台任务队列
        def draw_task():
            if type(self.modelManager1.model)!=type(None):
                self.structure_view.update_figure(self.modelManager1.model)
                return '绘制完成'
            else:
                raise ValueError('模型尚未创建')
        tasks=[draw_task]
        tasks_name=['绘制网络结构图']
        tasks_feedback_trigger=[1]
        #创建后台任务
        self.background=WorkThread(tasks,tasks_name,
                                   tasks_feedback_trigger=tasks_feedback_trigger)
        #绑定信号槽以更新监控窗口
        self.background.task_trigger0.connect(self.monitor_append1)
        self.background.task_trigger1.connect(self.show_structure1)
        self.background.thread_trigger.connect(self.update_background_status)
        #启动后台任务
        self.background.start()
            
    #显示结构图
    def show_structure1(self,message):
        self.structure_view.draw()
        if len(self.modelManager1.model.layers)>7:
            message+='\n注意：多于5层的隐含层会导致显示不全'
        self.monitor_append1(message)
        
    #绘制神经元作用图            
    def draw_neuron_action1(self):
        #后台线程忙碌时进行阻断
        if self.background_status!='free':
            self.monitor_append1('\n后台有任务正在运行，请等待\n')
            return
        #后台任务队列
        def draw_task():
            if type(self.modelManager1.model)==type(None):
                raise ValueError('模型尚未创建')
            if self.comboBox_neurons1.currentIndex()==-1:
                raise ValueError('需要选择一个神经元')
            if self.comboBox_layer1.currentIndex()==0:
                layer_id=len(self.modelManager1.model.layers)-1
            else:
                layer_id=self.comboBox_layer1.currentIndex()
            neuron_id=self.comboBox_neurons1.currentIndex()
            x=self.modelManager1.train_X[:,0]
            y=self.modelManager1.train_X[:,1]
            x0,y0=generate_grid_data(x,y,broaden=0.2,steps=51)
            z0=grid_predict2(layer_id,neuron_id,x0,y0,self.modelManager1.model)
            self.neuron_action.update_figure(x0,y0,z0)
            return '绘制完成'
        tasks=[draw_task]
        tasks_name=['绘制神经元作用图']
        tasks_feedback_trigger=[1]
        #创建后台任务
        self.background=WorkThread(tasks,tasks_name,
                                       tasks_feedback_trigger=tasks_feedback_trigger)
        #绑定信号槽以更新监控窗口
        self.background.task_trigger0.connect(self.monitor_append1)
        self.background.task_trigger1.connect(self.neuron_action.draw)
        self.background.thread_trigger.connect(self.update_background_status)
        #启动后台任务
        self.background.start()
     
    #显示神经元作用图
    def show_neuron_action1(self,message):
        self.neuron_action.draw()
        self.monitor_append1(message)
    
    #在监视窗口添加消息
    def monitor_append1(self,message):
        if len(message)>0:
            self.textBrowser_monitor1.append(message)
            
    def monitor_append2(self,message):
        if len(message)>0:
            self.textBrowser_monitor2.append(message)
            
    #训练监控
    def train_monitor1(self,messages):
        self.monitor_append1(messages[0])
        self.fitting_result.draw()
        self.training_score_view1.draw()
        self.training_cost_view1.draw()

    def train_monitor2(self,messages):
        self.monitor_append2(messages[0])
        self.training_score_view2.draw()
        self.training_cost_view2.draw()
    
    #清空监控窗口
    def monitor_clear1(self):
        self.textBrowser_monitor1.clear()
        
    def monitor_clear2(self):
        self.textBrowser_monitor2.clear()
        
    #更新数据集信息
    def update_dataset_info2(self,message):
        self.textBrowser_dataset_info2.clear()
        self.textBrowser_dataset_info2.setHtml(message)
        self.textBrowser_dataset_info2.setFont(QFont("Microsoft YaHei"))
        #self.textBrowser_dataset_info2.setVerticalScrollBarPolicy(1)

    #数据集路径设置
    def set_dataset_path2(self):
        if self.dataset_path==None:
            start_path='C:\\'
        else:
            start_path=self.dataset_path
        directory=QFileDialog.getExistingDirectory(self,"选取文件夹",start_path) 
        if directory!='':
            self.dataset_path=directory
            self.monitor_append2('数据集路径设置为: '+directory)
    
    #图片选择
    def select_image2(self):
        if self.dataset_path==None:
            start_path='C:\\'
        else:
            start_path=self.dataset_path
        fileName,filetype=QFileDialog.getOpenFileName(
                self,"选取文件",start_path,"PNG Image (*.png);;JPG Image (*.jpg)") 
        if fileName!='':
            self.pixmap=self.modelManager2.load_one_image(fileName)
            self.monitor_append2('选中图片: '+fileName)
            self.label_image2.setPixmap(self.pixmap)
            self.label_image2.setScaledContents(True)
    
    #导出图片        
    def export_images2(self):
        #后台线程忙碌时进行阻断
        if self.background_status!='free':
            self.monitor_append2('\n后台有任务正在运行，请等待\n')
            return
        #清空监控窗口
        self.monitor_clear2()
        #载入设置
        self.modelManager2.dataset_config(
                dataset_type=self.comboBox_dataset_type2.currentText(),
                dataset_path=self.dataset_path
                )
        #后台任务队列
        tasks=[self.modelManager2.export_images]
        tasks_name=['导出图片'+str(self.modelManager2.dataset_type)]
        #创建后台任务
        self.background=WorkThread(tasks,tasks_name)
        #将信号发射方法设置为模型管理器的外部监视方法，
        #这样长耗时任务可以反馈进度(需要调度的对象提供该参数支持)
        self.modelManager2.set_external_monitor(self.background.task_trigger0.emit)
        #绑定信号槽以更新监控窗口
        self.background.task_trigger0.connect(self.monitor_append2)
        self.background.thread_trigger.connect(self.update_background_status)
        #启动后台任务
        self.background.start()
        
    #验证        
    def verify2(self):
        #后台线程忙碌时进行阻断
        if self.background_status!='free':
            self.monitor_append2('\n后台有任务正在运行，请等待\n')
            return
        #载入设置
        self.modelManager2.dataset_config(
                dataset_type=self.comboBox_dataset_type2.currentText(),
                dataset_path=self.dataset_path
                )
        #后台任务队列
        tasks=[self.modelManager2.predict]
        tasks_name=['测试集验证']
        #创建后台任务
        self.background=WorkThread(tasks,tasks_name)
        #将信号发射方法设置为模型管理器的外部监视方法，
        #这样长耗时任务可以反馈进度(需要调度的对象提供该参数支持)
        self.modelManager2.set_external_monitor(self.background.task_trigger4.emit)
        #绑定信号槽以更新监控窗口
        self.background.task_trigger0.connect(self.monitor_append2)
        self.background.task_trigger4.connect(self.verify_output2)
        self.background.thread_trigger.connect(self.update_background_status)
        #启动后台任务
        self.background.start()
        
    #单图验证        
    def verify_one2(self):
        #后台线程忙碌时进行阻断
        if self.background_status!='free':
            self.monitor_append2('\n后台有任务正在运行，请等待\n')
            return
        try:
            message=self.modelManager2.predict_one()
            self.verify_output2(message)
        except Exception as e:
            self.monitor_append2('执行出错: '+str(e)+'\n')
    
    #验证输出    
    def verify_output2(self,message):
        if type(message)==type(()):
            self.pixmap=message[0]
            self.label_image2.setPixmap(self.pixmap)
            self.label_image2.setScaledContents(True)
            infos=message[1].split('|')
        else:
            infos=message.split('|')  
        if self.modelManager2.dataset_type=='mnist':
            self.textBrowser_class2.setHtml('''
                <div class="text" 
                style=" text-align:center;
                        font-size:60px;
                        line-height:90px;
                        font-weight:bold;">
                %s</div>'''%infos[0])
            
        else:
            self.textBrowser_class2.setHtml('''
                <div class="text" 
                style=" text-align:center;
                        font-size:30px;
                        line-height:75px;
                        font-weight:bold;">
                %s</div>'''%infos[0])
        self.textBrowser_class_prob2.setHtml('''
                <div class="text" 
                style=" text-align:center;
                        font-size:20px;">
                %s</div>'''%infos[1])
        if infos[2]=='正确':
            color='green'
        elif infos[2]=='错误':
            color='red'
        else:
            color='black'
        self.textBrowser_check2.setHtml('''
            <div class="text" 
            style=" text-align:center;
                    font-size:30px;
                    line-height:75px;
                    font-weight:bold;
                    color:%s;">
            %s</div>'''%(color,infos[2]))
        self.textBrowser_accuracy2.setHtml('''
            <div class="text" 
            style=" text-align:center;
                    font-size:20px;
                    line-height:30px;">
            %s</div>'''%infos[3])
    
    #打开画板
    def open_paintboard(self):
        #后台线程忙碌时进行阻断
        if self.background_status!='free':
            self.monitor_append2('\n后台有任务正在运行，请等待\n')
            return
        if self.modelManager2.dataset_type=='mnist':
            self.paintboard=PaintBoard()
        else:
            self.paintboard=PaintBoard(mode='rgb')
        #设置模态（阻塞其他窗体直到该窗体关闭）
        self.paintboard.setWindowModality(Qt.ApplicationModal)
        self.paintboard.setWindowTitle('画板')
        self.paintboard.return_trigger.connect(self.get_pboard_image)
        self.paintboard.show()
    
    #获取画板图片    
    def get_pboard_image(self,pixmap):
        #缩放至支持的尺寸
        #注：缩放导致图片细节会丢失很多
        try:
            pixmap=self.modelManager2.load_one_pixmap(pixmap)
            self.label_image2.setPixmap(pixmap)
            self.label_image2.setScaledContents(True)
        except Exception as e:
            self.monitor_append2('执行出错: '+str(e)+'\n')
  
#后台任务线程      
class WorkThread(QThread):
    # 线程信号和任务信号
    thread_trigger=pyqtSignal(str)
    task_trigger0=pyqtSignal(str)
    task_trigger1=pyqtSignal(str)
    task_trigger2=pyqtSignal(tuple)
    task_trigger3=pyqtSignal(tuple)
    task_trigger4=pyqtSignal(tuple)

    #tasks: 待执行任务队列
    #tasks_name: 待执行任务名称，可选
    #ignore_exception: 是否忽略异常继续执行后续任务，默认不忽略
    def __init__(self,tasks,tasks_name=None,
                 tasks_status_trigger=None,
                 tasks_feedback_trigger=None,
                 ignore_exception=False):
        super(WorkThread, self).__init__()
        self.tasks=tasks
        self.tasks_name=tasks_name
        if tasks_status_trigger==None:
            self.tasks_status_trigger=[0 for i in range(len(tasks))]
        else:
            self.tasks_status_trigger=tasks_status_trigger
        if tasks_feedback_trigger==None:
            self.tasks_feedback_trigger=[0 for i in range(len(tasks))]
        else:
            self.tasks_feedback_trigger=tasks_feedback_trigger
        self.ignore_exception=ignore_exception

    def run(self):
        #线程开始运行，发送线程忙碌信号
        self.thread_trigger.emit('busy')
        for i,task in enumerate(self.tasks):
            try:
                #发送任务开始的信号(统一)
                if type(self.tasks_name)!=type(None):
                    if self.tasks_name[i] not in ['',None]:
                        message='正在执行: '+self.tasks_name[i]
                        self.task_trigger_emit(self.tasks_status_trigger[i],message)
                #执行任务
                message=task()
                #完成一个任务发送一次信号，提供任务的反馈信息(独立)
                self.task_trigger_emit(self.tasks_feedback_trigger[i],message)
            except Exception as e:
                #出错时反馈异常信息，根据设置确定是否中断后续任务(统一)
                message='执行出错: '+str(e)+'\n'
                self.task_trigger_emit(self.tasks_status_trigger[i],message)
                if self.ignore_exception==False:
                    raise e
        #线程结束运行，发送线程空闲信号
        self.thread_trigger.emit('free')
        self.quit()
    
    #发送任务信号  
    def task_trigger_emit(self,trigger,message):
        if trigger==1:
            self.task_trigger1.emit(message)
        elif trigger==2:
            self.task_trigger2.emit(message)
        elif trigger==3:
            self.task_trigger3.emit(message)
        elif trigger==4:
            self.task_trigger4.emit(message)
        else:
            self.task_trigger0.emit(message)

#运行    
if ( __name__ == '__main__' ):
    app = None
    if ( not app ):
        app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    if ( app ):
        app.exec_()
