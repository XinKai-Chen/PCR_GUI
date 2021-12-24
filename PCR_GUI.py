from PySide2.QtWidgets import QApplication, QMainWindow, QGraphicsScene, QFileDialog, QMessageBox, QPlainTextEdit, \
    QWidget
from PySide2.QtWidgets import QFileDialog
from PySide2.QtUiTools import QUiLoader
from PySide2.QtCore import QFile, QDir, QSettings, Qt, QObject, Signal, Slot
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import sys
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time
import threading

from scipy import interpolate
import pylab as pl
import keras
import numpy
from sklearn.preprocessing import MinMaxScaler
import math
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import Bidirectional

matplotlib.use("Qt5Agg")  # 声明使用QT5


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        # self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch': [], 'epoch': []}
        # self.val_acc = {'batch':[], 'epoch':[]}
        self.saveLoss = []
        self.savevalLoss = []

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        # self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        # self.saveLoss.append(logs.get('loss'))
        # self.saveLoss.append(logs.get('loss'))

    # self.savevalLoss.append(logs.get('val_loss'))
    # self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        # self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        # self.val_acc['epoch'].append(logs.get('val_acc'))
        self.saveLoss.append(logs.get('loss'))
        self.savevalLoss.append(logs.get('val_loss'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        # plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'r', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            # plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'b', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('loss')
        plt.legend(loc="upper right")
        # plt.show()


def readFileData(file_path):
    """
        这个函数没用到，用于之后拓展
        从文件名目录地址中读取出文件中的数据并返回数据表

        :param file_path: 存放文件路径的list
        :return: x： 扩增曲线x值
                y： 扩增曲线y值
                Ct： 扩增曲线Ct值
    """
    global data
    if not os.path.exists(file_path):
        print(" this file path {} is not exists ".format(file_path))
        return None
    dir_path, full_file_name = os.path.split(file_path)
    file_name, extension = os.path.splitext(full_file_name)
    if extension not in ('.csv', '.xlsx'):
        print(" this file {} is not the correct file type ".format(full_file_name))
        return None
    else:
        if extension == '.csv':
            data = pd.read_csv(file_path)
        elif extension == '.xlsx':
            data = pd.read_excel(file_path)
    x = data.iloc[:, 0:40]
    # 如果没有label和Ct栏则y、CT为空，所以不需要出错处理
    label = data.iloc[:, data.columns == 'label']
    Ct = data.iloc[:, data.columns == 'Ct']
    return x, label, Ct


# 划分为训练可用的数据 用前time_step数据预测后一步数据
def create_dataset(dataset, time_step):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return numpy.array(dataX), numpy.array(dataY)


class CommunicationSignals(QObject):
    """
    自定义Qt通信信号类
    """
    updataGraphSig = Signal(object, object, int, str)
    addShowText = Signal(str)

    def __init__(self):
        super().__init__()


# @todo
class TrainingHandler(threading.Thread):
    """
    模型训练线程
    """

    def __init__(self, *args):
        # super(TrainingHandler, self).__init__()
        threading.Thread.__init__(self)
        self.__flag = threading.Event()  # 用于暂停线程的标识
        self.__flag.set()  # 设置为True
        self.__running = threading.Event()  # 用于停止线程的标识
        self.__running.set()  # 将running设置为True

        self.signals = args[0]
        self.current_path = args[1]
        self.model_save_directory = args[2]
        self.settings = args[3]
        self.file_paths = args[4]

    def getRunningState(self):
        if not self.__running.is_set():
            return False
        else:
            return True

    # @todo 线程执行程序
    def run(self):
        pcr_training_cycle_number = int(self.settings.training_cycle_number)
        interpolation_method = self.settings.interpolation_method
        traintestsplit_ratio = float(self.settings.trainTestSplit_ratio)
        prediction_timestep = int(self.settings.prediction_timestep)
        interpolation_number = int(self.settings.interpolation_number)
        totalinterpolation = pcr_training_cycle_number * interpolation_number
        for i in self.file_paths:
            dir_path, full_file_name = os.path.split(i)
            file_name, extension = os.path.splitext(full_file_name)
            training_data, label, ct = readFileData(i)
            for row in range(training_data.shape[0]):
                if not self.__running.is_set():
                    return
                self.__flag.wait()  # 为True时立即返回, 为False时阻塞直到内部的标识位为True后返回
                # @todo
                self.signals.addShowText.emit(str("Training PCR amplification curve prediction model at "
                                                  "row {} in file {}".format(row, str(i))))
                y = np.array(training_data.iloc[row, 0:40]).flatten()
                x = np.linspace(1, y.shape[0], y.shape[0])
                # @todo 目前可以正常发送信号，记录
                self.signals.updataGraphSig.emit(x, y, 0, "PCR amplification curve at "
                                                          "row {} in file {}".format(row, str(i)))
                # settings.plot_graph(x, y, None, graph_title="PCR amplification curve at "
                #                                             "row {} in file {}".format(row, str(i)))
                xnew = np.linspace(1, y.shape[0], totalinterpolation)
                f = interpolate.interp1d(x, y, kind=interpolation_method)
                # ‘slinear’, ‘quadratic’ and ‘cubic’ refer to a spline interpolation of first, second or third order)
                ynew = f(xnew)
                ynew_dataframe = pd.DataFrame(ynew)
                # 归一化
                scaler = MinMaxScaler(feature_range=(0, 1))
                ynew_dataframe = scaler.fit_transform(np.array(ynew_dataframe).reshape(-1, 1))
                # 训练集和测试集划分
                training_size = int(len(ynew_dataframe) * traintestsplit_ratio)
                test_size = len(ynew_dataframe) - training_size
                train_data, test_data = ynew_dataframe[0:training_size, :], ynew_dataframe[
                                                                            training_size:len(ynew_dataframe), :]

                # reshape into X=t,t+1,t+2,t+3 and Y=t+4
                time_step = prediction_timestep
                X_train, y_train = create_dataset(train_data, time_step)
                X_test, y_test = create_dataset(test_data, time_step)

                X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
                X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
                model = Sequential()

                # model.add(SimpleRNN(500, return_sequences=True))
                # model.add(SimpleRNN(500))

                model.add(LSTM(500, return_sequences=True, input_shape=(prediction_timestep, 1)))
                model.add(Bidirectional(LSTM(500, return_sequences=True), merge_mode='concat'))
                model.add(LSTM(500))

                model.add(Dense(1))
                model.compile(loss='mean_squared_error', optimizer='adam')

                history = LossHistory()
                model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=50, verbose=1,
                          callbacks=[history])
                # @todo 修改模型保存路径
                model_save_path = self.model_save_directory + "/{}_row_{}.h5".format(file_name, row)
                model.save(model_save_path)
                self.signals.addShowText.emit(str(" PCR amplification curve prediction model at "
                                                  "row {} in file {} training successfully, "
                                                  "model save in {}".format(row, str(i), model_save_path)))

                # 训练值和测试值预测分析
                train_predict = model.predict(X_train)
                test_predict = model.predict(X_test)
                test_predict1 = pd.DataFrame(test_predict)
                # test_predict1.to_csv('test_predict.csv', index=False)
                train_predict = scaler.inverse_transform(train_predict)
                test_predict = scaler.inverse_transform(test_predict)

                # 误差计算
                training_loss = math.sqrt(mean_squared_error(y_train, train_predict))
                test_loss = math.sqrt(mean_squared_error(y_test, test_predict))
                self.signals.addShowText.emit(
                    str("training loss is {}, test loss is {}".format(training_loss, test_loss)))
                look_back = prediction_timestep
                trainPredictPlot = numpy.empty_like(ynew_dataframe)
                trainPredictPlot[:, :] = np.nan
                # 从第八个数据开始，用8个x值数据预测y值数据，将所有x值预测的y值数据填入trainPredictPlot中
                trainPredictPlot[look_back:len(train_predict) + look_back, :] = train_predict  # 将训练数据预测值填入空数组中用于绘图
                # shift test predictions for plotting
                testPredictPlot = numpy.empty_like(ynew_dataframe)
                testPredictPlot[:, :] = numpy.nan
                # 从测试数据+8个后的数据开始，用8个x值数据预测y值数据，将所有测试数据x值预测的y值数据填入testPredictPlot中
                testPredictPlot[len(train_predict) + (look_back * 2) + 1:len(ynew_dataframe) - 1, :] = test_predict
                # plot baseline and predictions
                # plt.plot(scaler.inverse_transform(df),'blue')
                # plt.plot(trainPredictPlot,'red')
                # plt.plot(testPredictPlot,'green')
                # plt.show()

    def pause(self):
        self.__flag.clear()  # 设置为False, 让线程阻塞

    def resume(self):
        self.__flag.set()  # 设置为True, 让线程停止阻塞

    def stop(self):
        self.__flag.set()  # 将线程从暂停状态恢复, 如何已经暂停的话
        self.__running.clear()  # 设置为False


# @todo
class ExperimentHandler(threading.Thread):
    """
    实验线程
    """

    def __init__(self, *args):
        # super(TrainingHandler, self).__init__()
        threading.Thread.__init__(self)
        self.__flag = threading.Event()  # 用于暂停线程的标识
        self.__flag.set()  # 设置为True
        self.__running = threading.Event()  # 用于停止线程的标识
        self.__running.set()  # 将running设置为True

        self.signals = args[0]
        self.current_path = args[1]
        self.model_save_directory = args[2]
        self.settings = args[3]
        self.dirMonitored = args[4]

        self.event_handler = ExperimentFileHandler(self.signals, self.current_path, self.model_save_directory,
                                                   self.settings, self.dirMonitored)  # 监控处理事件的方法,这个类非常重要,可以根据自己的需要重写

    def getRunningState(self):
        if not self.__running.is_set():
            return False
        else:
            return True

    def run(self):
        self.observer = Observer()  # 定义监控类,多线程类 thread class
        self.observer.schedule(self.event_handler, self.dirMonitored, recursive=True)  # 指定监控路径/触发对应的监控事件类
        self.observer.start()  # 将observer运行在同一个线程之内,不阻塞主进程运行,可以调度observer来停止该线程
        print("begin monitor")
        try:
            while True:
                if not self.__running.is_set():
                    return
                self.__flag.wait()
                time.sleep(1)  # 监控频率（1s1次，根据自己的需求进行监控）
        except KeyboardInterrupt:
            self.observer.stop()
        self.observer.join()

    def pause(self):
        self.__flag.clear()  # 设置为False, 让线程阻塞

    def resume(self):
        self.__flag.set()  # 设置为True, 让线程停止阻塞

    def stop(self):
        self.__flag.set()  # 将线程从暂停状态恢复, 如何已经暂停的话
        self.__running.clear()  # 设置为False


class Settings:
    """
    界面功能实现的配置类，其中存储了PCR模型训练、预测相关的所有参数
    """

    def __init__(self):
        super().__init__()
        self.current_path = QDir.currentPath()
        self.settings_path = self.current_path + "/config.ini"
        if not os.path.exists(self.settings_path):
            self.initSettings()
        self.settings = QSettings(self.settings_path, QSettings.IniFormat)
        self.pcrAmplificationCycleNumber_Range = [0, 10000]
        self.allInterpolationMethod = ["nearest", "zero", "slinear", "quadratic", "cubic"]
        self.traintestsplitRatio_Range = [0, 1]
        self.predictionTimestep_Range = [0, 10000]
        self.interpolationNumber_Range = [0, 10000]

        self.file_path = self.settings.value("file and directory/Default file path")
        self.directory_path = self.settings.value("file and directory/Default Directory path")

        self.full_cycle_number = self.settings.value("Training and Experiment Settings"
                                                     "/Default PCR amplification Full Cycle Number")
        self.training_cycle_number = self.settings.value("Training and Experiment Settings"
                                                         "/Default PCR Training Cycle Number")
        self.interpolation_method = self.settings.value("Training and Experiment Settings/"
                                                        "Default Interpolation method")
        self.interpolation_number = self.settings.value("Training and Experiment Settings/"
                                                        "Default Interpolation Number")
        self.trainTestSplit_ratio = self.settings.value("Training and Experiment Settings/"
                                                        "Default Train-Test-Split Ratio")
        self.prediction_timestep = self.settings.value("Training and Experiment Settings/"
                                                       "Default Prediction Timestep")
        self.initial_prediction_cycle_number = self.settings.value("Training and Experiment Settings/"
                                                                   "Default Initial prediction cycle number")

    """
    初始化配置，如果没有配置文件则在本地自动生成配置文件并设置默认值，如果有配置文件转到readSettingsFile函数读取配置文件参数值
    """

    def initSettings(self):
        self.settings.beginGroup("file and directory")
        self.settings.setValue("Default file path", self.current_path + "/train.csv")
        self.settings.setValue("Default Directory path", self.current_path)
        self.settings.endGroup()

        self.settings.beginGroup("Training and Experiment Settings")
        self.settings.setValue("Default PCR amplification Full Cycle Number", 40)
        self.settings.setValue("Default PCR Training Cycle Number", 21)
        self.settings.setValue("Default Interpolation method", "cubic")
        self.settings.setValue("Default Train-Test-Split Ratio", 0.92)
        self.settings.setValue("Default Prediction Timestep", 8)
        self.settings.setValue("Default Interpolation Number", 10)
        self.settings.setValue("Default Initial prediction cycle number", 18)
        self.settings.endGroup()

    " 读取配置文件初始化设置 "

    def readSettingsFile(self):
        self.settings = QSettings(self.settings_path, QSettings.IniFormat)


class MainWidgetSettings(QWidget):
    """
    GUI配置界面类，保存配置文件并可以自动初始化读取和保存
    """

    def __init__(self):
        super().__init__()
        self.current_path = QDir.currentPath()
        self.Ui_MainWindow_File_Path = self.current_path + "/UI/PCR_Settings.ui"
        Ui_MainWindow_File = QFile(self.Ui_MainWindow_File_Path)
        Ui_MainWindow_File.open(QFile.ReadOnly)
        Ui_MainWindow_File.close()
        self.ui = QUiLoader().load(Ui_MainWindow_File)
        self.ui.setParent(self)
        self.settings_path = self.current_path + "/config.ini"
        self.settings = Settings()
        self.initUI()

        self.default_filePath = self.settings.settings.value("file and directory/Default file path")
        self.default_directoryPath = self.settings.settings.value("file and directory/Default Directory path")
        self.default_pcrAmplificationCycleNumber = self.settings.settings.value(
            "Training and Experiment Settings/Default PCR amplification Full Cycle Number")
        self.default_pcrTrainingCycleNumber = self.settings.settings.value(
            "Training and Experiment Settings/Default PCR Training Cycle Number")
        self.default_interpolationMethod = self.settings.settings.value(
            "Training and Experiment Settings/Default Interpolation method")
        self.default_traintestsplitRatio = self.settings.settings.value(
            "Training and Experiment Settings/Default Train-Test-Split Ratio")
        self.default_predictionTimestep = self.settings.settings.value(
            "Training and Experiment Settings/Default Prediction Timestep")
        self.default_interpolationNumber = self.settings.settings.value(
            "Training and Experiment Settings/Default Interpolation Number")
        self.default_initialPredictionCycleNumber = self.settings.settings.value("Training and Experiment Settings/"
                                                                                 "Default Initial prediction cycle number")

        self.showSettings()
        self.connections()

    def initUI(self):
        self.ui.spinBox_pcrAmplificationCycleNumber.setRange(self.settings.pcrAmplificationCycleNumber_Range[0],
                                                             self.settings.pcrAmplificationCycleNumber_Range[1])
        self.ui.spinBox_pcrTrainingCycleNumber.setRange(self.settings.pcrAmplificationCycleNumber_Range[0],
                                                        self.settings.pcrAmplificationCycleNumber_Range[1])
        for method in self.settings.allInterpolationMethod:
            self.ui.comboBox_interpolationMethod.addItem(method)
        self.ui.doubleSpinBox_traintestsplitRatio.setSingleStep(0.01)
        self.ui.doubleSpinBox_traintestsplitRatio.setRange(self.settings.traintestsplitRatio_Range[0],
                                                           self.settings.traintestsplitRatio_Range[1])
        self.ui.spinBox_predictionTimestep.setRange(self.settings.predictionTimestep_Range[0],
                                                    self.settings.predictionTimestep_Range[1])
        self.ui.spinBox_interpolationNumber.setRange(self.settings.interpolationNumber_Range[0],
                                                     self.settings.interpolationNumber_Range[1])
        self.ui.spinBox_initialPredictionCycleNumber.setRange(0, int(self.settings.full_cycle_number))

    def connections(self):
        self.ui.pushButton_save.clicked.connect(self.saveSettings)

        self.ui.lineEdit_filePath.textChanged.connect(self.modifySettings)
        self.ui.lineEdit_directoryPath.textChanged.connect(self.modifySettings)
        self.ui.spinBox_pcrAmplificationCycleNumber.valueChanged.connect(self.modifySettings)
        self.ui.spinBox_pcrTrainingCycleNumber.valueChanged.connect(self.modifySettings)
        self.ui.comboBox_interpolationMethod.currentTextChanged.connect(self.modifySettings)
        self.ui.doubleSpinBox_traintestsplitRatio.valueChanged.connect(self.modifySettings)
        self.ui.spinBox_predictionTimestep.valueChanged.connect(self.modifySettings)
        self.ui.spinBox_interpolationNumber.valueChanged.connect(self.modifySettings)
        self.ui.spinBox_initialPredictionCycleNumber.valueChanged.connect(self.modifySettings)

    def showSettings(self):
        self.ui.lineEdit_filePath.setText(self.default_filePath)
        self.ui.lineEdit_directoryPath.setText(self.default_directoryPath)

        self.ui.spinBox_pcrAmplificationCycleNumber.setValue(int(self.default_pcrAmplificationCycleNumber))
        self.ui.spinBox_pcrTrainingCycleNumber.setValue(int(self.default_pcrTrainingCycleNumber))
        self.ui.comboBox_interpolationMethod.setCurrentText(self.default_interpolationMethod)
        self.ui.doubleSpinBox_traintestsplitRatio.setValue(float(self.default_traintestsplitRatio))
        self.ui.spinBox_predictionTimestep.setValue(int(self.default_predictionTimestep))
        self.ui.spinBox_interpolationNumber.setValue(int(self.default_interpolationNumber))
        self.ui.spinBox_initialPredictionCycleNumber.setValue(int(self.default_initialPredictionCycleNumber))

    def modifySettings(self):
        self.default_filePath = self.ui.lineEdit_filePath.text()
        self.default_directoryPath = self.ui.lineEdit_filePath.text()
        self.default_pcrAmplificationCycleNumber = self.ui.spinBox_pcrAmplificationCycleNumber.value()
        self.default_pcrTrainingCycleNumber = self.ui.spinBox_pcrTrainingCycleNumber.value()
        self.default_interpolationMethod = self.ui.comboBox_interpolationMethod.currentText()
        self.default_traintestsplitRatio = self.ui.doubleSpinBox_traintestsplitRatio.value()
        self.default_predictionTimestep = self.ui.spinBox_predictionTimestep.value()
        self.default_interpolationNumber = self.ui.spinBox_interpolationNumber.value()
        self.default_initialPredictionCycleNumber = self.ui.spinBox_initialPredictionCycleNumber.value()

    def saveSettings(self):
        self.settings.settings.setValue("file and directory/Default file path", self.default_filePath)
        self.settings.settings.setValue("file and directory/Default Directory path", self.default_directoryPath)
        self.settings.settings.setValue("Training and Experiment Settings/Default PCR amplification Full Cycle Number",
                                        self.default_pcrAmplificationCycleNumber)
        self.settings.settings.setValue("Training and Experiment Settings/Default PCR Training Cycle Number",
                                        self.default_pcrTrainingCycleNumber)
        self.settings.settings.setValue("Training and Experiment Settings/Default Interpolation method",
                                        self.default_interpolationMethod)
        self.settings.settings.setValue("Training and Experiment Settings/Default Train-Test-Split Ratio",
                                        self.default_traintestsplitRatio)
        self.settings.settings.setValue("Training and Experiment Settings/Default Prediction Timestep",
                                        self.default_predictionTimestep)
        self.settings.settings.setValue("Training and Experiment Settings/Default Interpolation Number",
                                        self.default_interpolationNumber)
        self.settings.settings.setValue("Training and Experiment Settings/Default Initial prediction cycle number",
                                        self.default_initialPredictionCycleNumber)
        self.settings.settings.sync()


class MyFigureCanvas(FigureCanvas):
    """
    通过继承FigureCanvas类，使得该类既是一个PyQt5的Qwidget，又是一个matplotlib的FigureCanvas，这是连接pyqt5与matplotlib的关键
    """

    def __init__(self, parent=None, width=10, height=5, xlim=(0, 2500), ylim=(-2, 2), dpi=100):
        # 创建一个Figure画布
        fig = plt.Figure(figsize=(width, height), dpi=dpi, tight_layout=True)  # tight_layout: 用于去除画图时两边的空白

        FigureCanvas.__init__(self, fig)  # 初始化父类
        self.setParent(parent)

        self.axes = fig.add_subplot(111)  # 调用figure下面的add_subplot方法，类似于matplotlib.pyplot下面的subplot方法
        self.axes.spines['top'].set_visible(False)  # 去掉上面的横线
        self.axes.spines['right'].set_visible(False)
        self.axes.set_xlim(xlim)
        self.axes.set_ylim(ylim)


# @todo python自定义类继承Thread类实现线程可暂停
class Job(threading.Thread):
    """
    可用于python中Thread线程启动、停止、暂停和恢复的类模板
    """

    def __init__(self, *args):
        super(Job, self).__init__()
        self.__flag = threading.Event()  # 用于暂停线程的标识
        self.__flag.set()  # 设置为True
        self.__running = threading.Event()  # 用于停止线程的标识
        self.__running.set()  # 将running设置为True

    def run(self):
        while self.__running.isSet():
            self.__flag.wait()  # 为True时立即返回, 为False时阻塞直到内部的标识位为True后返回
            time.sleep(1)

    def pause(self):
        self.__flag.clear()  # 设置为False, 让线程阻塞

    def resume(self):
        self.__flag.set()  # 设置为True, 让线程停止阻塞

    def stop(self):
        self.__flag.set()  # 将线程从暂停状态恢复, 如何已经暂停的话
        self.__running.clear()  # 设置为False


class MainWindowTraining(QMainWindow):
    """
    预测模型训练界面类
    """
    returnMainWindowSignal = Signal()

    def __init__(self, settings):
        super().__init__()
        # self.training_handler = TrainingHandler()
        self.current_path = QDir.currentPath()
        self.Ui_MainWindow_File_Path = self.current_path + "/UI/PCR_Training.ui"
        Ui_MainWindow_File = QFile(self.Ui_MainWindow_File_Path)
        Ui_MainWindow_File.open(QFile.ReadOnly)
        Ui_MainWindow_File.close()
        self.ui = QUiLoader().load(Ui_MainWindow_File)
        self.ui.setParent(self)
        self.file_type = ('.csv', '.xlsx')
        self.settings = settings
        self.ui.plainTextEdit_display.appendPlainText("current directory path is " + str(self.current_path) + "\n")
        self.file_paths = []
        self.files_number = 0

        self.pcr_training_cycle_number = int(self.settings.training_cycle_number)
        self.interpolation_method = self.settings.interpolation_method
        self.traintestsplit_ratio = float(self.settings.trainTestSplit_ratio)
        self.prediction_timestep = int(self.settings.prediction_timestep)
        self.interpolation_number = int(self.settings.interpolation_number)

        self.initUI()
        # 初始化 gv_visual_data 的显示
        self.gv_visual_data_content = MyFigureCanvas(width=self.ui.graphicsView_plotGraph.width() / 101,
                                                     height=self.ui.graphicsView_plotGraph.height() / 101,
                                                     xlim=(0, 2 * np.pi),
                                                     ylim=(-1, 1))
        self.initGraphPloting()

        self.signals = CommunicationSignals()
        self.model_save_directory = self.current_path + "/Model_Save"

        self.training_handler = TrainingHandler(self.signals, self.current_path,
                                                self.model_save_directory, self.settings, self.file_paths)
        self.connections()

    def initUI(self):
        self.ui.spinBox_pcrTrainingCycleNumber.setRange(0, int(self.settings.full_cycle_number))

        for method in self.settings.allInterpolationMethod:
            self.ui.comboBox_interpolationMethod.addItem(method)
        self.ui.doubleSpinBox_traintestsplitRatio.setSingleStep(0.01)
        self.ui.doubleSpinBox_traintestsplitRatio.setRange(self.settings.traintestsplitRatio_Range[0],
                                                           self.settings.traintestsplitRatio_Range[1])
        self.ui.spinBox_predictionTimestep.setRange(self.settings.predictionTimestep_Range[0],
                                                    self.settings.predictionTimestep_Range[1])
        self.ui.spinBox_interpolationNumber.setRange(self.settings.interpolationNumber_Range[0],
                                                     self.settings.interpolationNumber_Range[1])

        self.ui.spinBox_pcrTrainingCycleNumber.setValue(self.pcr_training_cycle_number)
        self.ui.comboBox_interpolationMethod.setCurrentText(self.interpolation_method)
        self.ui.doubleSpinBox_traintestsplitRatio.setValue(self.traintestsplit_ratio)
        self.ui.spinBox_predictionTimestep.setValue(self.prediction_timestep)
        self.ui.spinBox_interpolationNumber.setValue(self.interpolation_number)

    def connections(self):
        self.ui.pushButton_ChooseFile.clicked.connect(self.readFile)
        self.ui.pushButton_ChooseDir.clicked.connect(self.readDir)
        self.ui.pushButton_start.clicked.connect(self.run)
        self.ui.pushButton_stop.clicked.connect(self.stop)
        self.ui.pushButton_pause.clicked.connect(self.pause)

        self.ui.spinBox_pcrTrainingCycleNumber.valueChanged.connect(self.modifySettings)
        self.ui.comboBox_interpolationMethod.currentTextChanged.connect(self.modifySettings)
        self.ui.doubleSpinBox_traintestsplitRatio.valueChanged.connect(self.modifySettings)
        self.ui.spinBox_predictionTimestep.valueChanged.connect(self.modifySettings)
        self.ui.spinBox_interpolationNumber.valueChanged.connect(self.modifySettings)

        self.training_handler.signals.updataGraphSig.connect(self.plot_graph)
        self.training_handler.signals.addShowText.connect(self.addShowText)
        self.ui.pushButton_back.clicked.connect(self.returnToMainWindow)
        self.ui.pushButton_clearDisplay.clicked.connect(lambda arg=self: self.ui.plainTextEdit_display.setPlainText(""))

    def modifySettings(self):
        self.pcr_training_cycle_number = self.ui.spinBox_pcrTrainingCycleNumber.value()
        self.interpolation_method = self.ui.comboBox_interpolationMethod.currentText()
        self.traintestsplit_ratio = self.ui.doubleSpinBox_traintestsplitRatio.value()
        self.prediction_timestep = self.ui.spinBox_predictionTimestep.value()
        self.interpolation_number = self.ui.spinBox_interpolationNumber.value()

        self.settings.pcr_training_cycle_number = self.ui.spinBox_pcrTrainingCycleNumber.value()
        self.settings.interpolation_method = self.ui.comboBox_interpolationMethod.currentText()
        self.settings.traintestsplit_ratio = self.ui.doubleSpinBox_traintestsplitRatio.value()
        self.settings.prediction_timestep = self.ui.spinBox_predictionTimestep.value()
        self.settings.interpolation_number = self.ui.spinBox_interpolationNumber.value()

    def readFile(self):
        self.file_paths.clear()
        filePath, _ = QFileDialog.getOpenFileName(
            self,  # 父窗口对象
            "Please choose the file you want to train",  # 标题
            self.current_path,  # 起始目录
            "file type (*.csv *.xlsx )"  # 选择类型过滤项，过滤内容在括号中
        )
        self.file_paths.append(filePath)
        self.files_number = 1
        self.ui.plainTextEdit_display.appendPlainText(str(filePath) + "\n")

    def readDir(self):
        self.file_paths.clear()
        self.files_number = 0
        # 选择文件夹
        dir_path = QFileDialog.getExistingDirectory(self, 'open the directory', self.current_path)

        self.ui.plainTextEdit_display.appendPlainText(" The directory path you choose is " + str(dir_path) + "\n")
        self.ui.plainTextEdit_display.appendPlainText("It contains the following available files" + "\n")

        # 读取文件夹文件
        self.file_paths.clear()
        for root, dirs, files in os.walk(dir_path, topdown=False):
            for file in files:
                (filename, extension) = os.path.splitext(file)
                if extension not in self.file_type:
                    continue
                file_path = os.path.join(root, file)
                self.file_paths.append(file_path)
                self.files_number = self.files_number + 1
                self.ui.plainTextEdit_display.appendPlainText(str(file) + "\n")

        if len(self.file_paths) <= 0:
            warning_text = " This is an empty directory, please select another directory "
            self.ui.plainTextEdit_display.appendPlainText(str(warning_text) + "\n")
            return

    def initGraphPloting(self):
        # 加载的图形（FigureCanvas）不能直接放到graphicview控件中，必须先放到graphicScene，然后再把graphicscene放到graphicview中
        self.graphic_scene = QGraphicsScene()  # 创建一个QGraphicsScene
        self.graphic_scene.addWidget(
            self.gv_visual_data_content)  # 把图形放到QGraphicsScene中，注意：图形是作为一个QWidget放到放到QGraphicsScene中的
        self.ui.graphicsView_plotGraph.setScene(self.graphic_scene)  # 把QGraphicsScene放入QGraphicsView

    @Slot(object, object, int, str)
    def plot_graph(self, x, y, Ct=0, graph_title='PCR amplification curve'):
        """


        :return: None
        """
        self.ui.plainTextEdit_display.appendPlainText(str(graph_title) + "\n")
        self.gv_visual_data_content.axes.clear()  # 由于图片需要反复绘制，所以每次绘制前清空，然后绘图
        self.gv_visual_data_content.axes.plot(x, y)
        self.gv_visual_data_content.axes.set_title(graph_title)
        self.gv_visual_data_content.draw()  # 刷新画布显示图片，否则不刷新显示

    @Slot(str)
    def addShowText(self, showText):
        self.ui.plainTextEdit_display.appendPlainText(showText + "\n")

    def run(self):
        if not self.file_paths:
            print("file list is empty, please select file or directory")
            self.ui.plainTextEdit_display.appendPlainText(
                str("file list is empty, please select file or directory") + "\n")
            return
            # @todo 修改
        self.ui.plainTextEdit_display.appendPlainText(
            str("Start training PCR amplification curve prediction model") + "\n")
        if not self.training_handler.is_alive():
            if self.training_handler.getRunningState():
                self.ui.plainTextEdit_display.appendPlainText("training start" + "\n")
                self.training_handler.start()
            else:
                self.ui.plainTextEdit_display.appendPlainText("training restart" + "\n")
                self.training_handler = TrainingHandler(self.signals, self.current_path,
                                                        self.model_save_directory, self.settings, self.file_paths)
                self.training_handler.start()
        else:
            self.ui.plainTextEdit_display.appendPlainText("training resume" + "\n")
            self.training_handler.resume()

    def stop(self):
        self.ui.plainTextEdit_display.appendPlainText("training stop" + "\n")
        self.training_handler.stop()

    def pause(self):
        self.ui.plainTextEdit_display.appendPlainText("training pause" + "\n")
        self.training_handler.pause()

    def returnToMainWindow(self):
        self.returnMainWindowSignal.emit()


class ExperimentFileHandler(FileSystemEventHandler):
    """
    PCR预测实验实时监控文件夹下的文件操作，如果对文件进行了修改回调进入on_modified函数
    执行PCR模型训练和预测并显示相关结果到控制界面上
    """

    def __init__(self, *args):
        super().__init__()

        self.signals = args[0]
        self.current_path = args[1]
        self.model_save_directory = args[2]
        self.settings = args[3]
        self.dirMonitored = args[4]

    # 文件修改回调函数：执行PCR曲线预测模型的训练和递推预测
    def on_modified(self, event):
        print(self.dirMonitored + "/Experiment.csv")
        y = pd.read_csv(self.dirMonitored + "/Experiment.csv")
        current_pcr_cycle_number = y.shape[1]
        if current_pcr_cycle_number < int(self.settings.initial_prediction_cycle_number):
            self.signals.addShowText.emit(str("current pcr cycle number is {}, "
                                              "did not reach the start prediction cycle number"
                                              .format(current_pcr_cycle_number)))
            return
        if current_pcr_cycle_number == int(self.settings.full_cycle_number):
            self.signals.addShowText.emit(str("current pcr cycle number is {}, "
                                              "reached the full cycle number"
                                              .format(current_pcr_cycle_number)))
            return
        dir_path, full_file_name = os.path.split(self.dirMonitored)
        file_name, extension = os.path.splitext(full_file_name)
        interpolation = int(self.settings.interpolation_number)
        # 前19个周期训练， 预测后21个周期
        totalinterpolation = current_pcr_cycle_number * interpolation
        T = int(self.settings.prediction_timestep)
        N = (int(self.settings.full_cycle_number) - current_pcr_cycle_number) * interpolation
        trainratio = float(self.settings.trainTestSplit_ratio)

        y = np.array(y.iloc[0, 0:current_pcr_cycle_number]).flatten()
        # 数据插值
        x = np.linspace(1, y.shape[0], y.shape[0])
        xnew = np.linspace(1, y.shape[0], totalinterpolation)  # 在1~y[0]一种生成190个点
        # @todo 目前可以正常发送信号，记录
        self.signals.updataGraphSig.emit(x, y, 0, "PCR amplification curve at {} cycle in file {}"
                                         .format(current_pcr_cycle_number, self.dirMonitored))
        # 选择插值方式
        f = interpolate.interp1d(x, y, kind=self.settings.interpolation_method)
        ynew = f(xnew)
        df = pd.DataFrame(ynew)

        # 归一化
        scaler = MinMaxScaler(feature_range=(0, 1))
        df = scaler.fit_transform(np.array(df).reshape(-1, 1))

        # 训练集和测试集划分
        training_size = int(len(df) * trainratio)
        train_data, test_data = df[0:training_size, :], df[training_size:len(df), :]

        # reshape into X=t,t+1,t+2,t+3 and Y=t+4
        time_step = T
        X_train, y_train = create_dataset(train_data, time_step)
        X_test, y_test = create_dataset(test_data, time_step)

        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        model = Sequential()

        # model.add(SimpleRNN(500, return_sequences=True))
        # model.add(SimpleRNN(500))

        model.add(LSTM(500, return_sequences=True, input_shape=(T, 1)))
        model.add(Bidirectional(LSTM(500, return_sequences=True), merge_mode='concat'))
        model.add(LSTM(500))

        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')

        history = LossHistory()
        model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=50, verbose=1,
                  callbacks=[history])
        model_save_path = self.model_save_directory + "/{}_at_{}_cycle_number.h5". \
            format(file_name, current_pcr_cycle_number)
        model.save(model_save_path)

        # # 训练损失值分析
        # Loss = pd.DataFrame(history.saveLoss)
        # Loss.to_csv('saveLoss.csv', index=False)
        # valLoss = pd.DataFrame(history.savevalLoss)
        # valLoss.to_csv('savevalLoss.csv', index=False)
        # score = model.evaluate(X_test, y_test, verbose=0)
        # history.loss_plot('epoch')

        # 训练值和测试值预测分析
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)
        test_predict1 = pd.DataFrame(test_predict)
        train_predict = scaler.inverse_transform(train_predict)
        test_predict = scaler.inverse_transform(test_predict)

        # 误差计算
        training_loss = math.sqrt(mean_squared_error(y_train, train_predict))
        test_loss = math.sqrt(mean_squared_error(y_test, test_predict))
        self.signals.addShowText.emit(
            str("training loss is {}, test loss is {}".format(training_loss, test_loss)))
        look_back = T
        trainPredictPlot = numpy.empty_like(df)
        trainPredictPlot[:, :] = np.nan
        # 从第八个数据开始，用8个x值数据预测y值数据，将所有x值预测的y值数据填入trainPredictPlot中
        trainPredictPlot[look_back:len(train_predict) + look_back, :] = train_predict  # 将训练数据预测值填入空数组中用于绘图
        # shift test predictions for plotting
        testPredictPlot = numpy.empty_like(df)
        testPredictPlot[:, :] = numpy.nan
        # 从测试数据+8个后的数据开始，用8个x值数据预测y值数据，将所有测试数据x值预测的y值数据填入testPredictPlot中
        testPredictPlot[len(train_predict) + (look_back * 2) + 1:len(df) - 1, :] = test_predict
        # plot baseline and predictions
        # plt.plot(scaler.inverse_transform(df),'blue')
        # plt.plot(trainPredictPlot,'red')
        # plt.plot(testPredictPlot,'green')
        # plt.show()

        # 预测#########################################################################################
        # 用测试数据第9个开始的8个数据作为起始值开始向后预测，一共要预测和之前插值一样多的预测点
        x_input = test_data[len(test_data) - T:].reshape(1, -1)
        temp_input = list(x_input)
        temp_input = temp_input[0].tolist()

        lst_output = []
        n_steps = T
        i = 0
        while i < N:
            if len(temp_input) > T:
                x_input = np.array(temp_input[1:])
                x_input = x_input.reshape(1, -1)
                x_input = x_input.reshape((1, n_steps, 1))
                yhat = model.predict(x_input, verbose=0)
                temp_input.extend(yhat[0].tolist())
                temp_input = temp_input[1:]
                lst_output.extend(yhat.tolist())
                i = i + 1
            else:
                x_input = x_input.reshape((1, n_steps, 1))
                yhat = model.predict(x_input, verbose=0)
                temp_input.extend(yhat[0].tolist())
                lst_output.extend(yhat.tolist())
                i = i + 1

        output = scaler.inverse_transform(lst_output)
        print(output.shape)
        x = np.arange(0, output.shape[0], 1)
        self.signals.updataGraphSig.emit(x, output, 0, "PCR amplification curve prediction result"
                                                       " at {} cycle in file {}".format(current_pcr_cycle_number,
                                                                                        self.dirMonitored))

        # @todo
        # inputdata = pd.read_csv('quadratic.csv')
        # alldata = pd.concat([inputdata, output], axis=0, ignore_index=False)  # 提取特征与标签拼接
        # alldata.to_csv('alldata.csv', index=True)
        # tranisientdata = pd.read_csv('alldata.csv')
        # endpoint_value = tranisientdata.iloc[399]
        # abc.append(endpoint_value)
        #
        # abcdata = pd.DataFrame(abc)
        # abcdata.to_csv('abcdata.csv', index=True)
        # # losstrain=pd.read_csv('saveLoss.csv')
        # # lossval=pd.read_csv('savevalLoss.csv')
        # # allloss=pd.concat([losstrain,lossval],axis=1,ignore_index=False)
        # # allloss.to_csv('allloss.csv',index=True)


# @todo 抽象一个界面基类，复用一部分代码，其他代码基本一致可以复制(类继承不可以在不重写其他方法的基础上强制重写构造而不调用父类构造)
class MainWindowExperiment(QMainWindow):
    """
    实验主界面类
    """
    returnMainWindowSignal = Signal()

    def __init__(self, settings):
        super().__init__()
        self.current_path = QDir.currentPath()
        self.Ui_MainWindow_File_Path = self.current_path + "/UI/PCR_Experiment.ui"
        Ui_MainWindow_File = QFile(self.Ui_MainWindow_File_Path)
        Ui_MainWindow_File.open(QFile.ReadOnly)
        Ui_MainWindow_File.close()
        self.ui = QUiLoader().load(Ui_MainWindow_File)
        self.ui.setParent(self)
        self.file_type = ('.csv', '.xlsx')
        self.settings = settings
        self.ui.plainTextEdit_display.appendPlainText("current directory path is " + str(self.current_path) + "\n")
        self.dirMonitored = None
        self.file_paths = []
        self.files_number = 0

        self.initial_prediction_cycle_number = int(self.settings.initial_prediction_cycle_number)
        self.interpolation_method = self.settings.interpolation_method
        self.traintestsplit_ratio = float(self.settings.trainTestSplit_ratio)
        self.prediction_timestep = int(self.settings.prediction_timestep)
        self.interpolation_number = int(self.settings.interpolation_number)

        self.initUI()
        # 初始化 gv_visual_data 的显示
        self.gv_visual_data_content = MyFigureCanvas(width=self.ui.graphicsView_plotGraph.width() / 101,
                                                     height=self.ui.graphicsView_plotGraph.height() / 101,
                                                     xlim=(0, 2 * np.pi),
                                                     ylim=(-1, 1))
        self.initGraphPloting()

        self.signals = CommunicationSignals()
        self.model_save_directory = self.current_path + "/Model_Save"

        self.training_handler = ExperimentHandler(self.signals, self.current_path,
                                                  self.model_save_directory, self.settings, self.dirMonitored)
        self.connections()

    def chooseMonitoredDir(self):
        self.dirMonitored = QFileDialog.getExistingDirectory(self, 'open the directory', self.current_path)
        self.training_handler = ExperimentHandler(self.signals, self.current_path,
                                                  self.model_save_directory, self.settings, self.dirMonitored)
        self.ui.plainTextEdit_display.appendPlainText("Mnoitor directory is  : " + str(self.dirMonitored) + "\n")

    def initUI(self):
        self.ui.spinBox_initialPredictionCycleNumber.setRange(0, int(self.settings.full_cycle_number))
        for method in self.settings.allInterpolationMethod:
            self.ui.comboBox_interpolationMethod.addItem(method)
        self.ui.doubleSpinBox_traintestsplitRatio.setSingleStep(0.01)
        self.ui.doubleSpinBox_traintestsplitRatio.setRange(self.settings.traintestsplitRatio_Range[0],
                                                           self.settings.traintestsplitRatio_Range[1])
        self.ui.spinBox_predictionTimestep.setRange(self.settings.predictionTimestep_Range[0],
                                                    self.settings.predictionTimestep_Range[1])
        self.ui.spinBox_interpolationNumber.setRange(self.settings.interpolationNumber_Range[0],
                                                     self.settings.interpolationNumber_Range[1])

        self.ui.spinBox_initialPredictionCycleNumber.setValue(self.initial_prediction_cycle_number)
        self.ui.comboBox_interpolationMethod.setCurrentText(self.interpolation_method)
        self.ui.doubleSpinBox_traintestsplitRatio.setValue(self.traintestsplit_ratio)
        self.ui.spinBox_predictionTimestep.setValue(self.prediction_timestep)
        self.ui.spinBox_interpolationNumber.setValue(self.interpolation_number)

    def connections(self):
        self.ui.pushButton_start.clicked.connect(self.run)
        self.ui.pushButton_stop.clicked.connect(self.stop)
        self.ui.pushButton_pause.clicked.connect(self.pause)

        self.ui.spinBox_initialPredictionCycleNumber.valueChanged.connect(self.modifySettings)
        self.ui.comboBox_interpolationMethod.currentTextChanged.connect(self.modifySettings)
        self.ui.doubleSpinBox_traintestsplitRatio.valueChanged.connect(self.modifySettings)
        self.ui.spinBox_predictionTimestep.valueChanged.connect(self.modifySettings)
        self.ui.spinBox_interpolationNumber.valueChanged.connect(self.modifySettings)

        self.training_handler.event_handler.signals.updataGraphSig.connect(self.plot_graph)
        self.training_handler.event_handler.signals.addShowText.connect(self.addShowText)
        self.ui.pushButton_back.clicked.connect(self.returnToMainWindow)
        self.ui.pushButton_clearDisplay.clicked.connect(lambda arg=self: self.ui.plainTextEdit_display.setPlainText(""))
        self.ui.pushButton_ChooseFileMonitored.clicked.connect(self.chooseMonitoredDir)

    def modifySettings(self):
        self.initial_prediction_cycle_number = self.ui.spinBox_initialPredictionCycleNumber.value()
        self.interpolation_method = self.ui.comboBox_interpolationMethod.currentText()
        self.traintestsplit_ratio = self.ui.doubleSpinBox_traintestsplitRatio.value()
        self.prediction_timestep = self.ui.spinBox_predictionTimestep.value()
        self.interpolation_number = self.ui.spinBox_interpolationNumber.value()

        self.settings.initial_prediction_cycle_number = self.ui.spinBox_initialPredictionCycleNumber.value()
        self.settings.interpolation_method = self.ui.comboBox_interpolationMethod.currentText()
        self.settings.traintestsplit_ratio = self.ui.doubleSpinBox_traintestsplitRatio.value()
        self.settings.prediction_timestep = self.ui.spinBox_predictionTimestep.value()
        self.settings.interpolation_number = self.ui.spinBox_interpolationNumber.value()

    def initGraphPloting(self):
        # 加载的图形（FigureCanvas）不能直接放到graphicview控件中，必须先放到graphicScene，然后再把graphicscene放到graphicview中
        self.graphic_scene = QGraphicsScene()  # 创建一个QGraphicsScene
        self.graphic_scene.addWidget(
            self.gv_visual_data_content)  # 把图形放到QGraphicsScene中，注意：图形是作为一个QWidget放到放到QGraphicsScene中的
        self.ui.graphicsView_plotGraph.setScene(self.graphic_scene)  # 把QGraphicsScene放入QGraphicsView

    @Slot(object, object, int, str)
    def plot_graph(self, x, y, Ct=0, graph_title='PCR amplification curve'):
        """


        :return: None
        """
        self.ui.plainTextEdit_display.appendPlainText(str(graph_title) + "\n")
        self.gv_visual_data_content.axes.clear()  # 由于图片需要反复绘制，所以每次绘制前清空，然后绘图
        self.gv_visual_data_content.axes.plot(x, y)
        self.gv_visual_data_content.axes.set_title(graph_title)
        self.gv_visual_data_content.draw()  # 刷新画布显示图片，否则不刷新显示

    @Slot(str)
    def addShowText(self, showText):
        self.ui.plainTextEdit_display.appendPlainText(showText + "\n")

    """
    开启多线程执行训练程序
    """

    def run(self):
        if not os.path.exists(self.dirMonitored + "/Experiment.csv"):
            self.ui.plainTextEdit_display.appendPlainText(
                str("file Monitored is empty, please select file") + "\n")
            return
        self.ui.plainTextEdit_display.appendPlainText(
            str("Start Monitor file at path : {}").format(self.dirMonitored) + "\n")
        self.training_handler.start()

    def stop(self):
        self.ui.plainTextEdit_display.appendPlainText("monitor stop" + "\n")
        self.training_handler.stop()

    def pause(self):
        self.ui.plainTextEdit_display.appendPlainText("monitor pause" + "\n")
        self.training_handler.pause()

    def returnToMainWindow(self):
        self.returnMainWindowSignal.emit()


# @todo
class MainWindow(QMainWindow):
    """
    主控功能选择界面类，可选择进入训练主界面和实验主界面
    """

    def __init__(self):
        super().__init__()
        self.settings = Settings()
        self.current_path = os.getcwd()
        self.Ui_MainWindow_File_Path = self.current_path + "/UI/PCR_MainWindow.ui"
        Ui_MainWindow_File = QFile(self.Ui_MainWindow_File_Path)
        Ui_MainWindow_File.open(QFile.ReadOnly)
        Ui_MainWindow_File.close()
        self.ui = QUiLoader().load(Ui_MainWindow_File)
        self.ui.setParent(self)
        # self.training_handler = TrainingHandler()
        self.connections()
        self.resize(800, 500)
        self.show()

    def connections(self):
        self.ui.pushButton_experiment.clicked.connect(self.chooseExperimentMode)
        self.ui.pushButton_training.clicked.connect(self.chooseTrainingMode)
        self.ui.actionPropertyConfiguration.triggered.connect(self.setSettings)
        # @todo

    def chooseExperimentMode(self):
        self.UI_MainWindowExperiment = MainWindowExperiment(self.settings)
        self.UI_MainWindowExperiment.returnMainWindowSignal.connect(self.experimentUIReturn)
        self.UI_MainWindowExperiment.ui.actionPropertyConfiguration.triggered.connect(self.setSettings)
        self.hide()
        self.UI_MainWindowExperiment.resize(1250, 800)
        self.UI_MainWindowExperiment.show()

    def chooseTrainingMode(self):
        self.UI_MainWindowTraining = MainWindowTraining(self.settings)
        self.UI_MainWindowTraining.returnMainWindowSignal.connect(self.trainingUIReturn)
        self.UI_MainWindowTraining.ui.actionPropertyConfiguration.triggered.connect(self.setSettings)
        self.hide()
        self.UI_MainWindowTraining.resize(1300, 900)
        self.UI_MainWindowTraining.show()

    def setSettings(self):
        self.UI_MainWindowSettings = MainWidgetSettings()
        flags = self.UI_MainWindowSettings.windowFlags()
        self.UI_MainWindowSettings.setWindowFlags(flags | Qt.WindowStaysOnTopHint | Qt.Window)
        self.UI_MainWindowSettings.setWindowModality(Qt.ApplicationModal)
        self.UI_MainWindowSettings.setWindowFlags(flags)
        self.UI_MainWindowSettings.show()

    @Slot()
    def experimentUIReturn(self):
        self.UI_MainWindowExperiment.hide()
        self.show()

    @Slot()
    def trainingUIReturn(self):
        self.UI_MainWindowTraining.hide()
        self.show()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    sys.exit(app.exec_())
