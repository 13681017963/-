from django.shortcuts import render
# from Decisiontree import models
# from Decisiontree.models import
from django.http import HttpResponse
import json
from django.forms.models import model_to_dict
import datetime
import dateutil
import requests
from django.conf import settings
import os
import sys
import time
import numpy as np
import src.config
from src.config import algorithm
from src.config import diag_service
from src.config import station_config
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeRegressor  #调用回归树模型
from sklearn.ensemble import AdaBoostRegressor
from sklearn import ensemble
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.multioutput import MultiOutputRegressor
import dclimate.d_std_lib as d_std_lib
from mttkinter import mtTkinter as tk
import pywt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa import arima_model
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import r2_score
from django.conf import settings
from TimeSeries import models
globalIp = '192.168.1.123:8998'
import warnings
warnings.filterwarnings("ignore")


def get_data(var, year, method, month, station):
    test_year = year - 2010
    # 读气温矩平数据作为标签
    # 按月读取
    if var == 0 and method == 0 and station == 0:
        train = np.array(pd.read_csv('TimeSeries/static/beijing_train_tmean_1981_2010.csv')).reshape(-1, 1).squeeze()
        test = np.array(pd.read_csv('TimeSeries/static/beijing_test_tmean_2011_2018.csv')).reshape(-1, 1)[:test_year*12, :].squeeze()
    if var == 0 and method == 0 and station == 1:
        train = []
        test = []
        for i in range(20):
            train.append(np.array(pd.read_csv('TimeSeries/static/beijing_train_zd_tmean_1981_2010.csv'))[:, i:i+1].squeeze())
            test.append(np.array(pd.read_csv('TimeSeries/static/beijing_test_zd_tmean_2011_2018.csv'))[:, i:i+1].squeeze())
    if var == 1 and method == 0 and station == 0:
        train = np.array(pd.read_csv('TimeSeries/static/beijing_train_pr_1981_2010.csv')).reshape(-1, 1).squeeze()
        test = np.array(pd.read_csv('TimeSeries/static/beijing_test_pr_2011_2018.csv')).reshape(-1, 1)[:test_year*12, :].squeeze()
    if var == 1 and method == 0 and station == 1:
        train = []
        test = []
        for i in range(20):
            train.append(np.array(pd.read_csv('TimeSeries/static/beijing_train_zd_pr_1981_2010.csv'))[:, i:i+1].squeeze())
            test.append(np.array(pd.read_csv('TimeSeries/static/beijing_test_zd_pr_2011_2018.csv'))[:, i:i+1].squeeze())
    if var == 0 and method == 1 and station == 0:
        train = np.array(pd.read_csv('TimeSeries/static/beijing_train_tmean_1981_2010.csv')).reshape(-1, 1)[month-1::12, :].squeeze()
        temp = np.array(pd.read_csv('TimeSeries/static/beijing_test_tmean_2011_2018.csv')).reshape(-1, 1)[month-1:test_year*12:12, :]
        if len(temp) == 1:
            test = temp
        else:
            test = temp.squeeze()
    if var == 0 and method == 1 and station == 1:
        train = []
        test = []
        for i in range(20):
            train.append(np.array(pd.read_csv('TimeSeries/static/beijing_train_zd_tmean_1981_2010.csv'))[month-1::12, i:i+1].squeeze())
            temp = np.array(pd.read_csv('TimeSeries/static/beijing_test_zd_tmean_2011_2018.csv'))[month-1::12, i:i+1]
            if len(temp) == 1:
                test.append(temp)
            else:
                test.append(temp.squeeze())
    if var == 1 and method == 1 and station == 0:
        train = np.array(pd.read_csv('TimeSeries/static/beijing_train_pr_1981_2010.csv'))[month-1::12, :].squeeze()
        temp = np.array(pd.read_csv('TimeSeries/static/beijing_test_pr_2011_2018.csv'))[month-1:test_year*12:12, :]
        if len(temp) == 1:
            test = temp
        else:
            test = temp.squeeze()
    if var == 1 and method == 1 and station == 1:
        train = []
        test = []
        for i in range(20):
            train.append(np.array(pd.read_csv('TimeSeries/static/beijing_train_zd_pr_1981_2010.csv'))[month-1::12, i:i+1].squeeze())
            temp = np.array(pd.read_csv('TimeSeries/static/beijing_test_zd_pr_2011_2018.csv'))[month-1::12, i:i+1]
            if len(temp) == 1:
                test.append(temp)
            else:
                test.append(temp.squeeze())

    return train, test


def strToint4(a, b, c, d):
    a = int(a)
    b = int(b)
    c = int(c)
    d = int(d)
    return a, b, c, d


def is_illegal(var, year, method, month, predict_year):
    if not (var.isdigit()) or not (year.isdigit()) or not (method.isdigit()) or not (month.isdigit()) \
            or int(year) > int(predict_year) or int(year) < 2011 or int(month) < 1 or int(month) > 12:
        return True
    return False


def get_parameter(request):
    var = request.GET.get("var")
    year = request.GET.get("year")
    method = request.GET.get("method")
    month = request.GET.get("month")
    return var, year, method, month


#获取图形标题
def get_title(area, var, num, month):
    # title = station_config.get_area_name(area) + diag_config.get_diag_name(
    #     var) + "EOF第"+str(num+1)+"空间模态"
    # if num == 0:
    #     title = "二元决策树"
    # if num == 1:
    #     title = "随机森林"
    # if num == 2:
    #     title = "梯度提升树"
    # if num == 3:
    #     title = "自适应增强树"
    # title = "2018年" + str(month) + "月矩平预测图"
    title = ""
    # print(title)
    return title


#读取res站点数据开始绘图
def draw_map(var, sta_order_list, res, area, num, month, info):
    for i in range(num-1):
        result = res[i]
        Region_ID = area
        res_list = algorithm.make_station_list(sta_order_list, result)
        # OutPicFile1 = "d:\\eof_" + var + "_" + area +"_"+str(i+1)+ ".png"  # 图片输出路径
        OutPicFile1 = "static/" + "TimeSeries_" + info + "_" + str(i + 1) + ".png"
        # root_path = "DecisionTree"
        # print("!!!", root_path)
        LevelFile = 'src\\config\\LEV\\eof\\maplev_descisiontree.LEV'
        Region_Dict2 = algorithm.get_RegionID_by_XML('src\\config\\sky_region_config_utf8_2.xml', Region_ID)
        title = get_title(area, var, i, month)
        d_std_lib.DrawMapMain_XML_CFG(Region_Dict2, res_list, Levfile=LevelFile, \
                                      Title=title, imgfile=OutPicFile1,
                                      bShowImage=False, bDrawNumber=False, bDrawColorbar=True,
                                      format1='%1d')  # ,Title='')
        print('ok')


def mean_filter(kernel_size, data):
    if kernel_size%2==0 or kernel_size<=1:
        print('kernel_size滤波核的需为大于1的奇数')
        return
    else:
        padding_data = []
        mid = kernel_size//2
        for i in range(mid):
            padding_data.append(0)
        padding_data.extend(data.tolist())
        for i in range(mid):
            padding_data.append(0)
    result = []
    for i in range(0, len(padding_data)-2*mid, 1):
        temp = 0
        for j in range(kernel_size):
            temp += padding_data[i+j]
        temp = temp / kernel_size
        result.append(temp)
    return result


# 一维信号小波分解去噪重构
def Wavelet_Transform(data):
    w = pywt.Wavelet('db1')  # 选用Daubechies1小波
    maxlev = pywt.dwt_max_level(len(data), w.dec_len)
    coeffs = pywt.wavedec(data, 'db1', level=maxlev)  # 将信号进行小波分解
    threshold = 0.5
    for i in range(1, len(coeffs)):
        coeffs[i] = pywt.threshold(coeffs[i], threshold * max(coeffs[i]))  # 将噪声滤波  小于value的值设置为0，大于value的减去value作为新的值
    result = pywt.waverec(coeffs, 'db1')  # 将信号进行小波重构
    result = pd.DataFrame(result)
    result.fillna(method='ffill', inplace=True)  # 前值填充
    result = np.array(result)
    return result


def ARIMA(series, n, var, method, year, algorithm, index):
    """
    只讨论一阶差分的ARIMA模型，预测，数字索引从1开始
    series:时间序列
    n:需要往后预测的个数
    """
    series = np.array(series)
    series = pd.Series(series.reshape(-1))
    currentDir = os.getcwd()  # 当前工作路径
    # 一阶差分数据
    fd = series.diff(1)[1:]
    # plot_acf(fd).savefig(currentDir + '/一阶差分自相关图.png')
    # plot_pacf(fd).savefig(currentDir + '/一阶差分偏自相关图.png')
    # 一阶差分单位根检验
    unitP = adfuller(fd)[1]
    if unitP > 0.05:
        unitAssess = '单位根检验中p值为%.2f，大于0.05，该一阶差分序列可能为非平稳序列' % (unitP)
        print('单位根检验中p值为%.2f，大于0.05，认为该一阶差分序列判断为非平稳序列'%(unitP))
    else:
        unitAssess = '单位根检验中p值为%.2f，小于0.05，认为该一阶差分序列为平稳序列' % (unitP)
        print('单位根检验中p值为%.2f，小于0.05，认为该一阶差分序列判断为平稳序列'%(unitP))
    # 白噪声检验
    noiseP = acorr_ljungbox(fd, lags=1)[-1]
    if noiseP <= 0.05:
        noiseAssess = '白噪声检验中p值为%.2f，小于0.05，认为该一阶差分序列为非白噪声' % noiseP
        print('白噪声检验中p值为%.2f，小于0.05，认为该一阶差分序列为非白噪声'%noiseP)
    else:
        noiseAssess = '白噪声检验中p值%.2f，大于0.05，该一阶差分序列可能为白噪声' % noiseP
        print('白噪声检验中%.2f，大于0.05，认为该一阶差分序列为白噪声'%noiseP)
    # BIC准则确定p、q值
    pMax = int(series.shape[0] / 10)  # 一般阶数不超过length/10
    qMax = pMax  # 一般阶数不超过length/10

    pMax = qMax = 3

    # if method == 1:
    #     bics = list()
    #     for p in range(pMax + 1):
    #         tmp = list()
    #         for q in range(qMax + 1):
    #             try:
    #                 tmp.append(arima_model.ARIMA(series, (p, 1, q)).fit().bic)
    #             except Exception as e:
    #                 # print(str(e))
    #                 tmp.append(1e+10)  # 加入一个很大的数
    #         bics.append(tmp)
    #     bics = pd.DataFrame(bics)
    #     p, q = bics.stack().idxmin()
    # else:
    #     if var == 0:
    #         if algorithm == 0:
    #             if year == 2011:
    #                 if index == -1:
    #                     p, q = 3, 3
    #                 else:
    #                     if index == 0:
    #                         p, q = 3, 3
    #                     if index == 1:
    #                         p, q = 3, 3
    #                     if index == 2:
    #                         p, q = 3, 3
    #                     if index == 3:
    #                         p, q = 3, 3
    #                     if index == 4:
    #                         p, q = 3, 3
    #                     if index == 5:
    #                         p, q = 3, 3
    #                     if index == 6:
    #                         p, q = 3, 3
    #                     if index == 7:
    #                         p, q = 3, 3
    #                     if index == 8:
    #                         p, q = 3, 3
    #                     if index == 9:
    #                         p, q = 3, 3
    #                     if index == 10:
    #                         p, q = 3, 3
    #                     if index == 11:
    #                         p, q = 3, 3
    #                     if index == 12:
    #                         p, q = 3, 3
    #                     if index == 13:
    #                         p, q = 3, 3
    #                     if index == 14:
    #                         p, q = 3, 3
    #                     if index == 15:
    #                         p, q = 3, 3
    #                     if index == 16:
    #                         p, q = 3, 3
    #                     if index == 17:
    #                         p, q = 3, 3
    #                     if index == 18:
    #                         p, q = 3, 3
    #                     if index == 19:
    #                         p, q = 3, 3
    #             if year == 2012:
    #                 if index == -1:
    #                     p, q = 3, 3
    #                 else:
    #                     if index == 0:
    #                         p, q = 3, 3
    #                     if index == 1:
    #                         p, q = 3, 3
    #                     if index == 2:
    #                         p, q = 3, 3
    #                     if index == 3:
    #                         p, q = 3, 3
    #                     if index == 4:
    #                         p, q = 3, 3
    #                     if index == 5:
    #                         p, q = 3, 3
    #                     if index == 6:
    #                         p, q = 3, 3
    #                     if index == 7:
    #                         p, q = 3, 3
    #                     if index == 8:
    #                         p, q = 3, 3
    #                     if index == 9:
    #                         p, q = 3, 3
    #                     if index == 10:
    #                         p, q = 3, 3
    #                     if index == 11:
    #                         p, q = 3, 3
    #                     if index == 12:
    #                         p, q = 3, 3
    #                     if index == 13:
    #                         p, q = 3, 3
    #                     if index == 14:
    #                         p, q = 3, 3
    #                     if index == 15:
    #                         p, q = 3, 3
    #                     if index == 16:
    #                         p, q = 3, 3
    #                     if index == 17:
    #                         p, q = 3, 3
    #                     if index == 18:
    #                         p, q = 3, 3
    #                     if index == 19:
    #                         p, q = 3, 3
    #             if year == 2013:
    #                 if index == -1:
    #                     p, q = 3, 3
    #                 else:
    #                     if index == 0:
    #                         p, q = 3, 3
    #                     if index == 1:
    #                         p, q = 3, 3
    #                     if index == 2:
    #                         p, q = 3, 3
    #                     if index == 3:
    #                         p, q = 3, 3
    #                     if index == 4:
    #                         p, q = 3, 3
    #                     if index == 5:
    #                         p, q = 3, 3
    #                     if index == 6:
    #                         p, q = 3, 3
    #                     if index == 7:
    #                         p, q = 3, 3
    #                     if index == 8:
    #                         p, q = 3, 3
    #                     if index == 9:
    #                         p, q = 3, 3
    #                     if index == 10:
    #                         p, q = 3, 3
    #                     if index == 11:
    #                         p, q = 3, 3
    #                     if index == 12:
    #                         p, q = 3, 3
    #                     if index == 13:
    #                         p, q = 3, 3
    #                     if index == 14:
    #                         p, q = 3, 3
    #                     if index == 15:
    #                         p, q = 3, 3
    #                     if index == 16:
    #                         p, q = 3, 3
    #                     if index == 17:
    #                         p, q = 3, 3
    #                     if index == 18:
    #                         p, q = 3, 3
    #                     if index == 19:
    #                         p, q = 3, 3
    #             if year == 2014:
    #                 if index == -1:
    #                     p, q = 3, 3
    #                 else:
    #                     if index == 0:
    #                         p, q = 3, 3
    #                     if index == 1:
    #                         p, q = 3, 3
    #                     if index == 2:
    #                         p, q = 3, 3
    #                     if index == 3:
    #                         p, q = 3, 3
    #                     if index == 4:
    #                         p, q = 3, 3
    #                     if index == 5:
    #                         p, q = 3, 3
    #                     if index == 6:
    #                         p, q = 3, 3
    #                     if index == 7:
    #                         p, q = 3, 3
    #                     if index == 8:
    #                         p, q = 3, 3
    #                     if index == 9:
    #                         p, q = 3, 3
    #                     if index == 10:
    #                         p, q = 3, 3
    #                     if index == 11:
    #                         p, q = 3, 3
    #                     if index == 12:
    #                         p, q = 3, 3
    #                     if index == 13:
    #                         p, q = 3, 3
    #                     if index == 14:
    #                         p, q = 3, 3
    #                     if index == 15:
    #                         p, q = 3, 3
    #                     if index == 16:
    #                         p, q = 3, 3
    #                     if index == 17:
    #                         p, q = 3, 3
    #                     if index == 18:
    #                         p, q = 3, 3
    #                     if index == 19:
    #                         p, q = 3, 3
    #             if year == 2015:
    #                 if index == -1:
    #                     p, q = 3, 3
    #                 else:
    #                     if index == 0:
    #                         p, q = 3, 3
    #                     if index == 1:
    #                         p, q = 3, 3
    #                     if index == 2:
    #                         p, q = 3, 3
    #                     if index == 3:
    #                         p, q = 3, 3
    #                     if index == 4:
    #                         p, q = 3, 3
    #                     if index == 5:
    #                         p, q = 3, 3
    #                     if index == 6:
    #                         p, q = 3, 3
    #                     if index == 7:
    #                         p, q = 3, 3
    #                     if index == 8:
    #                         p, q = 3, 3
    #                     if index == 9:
    #                         p, q = 3, 3
    #                     if index == 10:
    #                         p, q = 3, 3
    #                     if index == 11:
    #                         p, q = 3, 3
    #                     if index == 12:
    #                         p, q = 3, 3
    #                     if index == 13:
    #                         p, q = 3, 3
    #                     if index == 14:
    #                         p, q = 3, 3
    #                     if index == 15:
    #                         p, q = 3, 3
    #                     if index == 16:
    #                         p, q = 3, 3
    #                     if index == 17:
    #                         p, q = 3, 3
    #                     if index == 18:
    #                         p, q = 3, 3
    #                     if index == 19:
    #                         p, q = 3, 3
    #             if year == 2016:
    #                 if index == -1:
    #                     p, q = 3, 3
    #                 else:
    #                     if index == 0:
    #                         p, q = 3, 3
    #                     if index == 1:
    #                         p, q = 3, 3
    #                     if index == 2:
    #                         p, q = 3, 3
    #                     if index == 3:
    #                         p, q = 3, 3
    #                     if index == 4:
    #                         p, q = 3, 3
    #                     if index == 5:
    #                         p, q = 3, 3
    #                     if index == 6:
    #                         p, q = 3, 3
    #                     if index == 7:
    #                         p, q = 3, 3
    #                     if index == 8:
    #                         p, q = 3, 3
    #                     if index == 9:
    #                         p, q = 3, 3
    #                     if index == 10:
    #                         p, q = 3, 3
    #                     if index == 11:
    #                         p, q = 3, 3
    #                     if index == 12:
    #                         p, q = 3, 3
    #                     if index == 13:
    #                         p, q = 3, 3
    #                     if index == 14:
    #                         p, q = 3, 3
    #                     if index == 15:
    #                         p, q = 3, 3
    #                     if index == 16:
    #                         p, q = 3, 3
    #                     if index == 17:
    #                         p, q = 3, 3
    #                     if index == 18:
    #                         p, q = 3, 3
    #                     if index == 19:
    #                         p, q = 3, 3
    #             if year == 2017:
    #                 if index == -1:
    #                     p, q = 3, 3
    #                 else:
    #                     if index == 0:
    #                         p, q = 3, 3
    #                     if index == 1:
    #                         p, q = 3, 3
    #                     if index == 2:
    #                         p, q = 3, 3
    #                     if index == 3:
    #                         p, q = 3, 3
    #                     if index == 4:
    #                         p, q = 3, 3
    #                     if index == 5:
    #                         p, q = 3, 3
    #                     if index == 6:
    #                         p, q = 3, 3
    #                     if index == 7:
    #                         p, q = 3, 3
    #                     if index == 8:
    #                         p, q = 3, 3
    #                     if index == 9:
    #                         p, q = 3, 3
    #                     if index == 10:
    #                         p, q = 3, 3
    #                     if index == 11:
    #                         p, q = 3, 3
    #                     if index == 12:
    #                         p, q = 3, 3
    #                     if index == 13:
    #                         p, q = 3, 3
    #                     if index == 14:
    #                         p, q = 3, 3
    #                     if index == 15:
    #                         p, q = 3, 3
    #                     if index == 16:
    #                         p, q = 3, 3
    #                     if index == 17:
    #                         p, q = 3, 3
    #                     if index == 18:
    #                         p, q = 3, 3
    #                     if index == 19:
    #                         p, q = 3, 3
    #             if year == 2018:
    #                 if index == -1:
    #                     p, q = 8, 8
    #                 else:
    #                     if index == 0:
    #                         p, q = 8, 8
    #                     if index == 1:
    #                         p, q = 8, 8
    #                     if index == 2:
    #                         p, q = 8, 8
    #                     if index == 3:
    #                         p, q = 8, 8
    #                     if index == 4:
    #                         p, q = 8, 8
    #                     if index == 5:
    #                         p, q = 8, 8
    #                     if index == 6:
    #                         p, q = 8, 8
    #                     if index == 7:
    #                         p, q = 8, 8
    #                     if index == 8:
    #                         p, q = 8, 8
    #                     if index == 9:
    #                         p, q = 8, 8
    #                     if index == 10:
    #                         p, q = 8, 8
    #                     if index == 11:
    #                         p, q = 8, 8
    #                     if index == 12:
    #                         p, q = 8, 8
    #                     if index == 13:
    #                         p, q = 8, 8
    #                     if index == 14:
    #                         p, q = 8, 8
    #                     if index == 15:
    #                         p, q = 8, 8
    #                     if index == 16:
    #                         p, q = 8, 8
    #                     if index == 17:
    #                         p, q = 8, 8
    #                     if index == 18:
    #                         p, q = 8, 8
    #                     if index == 19:
    #                         p, q = 8, 8
    #         elif algorithm == 1:
    #             if year == 2011:
    #                 if index == -1:
    #                     p, q = 3, 3
    #                 else:
    #                     if index == 0:
    #                         p, q = 3, 3
    #                     if index == 1:
    #                         p, q = 3, 3
    #                     if index == 2:
    #                         p, q = 3, 3
    #                     if index == 3:
    #                         p, q = 3, 3
    #                     if index == 4:
    #                         p, q = 3, 3
    #                     if index == 5:
    #                         p, q = 3, 3
    #                     if index == 6:
    #                         p, q = 3, 3
    #                     if index == 7:
    #                         p, q = 3, 3
    #                     if index == 8:
    #                         p, q = 3, 3
    #                     if index == 9:
    #                         p, q = 3, 3
    #                     if index == 10:
    #                         p, q = 3, 3
    #                     if index == 11:
    #                         p, q = 3, 3
    #                     if index == 12:
    #                         p, q = 3, 3
    #                     if index == 13:
    #                         p, q = 3, 3
    #                     if index == 14:
    #                         p, q = 3, 3
    #                     if index == 15:
    #                         p, q = 3, 3
    #                     if index == 16:
    #                         p, q = 3, 3
    #                     if index == 17:
    #                         p, q = 3, 3
    #                     if index == 18:
    #                         p, q = 3, 3
    #                     if index == 19:
    #                         p, q = 3, 3
    #             if year == 2012:
    #                 if index == -1:
    #                     p, q = 3, 3
    #                 else:
    #                     if index == 0:
    #                         p, q = 3, 3
    #                     if index == 1:
    #                         p, q = 3, 3
    #                     if index == 2:
    #                         p, q = 3, 3
    #                     if index == 3:
    #                         p, q = 3, 3
    #                     if index == 4:
    #                         p, q = 3, 3
    #                     if index == 5:
    #                         p, q = 3, 3
    #                     if index == 6:
    #                         p, q = 3, 3
    #                     if index == 7:
    #                         p, q = 3, 3
    #                     if index == 8:
    #                         p, q = 3, 3
    #                     if index == 9:
    #                         p, q = 3, 3
    #                     if index == 10:
    #                         p, q = 3, 3
    #                     if index == 11:
    #                         p, q = 3, 3
    #                     if index == 12:
    #                         p, q = 3, 3
    #                     if index == 13:
    #                         p, q = 3, 3
    #                     if index == 14:
    #                         p, q = 3, 3
    #                     if index == 15:
    #                         p, q = 3, 3
    #                     if index == 16:
    #                         p, q = 3, 3
    #                     if index == 17:
    #                         p, q = 3, 3
    #                     if index == 18:
    #                         p, q = 3, 3
    #                     if index == 19:
    #                         p, q = 3, 3
    #             if year == 2013:
    #                 if index == -1:
    #                     p, q = 3, 3
    #                 else:
    #                     if index == 0:
    #                         p, q = 3, 3
    #                     if index == 1:
    #                         p, q = 3, 3
    #                     if index == 2:
    #                         p, q = 3, 3
    #                     if index == 3:
    #                         p, q = 3, 3
    #                     if index == 4:
    #                         p, q = 3, 3
    #                     if index == 5:
    #                         p, q = 3, 3
    #                     if index == 6:
    #                         p, q = 3, 3
    #                     if index == 7:
    #                         p, q = 3, 3
    #                     if index == 8:
    #                         p, q = 3, 3
    #                     if index == 9:
    #                         p, q = 3, 3
    #                     if index == 10:
    #                         p, q = 3, 3
    #                     if index == 11:
    #                         p, q = 3, 3
    #                     if index == 12:
    #                         p, q = 3, 3
    #                     if index == 13:
    #                         p, q = 3, 3
    #                     if index == 14:
    #                         p, q = 3, 3
    #                     if index == 15:
    #                         p, q = 3, 3
    #                     if index == 16:
    #                         p, q = 3, 3
    #                     if index == 17:
    #                         p, q = 3, 3
    #                     if index == 18:
    #                         p, q = 3, 3
    #                     if index == 19:
    #                         p, q = 3, 3
    #             if year == 2014:
    #                 if index == -1:
    #                     p, q = 3, 3
    #                 else:
    #                     if index == 0:
    #                         p, q = 3, 3
    #                     if index == 1:
    #                         p, q = 3, 3
    #                     if index == 2:
    #                         p, q = 3, 3
    #                     if index == 3:
    #                         p, q = 3, 3
    #                     if index == 4:
    #                         p, q = 3, 3
    #                     if index == 5:
    #                         p, q = 3, 3
    #                     if index == 6:
    #                         p, q = 3, 3
    #                     if index == 7:
    #                         p, q = 3, 3
    #                     if index == 8:
    #                         p, q = 3, 3
    #                     if index == 9:
    #                         p, q = 3, 3
    #                     if index == 10:
    #                         p, q = 3, 3
    #                     if index == 11:
    #                         p, q = 3, 3
    #                     if index == 12:
    #                         p, q = 3, 3
    #                     if index == 13:
    #                         p, q = 3, 3
    #                     if index == 14:
    #                         p, q = 3, 3
    #                     if index == 15:
    #                         p, q = 3, 3
    #                     if index == 16:
    #                         p, q = 3, 3
    #                     if index == 17:
    #                         p, q = 3, 3
    #                     if index == 18:
    #                         p, q = 3, 3
    #                     if index == 19:
    #                         p, q = 3, 3
    #             if year == 2015:
    #                 if index == -1:
    #                     p, q = 3, 3
    #                 else:
    #                     if index == 0:
    #                         p, q = 3, 3
    #                     if index == 1:
    #                         p, q = 3, 3
    #                     if index == 2:
    #                         p, q = 3, 3
    #                     if index == 3:
    #                         p, q = 3, 3
    #                     if index == 4:
    #                         p, q = 3, 3
    #                     if index == 5:
    #                         p, q = 3, 3
    #                     if index == 6:
    #                         p, q = 3, 3
    #                     if index == 7:
    #                         p, q = 3, 3
    #                     if index == 8:
    #                         p, q = 3, 3
    #                     if index == 9:
    #                         p, q = 3, 3
    #                     if index == 10:
    #                         p, q = 3, 3
    #                     if index == 11:
    #                         p, q = 3, 3
    #                     if index == 12:
    #                         p, q = 3, 3
    #                     if index == 13:
    #                         p, q = 3, 3
    #                     if index == 14:
    #                         p, q = 3, 3
    #                     if index == 15:
    #                         p, q = 3, 3
    #                     if index == 16:
    #                         p, q = 3, 3
    #                     if index == 17:
    #                         p, q = 3, 3
    #                     if index == 18:
    #                         p, q = 3, 3
    #                     if index == 19:
    #                         p, q = 3, 3
    #             if year == 2016:
    #                 if index == -1:
    #                     p, q = 3, 3
    #                 else:
    #                     if index == 0:
    #                         p, q = 3, 3
    #                     if index == 1:
    #                         p, q = 3, 3
    #                     if index == 2:
    #                         p, q = 3, 3
    #                     if index == 3:
    #                         p, q = 3, 3
    #                     if index == 4:
    #                         p, q = 3, 3
    #                     if index == 5:
    #                         p, q = 3, 3
    #                     if index == 6:
    #                         p, q = 3, 3
    #                     if index == 7:
    #                         p, q = 3, 3
    #                     if index == 8:
    #                         p, q = 3, 3
    #                     if index == 9:
    #                         p, q = 3, 3
    #                     if index == 10:
    #                         p, q = 3, 3
    #                     if index == 11:
    #                         p, q = 3, 3
    #                     if index == 12:
    #                         p, q = 3, 3
    #                     if index == 13:
    #                         p, q = 3, 3
    #                     if index == 14:
    #                         p, q = 3, 3
    #                     if index == 15:
    #                         p, q = 3, 3
    #                     if index == 16:
    #                         p, q = 3, 3
    #                     if index == 17:
    #                         p, q = 3, 3
    #                     if index == 18:
    #                         p, q = 3, 3
    #                     if index == 19:
    #                         p, q = 3, 3
    #             if year == 2017:
    #                 if index == -1:
    #                     p, q = 3, 3
    #                 else:
    #                     if index == 0:
    #                         p, q = 3, 3
    #                     if index == 1:
    #                         p, q = 3, 3
    #                     if index == 2:
    #                         p, q = 3, 3
    #                     if index == 3:
    #                         p, q = 3, 3
    #                     if index == 4:
    #                         p, q = 3, 3
    #                     if index == 5:
    #                         p, q = 3, 3
    #                     if index == 6:
    #                         p, q = 3, 3
    #                     if index == 7:
    #                         p, q = 3, 3
    #                     if index == 8:
    #                         p, q = 3, 3
    #                     if index == 9:
    #                         p, q = 3, 3
    #                     if index == 10:
    #                         p, q = 3, 3
    #                     if index == 11:
    #                         p, q = 3, 3
    #                     if index == 12:
    #                         p, q = 3, 3
    #                     if index == 13:
    #                         p, q = 3, 3
    #                     if index == 14:
    #                         p, q = 3, 3
    #                     if index == 15:
    #                         p, q = 3, 3
    #                     if index == 16:
    #                         p, q = 3, 3
    #                     if index == 17:
    #                         p, q = 3, 3
    #                     if index == 18:
    #                         p, q = 3, 3
    #                     if index == 19:
    #                         p, q = 3, 3
    #             if year == 2018:
    #                 if index == -1:
    #                     p, q = 8, 8
    #                 else:
    #                     if index == 0:
    #                         p, q = 8, 8
    #                     if index == 1:
    #                         p, q = 8, 8
    #                     if index == 2:
    #                         p, q = 8, 8
    #                     if index == 3:
    #                         p, q = 8, 8
    #                     if index == 4:
    #                         p, q = 8, 8
    #                     if index == 5:
    #                         p, q = 8, 8
    #                     if index == 6:
    #                         p, q = 8, 8
    #                     if index == 7:
    #                         p, q = 8, 8
    #                     if index == 8:
    #                         p, q = 8, 8
    #                     if index == 9:
    #                         p, q = 8, 8
    #                     if index == 10:
    #                         p, q = 8, 8
    #                     if index == 11:
    #                         p, q = 8, 8
    #                     if index == 12:
    #                         p, q = 8, 8
    #                     if index == 13:
    #                         p, q = 8, 8
    #                     if index == 14:
    #                         p, q = 8, 8
    #                     if index == 15:
    #                         p, q = 8, 8
    #                     if index == 16:
    #                         p, q = 8, 8
    #                     if index == 17:
    #                         p, q = 8, 8
    #                     if index == 18:
    #                         p, q = 8, 8
    #                     if index == 19:
    #                         p, q = 8, 8
    #         else:
    #             if year == 2011:
    #                 if index == -1:
    #                     p, q = 3, 3
    #                 else:
    #                     if index == 0:
    #                         p, q = 3, 3
    #                     if index == 1:
    #                         p, q = 3, 3
    #                     if index == 2:
    #                         p, q = 3, 3
    #                     if index == 3:
    #                         p, q = 3, 3
    #                     if index == 4:
    #                         p, q = 3, 3
    #                     if index == 5:
    #                         p, q = 3, 3
    #                     if index == 6:
    #                         p, q = 3, 3
    #                     if index == 7:
    #                         p, q = 3, 3
    #                     if index == 8:
    #                         p, q = 3, 3
    #                     if index == 9:
    #                         p, q = 3, 3
    #                     if index == 10:
    #                         p, q = 3, 3
    #                     if index == 11:
    #                         p, q = 3, 3
    #                     if index == 12:
    #                         p, q = 3, 3
    #                     if index == 13:
    #                         p, q = 3, 3
    #                     if index == 14:
    #                         p, q = 3, 3
    #                     if index == 15:
    #                         p, q = 3, 3
    #                     if index == 16:
    #                         p, q = 3, 3
    #                     if index == 17:
    #                         p, q = 3, 3
    #                     if index == 18:
    #                         p, q = 3, 3
    #                     if index == 19:
    #                         p, q = 3, 3
    #             if year == 2012:
    #                 if index == -1:
    #                     p, q = 3, 3
    #                 else:
    #                     if index == 0:
    #                         p, q = 3, 3
    #                     if index == 1:
    #                         p, q = 3, 3
    #                     if index == 2:
    #                         p, q = 3, 3
    #                     if index == 3:
    #                         p, q = 3, 3
    #                     if index == 4:
    #                         p, q = 3, 3
    #                     if index == 5:
    #                         p, q = 3, 3
    #                     if index == 6:
    #                         p, q = 3, 3
    #                     if index == 7:
    #                         p, q = 3, 3
    #                     if index == 8:
    #                         p, q = 3, 3
    #                     if index == 9:
    #                         p, q = 3, 3
    #                     if index == 10:
    #                         p, q = 3, 3
    #                     if index == 11:
    #                         p, q = 3, 3
    #                     if index == 12:
    #                         p, q = 3, 3
    #                     if index == 13:
    #                         p, q = 3, 3
    #                     if index == 14:
    #                         p, q = 3, 3
    #                     if index == 15:
    #                         p, q = 3, 3
    #                     if index == 16:
    #                         p, q = 3, 3
    #                     if index == 17:
    #                         p, q = 3, 3
    #                     if index == 18:
    #                         p, q = 3, 3
    #                     if index == 19:
    #                         p, q = 3, 3
    #             if year == 2013:
    #                 if index == -1:
    #                     p, q = 3, 3
    #                 else:
    #                     if index == 0:
    #                         p, q = 3, 3
    #                     if index == 1:
    #                         p, q = 3, 3
    #                     if index == 2:
    #                         p, q = 3, 3
    #                     if index == 3:
    #                         p, q = 3, 3
    #                     if index == 4:
    #                         p, q = 3, 3
    #                     if index == 5:
    #                         p, q = 3, 3
    #                     if index == 6:
    #                         p, q = 3, 3
    #                     if index == 7:
    #                         p, q = 3, 3
    #                     if index == 8:
    #                         p, q = 3, 3
    #                     if index == 9:
    #                         p, q = 3, 3
    #                     if index == 10:
    #                         p, q = 3, 3
    #                     if index == 11:
    #                         p, q = 3, 3
    #                     if index == 12:
    #                         p, q = 3, 3
    #                     if index == 13:
    #                         p, q = 3, 3
    #                     if index == 14:
    #                         p, q = 3, 3
    #                     if index == 15:
    #                         p, q = 3, 3
    #                     if index == 16:
    #                         p, q = 3, 3
    #                     if index == 17:
    #                         p, q = 3, 3
    #                     if index == 18:
    #                         p, q = 3, 3
    #                     if index == 19:
    #                         p, q = 3, 3
    #             if year == 2014:
    #                 if index == -1:
    #                     p, q = 3, 3
    #                 else:
    #                     if index == 0:
    #                         p, q = 3, 3
    #                     if index == 1:
    #                         p, q = 3, 3
    #                     if index == 2:
    #                         p, q = 3, 3
    #                     if index == 3:
    #                         p, q = 3, 3
    #                     if index == 4:
    #                         p, q = 3, 3
    #                     if index == 5:
    #                         p, q = 3, 3
    #                     if index == 6:
    #                         p, q = 3, 3
    #                     if index == 7:
    #                         p, q = 3, 3
    #                     if index == 8:
    #                         p, q = 3, 3
    #                     if index == 9:
    #                         p, q = 3, 3
    #                     if index == 10:
    #                         p, q = 3, 3
    #                     if index == 11:
    #                         p, q = 3, 3
    #                     if index == 12:
    #                         p, q = 3, 3
    #                     if index == 13:
    #                         p, q = 3, 3
    #                     if index == 14:
    #                         p, q = 3, 3
    #                     if index == 15:
    #                         p, q = 3, 3
    #                     if index == 16:
    #                         p, q = 3, 3
    #                     if index == 17:
    #                         p, q = 3, 3
    #                     if index == 18:
    #                         p, q = 3, 3
    #                     if index == 19:
    #                         p, q = 3, 3
    #             if year == 2015:
    #                 if index == -1:
    #                     p, q = 3, 3
    #                 else:
    #                     if index == 0:
    #                         p, q = 3, 3
    #                     if index == 1:
    #                         p, q = 3, 3
    #                     if index == 2:
    #                         p, q = 3, 3
    #                     if index == 3:
    #                         p, q = 3, 3
    #                     if index == 4:
    #                         p, q = 3, 3
    #                     if index == 5:
    #                         p, q = 3, 3
    #                     if index == 6:
    #                         p, q = 3, 3
    #                     if index == 7:
    #                         p, q = 3, 3
    #                     if index == 8:
    #                         p, q = 3, 3
    #                     if index == 9:
    #                         p, q = 3, 3
    #                     if index == 10:
    #                         p, q = 3, 3
    #                     if index == 11:
    #                         p, q = 3, 3
    #                     if index == 12:
    #                         p, q = 3, 3
    #                     if index == 13:
    #                         p, q = 3, 3
    #                     if index == 14:
    #                         p, q = 3, 3
    #                     if index == 15:
    #                         p, q = 3, 3
    #                     if index == 16:
    #                         p, q = 3, 3
    #                     if index == 17:
    #                         p, q = 3, 3
    #                     if index == 18:
    #                         p, q = 3, 3
    #                     if index == 19:
    #                         p, q = 3, 3
    #             if year == 2016:
    #                 if index == -1:
    #                     p, q = 3, 3
    #                 else:
    #                     if index == 0:
    #                         p, q = 3, 3
    #                     if index == 1:
    #                         p, q = 3, 3
    #                     if index == 2:
    #                         p, q = 3, 3
    #                     if index == 3:
    #                         p, q = 3, 3
    #                     if index == 4:
    #                         p, q = 3, 3
    #                     if index == 5:
    #                         p, q = 3, 3
    #                     if index == 6:
    #                         p, q = 3, 3
    #                     if index == 7:
    #                         p, q = 3, 3
    #                     if index == 8:
    #                         p, q = 3, 3
    #                     if index == 9:
    #                         p, q = 3, 3
    #                     if index == 10:
    #                         p, q = 3, 3
    #                     if index == 11:
    #                         p, q = 3, 3
    #                     if index == 12:
    #                         p, q = 3, 3
    #                     if index == 13:
    #                         p, q = 3, 3
    #                     if index == 14:
    #                         p, q = 3, 3
    #                     if index == 15:
    #                         p, q = 3, 3
    #                     if index == 16:
    #                         p, q = 3, 3
    #                     if index == 17:
    #                         p, q = 3, 3
    #                     if index == 18:
    #                         p, q = 3, 3
    #                     if index == 19:
    #                         p, q = 3, 3
    #             if year == 2017:
    #                 if index == -1:
    #                     p, q = 3, 3
    #                 else:
    #                     if index == 0:
    #                         p, q = 3, 3
    #                     if index == 1:
    #                         p, q = 3, 3
    #                     if index == 2:
    #                         p, q = 3, 3
    #                     if index == 3:
    #                         p, q = 3, 3
    #                     if index == 4:
    #                         p, q = 3, 3
    #                     if index == 5:
    #                         p, q = 3, 3
    #                     if index == 6:
    #                         p, q = 3, 3
    #                     if index == 7:
    #                         p, q = 3, 3
    #                     if index == 8:
    #                         p, q = 3, 3
    #                     if index == 9:
    #                         p, q = 3, 3
    #                     if index == 10:
    #                         p, q = 3, 3
    #                     if index == 11:
    #                         p, q = 3, 3
    #                     if index == 12:
    #                         p, q = 3, 3
    #                     if index == 13:
    #                         p, q = 3, 3
    #                     if index == 14:
    #                         p, q = 3, 3
    #                     if index == 15:
    #                         p, q = 3, 3
    #                     if index == 16:
    #                         p, q = 3, 3
    #                     if index == 17:
    #                         p, q = 3, 3
    #                     if index == 18:
    #                         p, q = 3, 3
    #                     if index == 19:
    #                         p, q = 3, 3
    #             if year == 2018:
    #                 if index == -1:
    #                     p, q = 8, 8
    #                 else:
    #                     if index == 0:
    #                         p, q = 8, 8
    #                     if index == 1:
    #                         p, q = 8, 8
    #                     if index == 2:
    #                         p, q = 8, 8
    #                     if index == 3:
    #                         p, q = 8, 8
    #                     if index == 4:
    #                         p, q = 8, 8
    #                     if index == 5:
    #                         p, q = 8, 8
    #                     if index == 6:
    #                         p, q = 8, 8
    #                     if index == 7:
    #                         p, q = 8, 8
    #                     if index == 8:
    #                         p, q = 8, 8
    #                     if index == 9:
    #                         p, q = 8, 8
    #                     if index == 10:
    #                         p, q = 8, 8
    #                     if index == 11:
    #                         p, q = 8, 8
    #                     if index == 12:
    #                         p, q = 8, 8
    #                     if index == 13:
    #                         p, q = 8, 8
    #                     if index == 14:
    #                         p, q = 8, 8
    #                     if index == 15:
    #                         p, q = 8, 8
    #                     if index == 16:
    #                         p, q = 8, 8
    #                     if index == 17:
    #                         p, q = 8, 8
    #                     if index == 18:
    #                         p, q = 8, 8
    #                     if index == 19:
    #                         p, q = 8, 8
    #     else:
    #         if algorithm == 0:
    #             if year == 2011:
    #                 if index == -1:
    #                     p, q = 3, 3
    #                 else:
    #                     if index == 0:
    #                         p, q = 3, 3
    #                     if index == 1:
    #                         p, q = 3, 3
    #                     if index == 2:
    #                         p, q = 3, 3
    #                     if index == 3:
    #                         p, q = 3, 3
    #                     if index == 4:
    #                         p, q = 3, 3
    #                     if index == 5:
    #                         p, q = 3, 3
    #                     if index == 6:
    #                         p, q = 3, 3
    #                     if index == 7:
    #                         p, q = 3, 3
    #                     if index == 8:
    #                         p, q = 3, 3
    #                     if index == 9:
    #                         p, q = 3, 3
    #                     if index == 10:
    #                         p, q = 3, 3
    #                     if index == 11:
    #                         p, q = 3, 3
    #                     if index == 12:
    #                         p, q = 3, 3
    #                     if index == 13:
    #                         p, q = 3, 3
    #                     if index == 14:
    #                         p, q = 3, 3
    #                     if index == 15:
    #                         p, q = 3, 3
    #                     if index == 16:
    #                         p, q = 3, 3
    #                     if index == 17:
    #                         p, q = 3, 3
    #                     if index == 18:
    #                         p, q = 3, 3
    #                     if index == 19:
    #                         p, q = 3, 3
    #             if year == 2012:
    #                 if index == -1:
    #                     p, q = 3, 3
    #                 else:
    #                     if index == 0:
    #                         p, q = 3, 3
    #                     if index == 1:
    #                         p, q = 3, 3
    #                     if index == 2:
    #                         p, q = 3, 3
    #                     if index == 3:
    #                         p, q = 3, 3
    #                     if index == 4:
    #                         p, q = 3, 3
    #                     if index == 5:
    #                         p, q = 3, 3
    #                     if index == 6:
    #                         p, q = 3, 3
    #                     if index == 7:
    #                         p, q = 3, 3
    #                     if index == 8:
    #                         p, q = 3, 3
    #                     if index == 9:
    #                         p, q = 3, 3
    #                     if index == 10:
    #                         p, q = 3, 3
    #                     if index == 11:
    #                         p, q = 3, 3
    #                     if index == 12:
    #                         p, q = 3, 3
    #                     if index == 13:
    #                         p, q = 3, 3
    #                     if index == 14:
    #                         p, q = 3, 3
    #                     if index == 15:
    #                         p, q = 3, 3
    #                     if index == 16:
    #                         p, q = 3, 3
    #                     if index == 17:
    #                         p, q = 3, 3
    #                     if index == 18:
    #                         p, q = 3, 3
    #                     if index == 19:
    #                         p, q = 3, 3
    #             if year == 2013:
    #                 if index == -1:
    #                     p, q = 3, 3
    #                 else:
    #                     if index == 0:
    #                         p, q = 3, 3
    #                     if index == 1:
    #                         p, q = 3, 3
    #                     if index == 2:
    #                         p, q = 3, 3
    #                     if index == 3:
    #                         p, q = 3, 3
    #                     if index == 4:
    #                         p, q = 3, 3
    #                     if index == 5:
    #                         p, q = 3, 3
    #                     if index == 6:
    #                         p, q = 3, 3
    #                     if index == 7:
    #                         p, q = 3, 3
    #                     if index == 8:
    #                         p, q = 3, 3
    #                     if index == 9:
    #                         p, q = 3, 3
    #                     if index == 10:
    #                         p, q = 3, 3
    #                     if index == 11:
    #                         p, q = 3, 3
    #                     if index == 12:
    #                         p, q = 3, 3
    #                     if index == 13:
    #                         p, q = 3, 3
    #                     if index == 14:
    #                         p, q = 3, 3
    #                     if index == 15:
    #                         p, q = 3, 3
    #                     if index == 16:
    #                         p, q = 3, 3
    #                     if index == 17:
    #                         p, q = 3, 3
    #                     if index == 18:
    #                         p, q = 3, 3
    #                     if index == 19:
    #                         p, q = 3, 3
    #             if year == 2014:
    #                 if index == -1:
    #                     p, q = 3, 3
    #                 else:
    #                     if index == 0:
    #                         p, q = 3, 3
    #                     if index == 1:
    #                         p, q = 3, 3
    #                     if index == 2:
    #                         p, q = 3, 3
    #                     if index == 3:
    #                         p, q = 3, 3
    #                     if index == 4:
    #                         p, q = 3, 3
    #                     if index == 5:
    #                         p, q = 3, 3
    #                     if index == 6:
    #                         p, q = 3, 3
    #                     if index == 7:
    #                         p, q = 3, 3
    #                     if index == 8:
    #                         p, q = 3, 3
    #                     if index == 9:
    #                         p, q = 3, 3
    #                     if index == 10:
    #                         p, q = 3, 3
    #                     if index == 11:
    #                         p, q = 3, 3
    #                     if index == 12:
    #                         p, q = 3, 3
    #                     if index == 13:
    #                         p, q = 3, 3
    #                     if index == 14:
    #                         p, q = 3, 3
    #                     if index == 15:
    #                         p, q = 3, 3
    #                     if index == 16:
    #                         p, q = 3, 3
    #                     if index == 17:
    #                         p, q = 3, 3
    #                     if index == 18:
    #                         p, q = 3, 3
    #                     if index == 19:
    #                         p, q = 3, 3
    #             if year == 2015:
    #                 if index == -1:
    #                     p, q = 3, 3
    #                 else:
    #                     if index == 0:
    #                         p, q = 3, 3
    #                     if index == 1:
    #                         p, q = 3, 3
    #                     if index == 2:
    #                         p, q = 3, 3
    #                     if index == 3:
    #                         p, q = 3, 3
    #                     if index == 4:
    #                         p, q = 3, 3
    #                     if index == 5:
    #                         p, q = 3, 3
    #                     if index == 6:
    #                         p, q = 3, 3
    #                     if index == 7:
    #                         p, q = 3, 3
    #                     if index == 8:
    #                         p, q = 3, 3
    #                     if index == 9:
    #                         p, q = 3, 3
    #                     if index == 10:
    #                         p, q = 3, 3
    #                     if index == 11:
    #                         p, q = 3, 3
    #                     if index == 12:
    #                         p, q = 3, 3
    #                     if index == 13:
    #                         p, q = 3, 3
    #                     if index == 14:
    #                         p, q = 3, 3
    #                     if index == 15:
    #                         p, q = 3, 3
    #                     if index == 16:
    #                         p, q = 3, 3
    #                     if index == 17:
    #                         p, q = 3, 3
    #                     if index == 18:
    #                         p, q = 3, 3
    #                     if index == 19:
    #                         p, q = 3, 3
    #             if year == 2016:
    #                 if index == -1:
    #                     p, q = 3, 3
    #                 else:
    #                     if index == 0:
    #                         p, q = 3, 3
    #                     if index == 1:
    #                         p, q = 3, 3
    #                     if index == 2:
    #                         p, q = 3, 3
    #                     if index == 3:
    #                         p, q = 3, 3
    #                     if index == 4:
    #                         p, q = 3, 3
    #                     if index == 5:
    #                         p, q = 3, 3
    #                     if index == 6:
    #                         p, q = 3, 3
    #                     if index == 7:
    #                         p, q = 3, 3
    #                     if index == 8:
    #                         p, q = 3, 3
    #                     if index == 9:
    #                         p, q = 3, 3
    #                     if index == 10:
    #                         p, q = 3, 3
    #                     if index == 11:
    #                         p, q = 3, 3
    #                     if index == 12:
    #                         p, q = 3, 3
    #                     if index == 13:
    #                         p, q = 3, 3
    #                     if index == 14:
    #                         p, q = 3, 3
    #                     if index == 15:
    #                         p, q = 3, 3
    #                     if index == 16:
    #                         p, q = 3, 3
    #                     if index == 17:
    #                         p, q = 3, 3
    #                     if index == 18:
    #                         p, q = 3, 3
    #                     if index == 19:
    #                         p, q = 3, 3
    #             if year == 2017:
    #                 if index == -1:
    #                     p, q = 3, 3
    #                 else:
    #                     if index == 0:
    #                         p, q = 3, 3
    #                     if index == 1:
    #                         p, q = 3, 3
    #                     if index == 2:
    #                         p, q = 3, 3
    #                     if index == 3:
    #                         p, q = 3, 3
    #                     if index == 4:
    #                         p, q = 3, 3
    #                     if index == 5:
    #                         p, q = 3, 3
    #                     if index == 6:
    #                         p, q = 3, 3
    #                     if index == 7:
    #                         p, q = 3, 3
    #                     if index == 8:
    #                         p, q = 3, 3
    #                     if index == 9:
    #                         p, q = 3, 3
    #                     if index == 10:
    #                         p, q = 3, 3
    #                     if index == 11:
    #                         p, q = 3, 3
    #                     if index == 12:
    #                         p, q = 3, 3
    #                     if index == 13:
    #                         p, q = 3, 3
    #                     if index == 14:
    #                         p, q = 3, 3
    #                     if index == 15:
    #                         p, q = 3, 3
    #                     if index == 16:
    #                         p, q = 3, 3
    #                     if index == 17:
    #                         p, q = 3, 3
    #                     if index == 18:
    #                         p, q = 3, 3
    #                     if index == 19:
    #                         p, q = 3, 3
    #             if year == 2018:
    #                 if index == -1:
    #                     p, q = 3, 3
    #                 else:
    #                     if index == 0:
    #                         p, q = 3, 3
    #                     if index == 1:
    #                         p, q = 3, 3
    #                     if index == 2:
    #                         p, q = 3, 3
    #                     if index == 3:
    #                         p, q = 3, 3
    #                     if index == 4:
    #                         p, q = 3, 3
    #                     if index == 5:
    #                         p, q = 3, 3
    #                     if index == 6:
    #                         p, q = 3, 3
    #                     if index == 7:
    #                         p, q = 3, 3
    #                     if index == 8:
    #                         p, q = 3, 3
    #                     if index == 9:
    #                         p, q = 3, 3
    #                     if index == 10:
    #                         p, q = 3, 3
    #                     if index == 11:
    #                         p, q = 3, 3
    #                     if index == 12:
    #                         p, q = 3, 3
    #                     if index == 13:
    #                         p, q = 3, 3
    #                     if index == 14:
    #                         p, q = 3, 3
    #                     if index == 15:
    #                         p, q = 3, 3
    #                     if index == 16:
    #                         p, q = 3, 3
    #                     if index == 17:
    #                         p, q = 3, 3
    #                     if index == 18:
    #                         p, q = 3, 3
    #                     if index == 19:
    #                         p, q = 3, 3
    #         elif algorithm == 1:
    #             if year == 2011:
    #                 if index == -1:
    #                     p, q = 3, 3
    #                 else:
    #                     if index == 0:
    #                         p, q = 3, 3
    #                     if index == 1:
    #                         p, q = 3, 3
    #                     if index == 2:
    #                         p, q = 3, 3
    #                     if index == 3:
    #                         p, q = 3, 3
    #                     if index == 4:
    #                         p, q = 3, 3
    #                     if index == 5:
    #                         p, q = 3, 3
    #                     if index == 6:
    #                         p, q = 3, 3
    #                     if index == 7:
    #                         p, q = 3, 3
    #                     if index == 8:
    #                         p, q = 3, 3
    #                     if index == 9:
    #                         p, q = 3, 3
    #                     if index == 10:
    #                         p, q = 3, 3
    #                     if index == 11:
    #                         p, q = 3, 3
    #                     if index == 12:
    #                         p, q = 3, 3
    #                     if index == 13:
    #                         p, q = 3, 3
    #                     if index == 14:
    #                         p, q = 3, 3
    #                     if index == 15:
    #                         p, q = 3, 3
    #                     if index == 16:
    #                         p, q = 3, 3
    #                     if index == 17:
    #                         p, q = 3, 3
    #                     if index == 18:
    #                         p, q = 3, 3
    #                     if index == 19:
    #                         p, q = 3, 3
    #             if year == 2012:
    #                 if index == -1:
    #                     p, q = 3, 3
    #                 else:
    #                     if index == 0:
    #                         p, q = 3, 3
    #                     if index == 1:
    #                         p, q = 3, 3
    #                     if index == 2:
    #                         p, q = 3, 3
    #                     if index == 3:
    #                         p, q = 3, 3
    #                     if index == 4:
    #                         p, q = 3, 3
    #                     if index == 5:
    #                         p, q = 3, 3
    #                     if index == 6:
    #                         p, q = 3, 3
    #                     if index == 7:
    #                         p, q = 3, 3
    #                     if index == 8:
    #                         p, q = 3, 3
    #                     if index == 9:
    #                         p, q = 3, 3
    #                     if index == 10:
    #                         p, q = 3, 3
    #                     if index == 11:
    #                         p, q = 3, 3
    #                     if index == 12:
    #                         p, q = 3, 3
    #                     if index == 13:
    #                         p, q = 3, 3
    #                     if index == 14:
    #                         p, q = 3, 3
    #                     if index == 15:
    #                         p, q = 3, 3
    #                     if index == 16:
    #                         p, q = 3, 3
    #                     if index == 17:
    #                         p, q = 3, 3
    #                     if index == 18:
    #                         p, q = 3, 3
    #                     if index == 19:
    #                         p, q = 3, 3
    #             if year == 2013:
    #                 if index == -1:
    #                     p, q = 3, 3
    #                 else:
    #                     if index == 0:
    #                         p, q = 3, 3
    #                     if index == 1:
    #                         p, q = 3, 3
    #                     if index == 2:
    #                         p, q = 3, 3
    #                     if index == 3:
    #                         p, q = 3, 3
    #                     if index == 4:
    #                         p, q = 3, 3
    #                     if index == 5:
    #                         p, q = 3, 3
    #                     if index == 6:
    #                         p, q = 3, 3
    #                     if index == 7:
    #                         p, q = 3, 3
    #                     if index == 8:
    #                         p, q = 3, 3
    #                     if index == 9:
    #                         p, q = 3, 3
    #                     if index == 10:
    #                         p, q = 3, 3
    #                     if index == 11:
    #                         p, q = 3, 3
    #                     if index == 12:
    #                         p, q = 3, 3
    #                     if index == 13:
    #                         p, q = 3, 3
    #                     if index == 14:
    #                         p, q = 3, 3
    #                     if index == 15:
    #                         p, q = 3, 3
    #                     if index == 16:
    #                         p, q = 3, 3
    #                     if index == 17:
    #                         p, q = 3, 3
    #                     if index == 18:
    #                         p, q = 3, 3
    #                     if index == 19:
    #                         p, q = 3, 3
    #             if year == 2014:
    #                 if index == -1:
    #                     p, q = 3, 3
    #                 else:
    #                     if index == 0:
    #                         p, q = 3, 3
    #                     if index == 1:
    #                         p, q = 3, 3
    #                     if index == 2:
    #                         p, q = 3, 3
    #                     if index == 3:
    #                         p, q = 3, 3
    #                     if index == 4:
    #                         p, q = 3, 3
    #                     if index == 5:
    #                         p, q = 3, 3
    #                     if index == 6:
    #                         p, q = 3, 3
    #                     if index == 7:
    #                         p, q = 3, 3
    #                     if index == 8:
    #                         p, q = 3, 3
    #                     if index == 9:
    #                         p, q = 3, 3
    #                     if index == 10:
    #                         p, q = 3, 3
    #                     if index == 11:
    #                         p, q = 3, 3
    #                     if index == 12:
    #                         p, q = 3, 3
    #                     if index == 13:
    #                         p, q = 3, 3
    #                     if index == 14:
    #                         p, q = 3, 3
    #                     if index == 15:
    #                         p, q = 3, 3
    #                     if index == 16:
    #                         p, q = 3, 3
    #                     if index == 17:
    #                         p, q = 3, 3
    #                     if index == 18:
    #                         p, q = 3, 3
    #                     if index == 19:
    #                         p, q = 3, 3
    #             if year == 2015:
    #                 if index == -1:
    #                     p, q = 3, 3
    #                 else:
    #                     if index == 0:
    #                         p, q = 3, 3
    #                     if index == 1:
    #                         p, q = 3, 3
    #                     if index == 2:
    #                         p, q = 3, 3
    #                     if index == 3:
    #                         p, q = 3, 3
    #                     if index == 4:
    #                         p, q = 3, 3
    #                     if index == 5:
    #                         p, q = 3, 3
    #                     if index == 6:
    #                         p, q = 3, 3
    #                     if index == 7:
    #                         p, q = 3, 3
    #                     if index == 8:
    #                         p, q = 3, 3
    #                     if index == 9:
    #                         p, q = 3, 3
    #                     if index == 10:
    #                         p, q = 3, 3
    #                     if index == 11:
    #                         p, q = 3, 3
    #                     if index == 12:
    #                         p, q = 3, 3
    #                     if index == 13:
    #                         p, q = 3, 3
    #                     if index == 14:
    #                         p, q = 3, 3
    #                     if index == 15:
    #                         p, q = 3, 3
    #                     if index == 16:
    #                         p, q = 3, 3
    #                     if index == 17:
    #                         p, q = 3, 3
    #                     if index == 18:
    #                         p, q = 3, 3
    #                     if index == 19:
    #                         p, q = 3, 3
    #             if year == 2016:
    #                 if index == -1:
    #                     p, q = 3, 3
    #                 else:
    #                     if index == 0:
    #                         p, q = 3, 3
    #                     if index == 1:
    #                         p, q = 3, 3
    #                     if index == 2:
    #                         p, q = 3, 3
    #                     if index == 3:
    #                         p, q = 3, 3
    #                     if index == 4:
    #                         p, q = 3, 3
    #                     if index == 5:
    #                         p, q = 3, 3
    #                     if index == 6:
    #                         p, q = 3, 3
    #                     if index == 7:
    #                         p, q = 3, 3
    #                     if index == 8:
    #                         p, q = 3, 3
    #                     if index == 9:
    #                         p, q = 3, 3
    #                     if index == 10:
    #                         p, q = 3, 3
    #                     if index == 11:
    #                         p, q = 3, 3
    #                     if index == 12:
    #                         p, q = 3, 3
    #                     if index == 13:
    #                         p, q = 3, 3
    #                     if index == 14:
    #                         p, q = 3, 3
    #                     if index == 15:
    #                         p, q = 3, 3
    #                     if index == 16:
    #                         p, q = 3, 3
    #                     if index == 17:
    #                         p, q = 3, 3
    #                     if index == 18:
    #                         p, q = 3, 3
    #                     if index == 19:
    #                         p, q = 3, 3
    #             if year == 2017:
    #                 if index == -1:
    #                     p, q = 3, 3
    #                 else:
    #                     if index == 0:
    #                         p, q = 3, 3
    #                     if index == 1:
    #                         p, q = 3, 3
    #                     if index == 2:
    #                         p, q = 3, 3
    #                     if index == 3:
    #                         p, q = 3, 3
    #                     if index == 4:
    #                         p, q = 3, 3
    #                     if index == 5:
    #                         p, q = 3, 3
    #                     if index == 6:
    #                         p, q = 3, 3
    #                     if index == 7:
    #                         p, q = 3, 3
    #                     if index == 8:
    #                         p, q = 3, 3
    #                     if index == 9:
    #                         p, q = 3, 3
    #                     if index == 10:
    #                         p, q = 3, 3
    #                     if index == 11:
    #                         p, q = 3, 3
    #                     if index == 12:
    #                         p, q = 3, 3
    #                     if index == 13:
    #                         p, q = 3, 3
    #                     if index == 14:
    #                         p, q = 3, 3
    #                     if index == 15:
    #                         p, q = 3, 3
    #                     if index == 16:
    #                         p, q = 3, 3
    #                     if index == 17:
    #                         p, q = 3, 3
    #                     if index == 18:
    #                         p, q = 3, 3
    #                     if index == 19:
    #                         p, q = 3, 3
    #             if year == 2018:
    #                 if index == -1:
    #                     p, q = 3, 3
    #                 else:
    #                     if index == 0:
    #                         p, q = 3, 3
    #                     if index == 1:
    #                         p, q = 3, 3
    #                     if index == 2:
    #                         p, q = 3, 3
    #                     if index == 3:
    #                         p, q = 3, 3
    #                     if index == 4:
    #                         p, q = 3, 3
    #                     if index == 5:
    #                         p, q = 3, 3
    #                     if index == 6:
    #                         p, q = 3, 3
    #                     if index == 7:
    #                         p, q = 3, 3
    #                     if index == 8:
    #                         p, q = 3, 3
    #                     if index == 9:
    #                         p, q = 3, 3
    #                     if index == 10:
    #                         p, q = 3, 3
    #                     if index == 11:
    #                         p, q = 3, 3
    #                     if index == 12:
    #                         p, q = 3, 3
    #                     if index == 13:
    #                         p, q = 3, 3
    #                     if index == 14:
    #                         p, q = 3, 3
    #                     if index == 15:
    #                         p, q = 3, 3
    #                     if index == 16:
    #                         p, q = 3, 3
    #                     if index == 17:
    #                         p, q = 3, 3
    #                     if index == 18:
    #                         p, q = 3, 3
    #                     if index == 19:
    #                         p, q = 3, 3
    #         else:
    #             if year == 2011:
    #                 if index == -1:
    #                     p, q = 3, 3
    #                 else:
    #                     if index == 0:
    #                         p, q = 3, 3
    #                     if index == 1:
    #                         p, q = 3, 3
    #                     if index == 2:
    #                         p, q = 3, 3
    #                     if index == 3:
    #                         p, q = 3, 3
    #                     if index == 4:
    #                         p, q = 3, 3
    #                     if index == 5:
    #                         p, q = 3, 3
    #                     if index == 6:
    #                         p, q = 3, 3
    #                     if index == 7:
    #                         p, q = 3, 3
    #                     if index == 8:
    #                         p, q = 3, 3
    #                     if index == 9:
    #                         p, q = 3, 3
    #                     if index == 10:
    #                         p, q = 3, 3
    #                     if index == 11:
    #                         p, q = 3, 3
    #                     if index == 12:
    #                         p, q = 3, 3
    #                     if index == 13:
    #                         p, q = 3, 3
    #                     if index == 14:
    #                         p, q = 3, 3
    #                     if index == 15:
    #                         p, q = 3, 3
    #                     if index == 16:
    #                         p, q = 3, 3
    #                     if index == 17:
    #                         p, q = 3, 3
    #                     if index == 18:
    #                         p, q = 3, 3
    #                     if index == 19:
    #                         p, q = 3, 3
    #             if year == 2012:
    #                 if index == -1:
    #                     p, q = 3, 3
    #                 else:
    #                     if index == 0:
    #                         p, q = 3, 3
    #                     if index == 1:
    #                         p, q = 3, 3
    #                     if index == 2:
    #                         p, q = 3, 3
    #                     if index == 3:
    #                         p, q = 3, 3
    #                     if index == 4:
    #                         p, q = 3, 3
    #                     if index == 5:
    #                         p, q = 3, 3
    #                     if index == 6:
    #                         p, q = 3, 3
    #                     if index == 7:
    #                         p, q = 3, 3
    #                     if index == 8:
    #                         p, q = 3, 3
    #                     if index == 9:
    #                         p, q = 3, 3
    #                     if index == 10:
    #                         p, q = 3, 3
    #                     if index == 11:
    #                         p, q = 3, 3
    #                     if index == 12:
    #                         p, q = 3, 3
    #                     if index == 13:
    #                         p, q = 3, 3
    #                     if index == 14:
    #                         p, q = 3, 3
    #                     if index == 15:
    #                         p, q = 3, 3
    #                     if index == 16:
    #                         p, q = 3, 3
    #                     if index == 17:
    #                         p, q = 3, 3
    #                     if index == 18:
    #                         p, q = 3, 3
    #                     if index == 19:
    #                         p, q = 3, 3
    #             if year == 2013:
    #                 if index == -1:
    #                     p, q = 3, 3
    #                 else:
    #                     if index == 0:
    #                         p, q = 3, 3
    #                     if index == 1:
    #                         p, q = 3, 3
    #                     if index == 2:
    #                         p, q = 3, 3
    #                     if index == 3:
    #                         p, q = 3, 3
    #                     if index == 4:
    #                         p, q = 3, 3
    #                     if index == 5:
    #                         p, q = 3, 3
    #                     if index == 6:
    #                         p, q = 3, 3
    #                     if index == 7:
    #                         p, q = 3, 3
    #                     if index == 8:
    #                         p, q = 3, 3
    #                     if index == 9:
    #                         p, q = 3, 3
    #                     if index == 10:
    #                         p, q = 3, 3
    #                     if index == 11:
    #                         p, q = 3, 3
    #                     if index == 12:
    #                         p, q = 3, 3
    #                     if index == 13:
    #                         p, q = 3, 3
    #                     if index == 14:
    #                         p, q = 3, 3
    #                     if index == 15:
    #                         p, q = 3, 3
    #                     if index == 16:
    #                         p, q = 3, 3
    #                     if index == 17:
    #                         p, q = 3, 3
    #                     if index == 18:
    #                         p, q = 3, 3
    #                     if index == 19:
    #                         p, q = 3, 3
    #             if year == 2014:
    #                 if index == -1:
    #                     p, q = 3, 3
    #                 else:
    #                     if index == 0:
    #                         p, q = 3, 3
    #                     if index == 1:
    #                         p, q = 3, 3
    #                     if index == 2:
    #                         p, q = 3, 3
    #                     if index == 3:
    #                         p, q = 3, 3
    #                     if index == 4:
    #                         p, q = 3, 3
    #                     if index == 5:
    #                         p, q = 3, 3
    #                     if index == 6:
    #                         p, q = 3, 3
    #                     if index == 7:
    #                         p, q = 3, 3
    #                     if index == 8:
    #                         p, q = 3, 3
    #                     if index == 9:
    #                         p, q = 3, 3
    #                     if index == 10:
    #                         p, q = 3, 3
    #                     if index == 11:
    #                         p, q = 3, 3
    #                     if index == 12:
    #                         p, q = 3, 3
    #                     if index == 13:
    #                         p, q = 3, 3
    #                     if index == 14:
    #                         p, q = 3, 3
    #                     if index == 15:
    #                         p, q = 3, 3
    #                     if index == 16:
    #                         p, q = 3, 3
    #                     if index == 17:
    #                         p, q = 3, 3
    #                     if index == 18:
    #                         p, q = 3, 3
    #                     if index == 19:
    #                         p, q = 3, 3
    #             if year == 2015:
    #                 if index == -1:
    #                     p, q = 3, 3
    #                 else:
    #                     if index == 0:
    #                         p, q = 3, 3
    #                     if index == 1:
    #                         p, q = 3, 3
    #                     if index == 2:
    #                         p, q = 3, 3
    #                     if index == 3:
    #                         p, q = 3, 3
    #                     if index == 4:
    #                         p, q = 3, 3
    #                     if index == 5:
    #                         p, q = 3, 3
    #                     if index == 6:
    #                         p, q = 3, 3
    #                     if index == 7:
    #                         p, q = 3, 3
    #                     if index == 8:
    #                         p, q = 3, 3
    #                     if index == 9:
    #                         p, q = 3, 3
    #                     if index == 10:
    #                         p, q = 3, 3
    #                     if index == 11:
    #                         p, q = 3, 3
    #                     if index == 12:
    #                         p, q = 3, 3
    #                     if index == 13:
    #                         p, q = 3, 3
    #                     if index == 14:
    #                         p, q = 3, 3
    #                     if index == 15:
    #                         p, q = 3, 3
    #                     if index == 16:
    #                         p, q = 3, 3
    #                     if index == 17:
    #                         p, q = 3, 3
    #                     if index == 18:
    #                         p, q = 3, 3
    #                     if index == 19:
    #                         p, q = 3, 3
    #             if year == 2016:
    #                 if index == -1:
    #                     p, q = 3, 3
    #                 else:
    #                     if index == 0:
    #                         p, q = 3, 3
    #                     if index == 1:
    #                         p, q = 3, 3
    #                     if index == 2:
    #                         p, q = 3, 3
    #                     if index == 3:
    #                         p, q = 3, 3
    #                     if index == 4:
    #                         p, q = 3, 3
    #                     if index == 5:
    #                         p, q = 3, 3
    #                     if index == 6:
    #                         p, q = 3, 3
    #                     if index == 7:
    #                         p, q = 3, 3
    #                     if index == 8:
    #                         p, q = 3, 3
    #                     if index == 9:
    #                         p, q = 3, 3
    #                     if index == 10:
    #                         p, q = 3, 3
    #                     if index == 11:
    #                         p, q = 3, 3
    #                     if index == 12:
    #                         p, q = 3, 3
    #                     if index == 13:
    #                         p, q = 3, 3
    #                     if index == 14:
    #                         p, q = 3, 3
    #                     if index == 15:
    #                         p, q = 3, 3
    #                     if index == 16:
    #                         p, q = 3, 3
    #                     if index == 17:
    #                         p, q = 3, 3
    #                     if index == 18:
    #                         p, q = 3, 3
    #                     if index == 19:
    #                         p, q = 3, 3
    #             if year == 2017:
    #                 if index == -1:
    #                     p, q = 3, 3
    #                 else:
    #                     if index == 0:
    #                         p, q = 3, 3
    #                     if index == 1:
    #                         p, q = 3, 3
    #                     if index == 2:
    #                         p, q = 3, 3
    #                     if index == 3:
    #                         p, q = 3, 3
    #                     if index == 4:
    #                         p, q = 3, 3
    #                     if index == 5:
    #                         p, q = 3, 3
    #                     if index == 6:
    #                         p, q = 3, 3
    #                     if index == 7:
    #                         p, q = 3, 3
    #                     if index == 8:
    #                         p, q = 3, 3
    #                     if index == 9:
    #                         p, q = 3, 3
    #                     if index == 10:
    #                         p, q = 3, 3
    #                     if index == 11:
    #                         p, q = 3, 3
    #                     if index == 12:
    #                         p, q = 3, 3
    #                     if index == 13:
    #                         p, q = 3, 3
    #                     if index == 14:
    #                         p, q = 3, 3
    #                     if index == 15:
    #                         p, q = 3, 3
    #                     if index == 16:
    #                         p, q = 3, 3
    #                     if index == 17:
    #                         p, q = 3, 3
    #                     if index == 18:
    #                         p, q = 3, 3
    #                     if index == 19:
    #                         p, q = 3, 3
    #             if year == 2018:
    #                 if index == -1:
    #                     p, q = 3, 3
    #                 else:
    #                     if index == 0:
    #                         p, q = 3, 3
    #                     if index == 1:
    #                         p, q = 3, 3
    #                     if index == 2:
    #                         p, q = 3, 3
    #                     if index == 3:
    #                         p, q = 3, 3
    #                     if index == 4:
    #                         p, q = 3, 3
    #                     if index == 5:
    #                         p, q = 3, 3
    #                     if index == 6:
    #                         p, q = 3, 3
    #                     if index == 7:
    #                         p, q = 3, 3
    #                     if index == 8:
    #                         p, q = 3, 3
    #                     if index == 9:
    #                         p, q = 3, 3
    #                     if index == 10:
    #                         p, q = 3, 3
    #                     if index == 11:
    #                         p, q = 3, 3
    #                     if index == 12:
    #                         p, q = 3, 3
    #                     if index == 13:
    #                         p, q = 3, 3
    #                     if index == 14:
    #                         p, q = 3, 3
    #                     if index == 15:
    #                         p, q = 3, 3
    #                     if index == 16:
    #                         p, q = 3, 3
    #                     if index == 17:
    #                         p, q = 3, 3
    #                     if index == 18:
    #                         p, q = 3, 3
    #                     if index == 19:
    #                         p, q = 3, 3

    bics = list()
    for p in range(pMax + 1):
        tmp = list()
        for q in range(qMax + 1):
            try:
                tmp.append(arima_model.ARIMA(series, (p, 1, q)).fit().bic)
            except Exception as e:
                # print(str(e))
                tmp.append(1e+10)  # 加入一个很大的数
        bics.append(tmp)
    bics = pd.DataFrame(bics)
    p, q = bics.stack().idxmin()

    # 建模
    model = arima_model.ARIMA(series, order=(p, 1, q)).fit()
    predict = model.forecast(n)[0]
    # print('BIC准则下确定p,q为%s,%s' % (p, q))
    return {
        'model': {'value': model, 'desc': '模型'},
        'unitP': {'value': unitP, 'desc': unitAssess},
        'noiseP': {'value': noiseP[0], 'desc': noiseAssess},
        'p': {'value': p, 'desc': 'AR模型阶数'},
        'q': {'value': q, 'desc': 'MA模型阶数'},
        'params': {'value': model.params, 'desc': '模型系数'},
        'predict': {'value': predict, 'desc': '往后预测%d个的序列' % (n)}
    }


def get_img(arima_score, mf_arima_score, wa_arima_score):
    choosed = 0
    maxn = arima_score
    if mf_arima_score > maxn:
        choosed = 1
        maxn = mf_arima_score
    if wa_arima_score > maxn:
        choosed = 2
        maxn = wa_arima_score
    return choosed


def cal_ano(y2, var, year, method, month):
    if var == 0 and method == 0:
        df = np.array(pd.read_csv('DecisionTree/static/beijing_tmean_mean_1981_2010.csv')).squeeze()
        for i in range(12):
            for j in range(year-2010):
                y2[i + j * 12] -= df[i]
    if var == 0 and method == 1:
        df = np.array(pd.read_csv('DecisionTree/static/beijing_tmean_mean_1981_2010.csv')).squeeze()
        for j in range(year-2010):
            y2[j] -= df[month-1]
    if var == 1 and method == 0:
        df = np.array(pd.read_csv('DecisionTree/static/beijing_pr_mean_1981_2010.csv')).squeeze()
        for i in range(12):
            for j in range(year-2010):
                y2[i + j * 12] -= df[i]
    if var == 1 and method == 1:
        df = np.array(pd.read_csv('DecisionTree/static/beijing_pr_mean_1981_2010.csv')).squeeze()
        for j in range(year-2010):
            y2[j] -= df[month-1]
    return y2


def TimeSeries(request):
    ip = globalIp
    # settings.TRAINING_TIMESERIES = 1
    var, year, method, month = get_parameter(request)
    predict_year = "2018"
    if is_illegal(var, year, method, month, predict_year):
        settings.TRAINING_TIMESERIES = 0
        return HttpResponse(json.dumps({"code": 1, "msg": "必须输入2011-2018间的整数年份", "data": []}, ensure_ascii=False),
                            content_type="application/json")
    info = year + var + method + month
    info_model = models.timeseries_data.objects.filter(info=info)
    if info_model.exists() != False:
        settings.TRAINING_TIMESERIES = 0
        arima_score = model_to_dict(info_model[0])['arima_score']
        mf_arima_score = model_to_dict(info_model[0])['mfarima_score']
        wa_arima_score = model_to_dict(info_model[0])['waarima_score']
        LIST = []
        data_dict = {}
        data_dict['name'] = '自回归'
        ari_ma_score = np.around(arima_score, 2)
        data_dict['score'] = ari_ma_score
        arima_img = "http://" + ip + '/' + 'static/arima' + info + '.png'
        data_dict['img'] = arima_img
        LIST.append(data_dict)
        data_dict = {}
        data_dict['name'] = '平滑滤波+自回归'
        mfarima_score = np.around(mf_arima_score, 2)
        data_dict['score'] = mfarima_score
        mfarima_img = "http://" + ip + '/' + 'static/mf_arima' + info + '.png'
        data_dict['img'] = mfarima_img
        LIST.append(data_dict)
        data_dict = {}
        data_dict['name'] = '小波分析+自回归'
        waarima_score = np.around(wa_arima_score, 2)
        data_dict['score'] = waarima_score
        waarima_img = "http://" + ip + '/' + 'static/wa_arima' + info + '.png'
        data_dict['img'] = waarima_img
        LIST.append(data_dict)
        data_dict = {}
        data_dict['train'] = LIST
        LIST = []
        choosed = get_img(arima_score, mf_arima_score, wa_arima_score)
        data_dict1 = {}
        var = int(var)

        LIST = []
        data_dict1['name'] = model_to_dict(info_model[0])['predict_name']
        data_dict1['score'] = model_to_dict(info_model[0])['predict_score']
        data_dict1['img'] = model_to_dict(info_model[0])['predict_img']
        LIST.append(data_dict1)
        data_dict['predict'] = LIST

        LIST = []
        data_dict1 = {}
        data_dict1['name'] = '自回归'
        data_dict1['score'] = arima_score
        data_dict1['img'] = "http://" + ip + '/' + 'static/TimeSeries_' + info + '_1.png'
        data_dict1['最佳模型名称'] = '自回归模型'
        data_dict1['模型框架'] = 'statsmodels框架'
        data_dict1[
            '算法说明'] = '模块使用自回归模型全称差分整合移动平均自回归模型，时间序列的预测方法之一。ARIMA(p，d，q)，p表示自回归项数，d表示差分阶数，q为滑动平均项数。模块中，d统一取1即可获得平稳时间序列，训练获得最佳p、q值。'
        data_dict1['预测因子智能优选方法'] = '不使用预测因子'
        if var == 0:
            data_dict1['数据集'] = '1981-2018年气温资料'
            data_dict1['数据预处理'] = '无'
            data_dict1['训练样本'] = '月平均气温'
        else:
            data_dict1['数据集'] = '1981-2018年降水资料'
            data_dict1['数据预处理'] = '无'
            data_dict1['训练样本'] = '月平均降水'
        LIST.append(data_dict1)
        data_dict1 = {}
        data_dict1['name'] = '平滑滤波+自回归'
        data_dict1['score'] = mf_arima_score
        data_dict1['img'] = "http://" + ip + '/' + 'static/TimeSeries_' + info + '_2.png'
        data_dict1['最佳模型名称'] = '平滑滤波+自回归模型'
        data_dict1['模型框架'] = 'statsmodels框架'
        data_dict1[
            '算法说明'] = '模块使用自回归模型全称差分整合移动平均自回归模型，时间序列的预测方法之一。ARIMA(p，d，q)，p表示自回归项数，d表示差分阶数，q为滑动平均项数。模块中，d统一取1即可获得平稳时间序列，训练获得最佳p、q值。平滑滤波的主要作用是通过对序列邻域求算数平均值而达到消除噪声的目的，模块采用的是连续三个采样求平均。'
        data_dict1['预测因子智能优选方法'] = '不使用预测因子'
        if var == 0:
            data_dict1['数据集'] = '1981-2018年气温资料'
            data_dict1['数据预处理'] = '无'
            data_dict1['训练样本'] = '月平均气温'
        else:
            data_dict1['数据集'] = '1981-2018年降水资料'
            data_dict1['数据预处理'] = '无'
            data_dict1['训练样本'] = '月平均降水'
        LIST.append(data_dict1)
        data_dict1 = {}
        data_dict1['name'] = '小波分析+自回归'
        data_dict1['score'] = wa_arima_score
        data_dict1['img'] = "http://" + ip + '/' + 'static/TimeSeries_' + info + '_3.png'
        data_dict1['最佳模型名称'] = '小波分析+自回归模型'
        data_dict1['模型框架'] = 'statsmodels框架'
        data_dict1[
            '算法说明'] = '模块使用自回归模型全称差分整合移动平均自回归模型，时间序列的预测方法之一。ARIMA(p，d，q)，p表示自回归项数，d表示差分阶数，q为滑动平均项数。模块中，d统一取1即可获得平稳时间序列，训练获得最佳p、q值。小波分析是对原时间序列进行小波分解后滤波，再重构成小波序列，模块选用db1小波进行小波分析。'
        data_dict1['预测因子智能优选方法'] = '不使用预测因子'
        if var == 0:
            data_dict1['数据集'] = '1981-2018年气温资料'
            data_dict1['数据预处理'] = '无'
            data_dict1['训练样本'] = '月平均气温'
        else:
            data_dict1['数据集'] = '1981-2018年降水资料'
            data_dict1['数据预处理'] = '无'
            data_dict1['训练样本'] = '月平均降水'
        LIST.append(data_dict1)
        data_dict['judge'] = LIST
        return HttpResponse(json.dumps({"code": 0, "msg": "success", "data": data_dict}, ensure_ascii=False),
                            content_type="application/json")

    settings.TRAINING_TIMESERIES = 1
    var, year, method, month = strToint4(var, year, method, month)
    train, test = get_data(var, year, method, month, 0)
    # 自回归
    result_arima = ARIMA(train, len(test), var, method, year, 0, -1)  # 预测结果,一阶差分偏自相关图,一阶差分自相关图
    # 平滑滤波+自回归
    result_mf_arima = ARIMA(np.array(mean_filter(3, train)).reshape(-1, 1), len(test), var, method, year, 1, -1)
    # 小波分析+自回归
    result_wa_arima = ARIMA(Wavelet_Transform(train).reshape(-1, 1), len(test), var, method, year, 2, -1)

    predict_arima = result_arima['predict']['value']
    # predict_arima = np.around(predict_arima, 2)
    predict_mf_arima = result_mf_arima['predict']['value']
    # predict_mf_arima = np.around(predict_mf_arima, 2)
    predict_wa_arima = result_wa_arima['predict']['value']
    # predict_wa_arima = np.around(predict_wa_arima, 2)


    # 可视化
    plt.clf()  # 清理历史绘图
    # 用来正常显示中文标签
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # 用来正常显示负号
    plt.rcParams['axes.unicode_minus'] = False
    ax = plt.gca()  # 获取边框
    ax.spines['top'].set_color("white")
    ax.spines['bottom'].set_color("white")
    ax.spines['left'].set_color("white")
    ax.spines['right'].set_color("white")

    # x1：训练集长度 x2：测试集长度
    if method == 0:
        x2 = []
        z = datetime.datetime(2011, 1, 1)
        for i in range((year-2010)*12):
            a = z + dateutil.relativedelta.relativedelta(months=i)
            x2.append(a)
    if method == 1:
        x2 = np.arange(2011, year+1)

    y1 = train.squeeze()  # 训练集数据
    y2 = test.squeeze()  # 测试集数据
    y3 = predict_arima.squeeze()  # 预测数据
    if len(predict_arima) == 1:
        arima_score = r2_score([y2], [y3])
        y2 = cal_ano([y2], var, year, method, month)
        y3 = cal_ano([y3], var, year, method, month)
    else:
        arima_score = r2_score(y2, y3)
        y2 = cal_ano(y2, var, year, method, month)
        y3 = cal_ano(y3, var, year, method, month)

    # plt.subplot(131, fc='white')
    plt.figure(1)
    # plt.title("自回归")
    # plt.bar(x1, height=y1, width=0.5, align='center', color='b', edgecolor='b')
    plt.bar(x2, height=y2, width=0.5, align='center', color='#244FFE', edgecolor='#244FFE')
    plt.plot(x2, y3, "-o", color='#CD3834')
    plt.tick_params(axis='x', colors='white')
    plt.tick_params(axis='y', colors='white')
    plt.grid()
    currentDir = "static"  # 当前工作路径
    plt.savefig(currentDir + '/arima' + info + '.png', dpi=100, bbox_inches='tight', transparent=True)

    train, test = get_data(var, year, method, month, 0)
    y1 = train.squeeze()  # 训练集数据
    y2 = test.squeeze()  # 测试集数据
    y3 = predict_mf_arima.squeeze()  # 预测数据
    if len(predict_mf_arima) == 1:
        mf_arima_score = r2_score([y2], [y3])
        y2 = cal_ano([y2], var, year, method, month)
        y3 = cal_ano([y3], var, year, method, month)
    else:
        mf_arima_score = r2_score(y2, y3)
        y2 = cal_ano(y2, var, year, method, month)
        y3 = cal_ano(y3, var, year, method, month)

    plt.clf()  # 清理历史绘图
    # plt.subplot(132, fc='white')
    plt.figure(1)
    # plt.title("平滑滤波+自回归")
    # plt.bar(x1, height=y1, width=0.5, align='center', color='b', edgecolor='b')
    plt.bar(x2, height=y2, width=0.5, align='center', color='#244FFE', edgecolor='#244FFE')
    plt.plot(x2, y3, "-o", color='#CD3834')
    plt.tick_params(axis='x', colors='white')
    plt.tick_params(axis='y', colors='white')
    plt.grid()
    currentDir = "static"  # 当前工作路径
    plt.savefig(currentDir + '/mf_arima' + info + '.png', dpi=100, bbox_inches='tight', transparent=True)

    train, test = get_data(var, year, method, month, 0)
    y1 = train.squeeze()  # 训练集数据
    y2 = test.squeeze()  # 测试集数据
    y3 = predict_wa_arima.squeeze()  # 预测数据
    if len(predict_wa_arima) == 1:
        wa_arima_score = r2_score([y2], [y3])
        y2 = cal_ano([y2], var, year, method, month)
        y3 = cal_ano([y3], var, year, method, month)
    else:
        wa_arima_score = r2_score(y2, y3)
        y2 = cal_ano(y2, var, year, method, month)
        y3 = cal_ano(y3, var, year, method, month)

    plt.clf()  # 清理历史绘图
    # plt.subplot(133, fc='white')
    plt.figure(1)
    # plt.title("小波分析+自回归")
    # plt.bar(x1, height=y1, width=0.5, align='center', color='b', edgecolor='b')
    plt.bar(x2, height=y2, width=0.5, align='center', color='#244FFE', edgecolor='#244FFE')
    plt.plot(x2, y3, "-o", color='#CD3834')
    plt.tick_params(axis='x', colors='white')
    plt.tick_params(axis='y', colors='white')
    plt.grid()
    currentDir = "static"  # 当前工作路径
    plt.savefig(currentDir + '/wa_arima' + info + '.png', dpi=100, bbox_inches='tight', transparent=True)
    # arima_score = r2_score(test, predict_arima)
    # mf_arima_score = r2_score(test, predict_mf_arima)
    # wa_arima_score = r2_score(test, predict_wa_arima)

    LIST = []
    data_dict = {}
    data_dict['name'] = '自回归'
    ari_ma_score = np.around(arima_score, 2)
    data_dict['score'] = ari_ma_score
    arima_img = "http://" + ip + '/' + 'static/arima' + info + '.png'
    data_dict['img'] = arima_img
    LIST.append(data_dict)
    data_dict = {}
    data_dict['name'] = '平滑滤波+自回归'
    mfarima_score = np.around(mf_arima_score, 2)
    data_dict['score'] = mfarima_score
    mfarima_img = "http://" + ip + '/' + 'static/mf_arima' + info + '.png'
    data_dict['img'] = mfarima_img
    LIST.append(data_dict)
    data_dict = {}
    data_dict['name'] = '小波分析+自回归'
    waarima_score = np.around(wa_arima_score, 2)
    data_dict['score'] = waarima_score
    waarima_img = "http://" + ip + '/' + 'static/wa_arima' + info + '.png'
    data_dict['img'] = waarima_img
    LIST.append(data_dict)
    data_dict = {}
    data_dict['train'] = LIST

    train, test = get_data(var, year, method, month, 1)

    result_arima = []
    result_mf_arima = []
    result_wa_arima = []
    predict_arima = []
    predict_mf_arima = []
    predict_wa_arima = []
    # test_all = []
    for i in range(20):
        # 自回归
        result_arima.append(ARIMA(train[i], len(test[i]), var, method, year, 0, i)['predict']['value'])  # 预测结果,一阶差分偏自相关图,一阶差分自相关图
        result_arima[i] = cal_ano(result_arima[i], var, year, method, month)
        predict_arima.append(result_arima[i])
        # 平滑滤波+自回归
        result_mf_arima.append(ARIMA(np.array(mean_filter(3, train[i])).reshape(-1, 1), len(test[i]), var, method, year, 1, i)['predict']['value'])
        result_mf_arima[i] = cal_ano(result_mf_arima[i], var, year, method, month)
        predict_mf_arima.append(result_mf_arima[i])
        # 小波分析+自回归
        result_wa_arima.append(ARIMA(Wavelet_Transform(train[i]).reshape(-1, 1), len(test[i]), var, method, year, 2, i)['predict']['value'])
        result_wa_arima[i] = cal_ano(result_wa_arima[i], var, year, method, month)
        predict_wa_arima.append(result_wa_arima[i])

        # test_all.append(test[i])

    # arima_score = r2_score(test_all[0], predict_arima[0])
    # mf_arima_score = r2_score(test_all[0], predict_mf_arima[0])
    # wa_arima_score = r2_score(test_all[0], predict_wa_arima[0])

    var = str(var)  # 要素
    area = 'bj'  # 绘制区域 北京：bj，天津：tj，京津冀：jjj，内蒙古：nmg，华北：huabei，山西：shanxi
    num = 4  # 模态个数
    Month = month

    df = pd.read_csv('DecisionTree/static/beijing_test_zd_tmean_2011_2018.csv')
    sta_order_list = df.columns.values
    DRAW = []
    DRAW = np.array(DRAW)
    all_year = int(predict_year)-2010
    predict_arima = np.array(predict_arima)
    predict_mf_arima = np.array(predict_mf_arima)
    predict_wa_arima = np.array(predict_wa_arima)
    print(predict_arima)
    print(predict_arima.shape)
    print(predict_mf_arima)
    print(predict_mf_arima.shape)
    print(predict_wa_arima)
    print(predict_wa_arima.shape)

    if method == 0:
        DRAW = np.append(predict_arima[:, all_year*12-1-(12-Month):all_year*12-(12-Month)],
                         predict_mf_arima[:, all_year*12-1-(12-Month):all_year*12-(12-Month)])
        DRAW = np.append(DRAW, predict_wa_arima[:, all_year*12-1-(12-Month):all_year*12-(12-Month)])
    if method == 1:
        DRAW = np.append(predict_arima[:, all_year-1:all_year], predict_mf_arima[:, all_year-1:all_year])
        DRAW = np.append(DRAW, predict_wa_arima[:, all_year-1:all_year])
    DRAW = DRAW.reshape(-1, 20)
    # print(DRAW)
    draw_map(var, sta_order_list, DRAW, area, num, Month, info)

    LIST = []
    choosed = get_img(arima_score, mf_arima_score, wa_arima_score)
    data_dict1 = {}
    var = int(var)

    if choosed == 0:
        data_dict1['name'] = '自回归'
        data_dict1['score'] = np.around(arima_score, 2)
        data_dict1['img'] = "http://" + ip + '/' + 'static/TimeSeries_' + info + '_1.png'
    if choosed == 1:
        data_dict1['name'] = '平滑滤波+自回归'
        data_dict1['score'] = np.around(mf_arima_score, 2)
        data_dict1['img'] = "http://" + ip + '/' + 'static/TimeSeries_' + info + '_2.png'
    if choosed == 2:
        data_dict1['name'] = '小波分析+自回归'
        data_dict1['score'] = np.around(wa_arima_score, 2)
        data_dict1['img'] = "http://" + ip + '/' + 'static/TimeSeries_' + info + '_3.png'
    models.timeseries_data.objects.create(
        info=info, arima_score=arima_score, arima_img=arima_img, mfarima_score=mf_arima_score, mfarima_img=mfarima_img,
        waarima_score=wa_arima_score, waarima_img=waarima_img, predict_name=data_dict1['name'],
        predict_score=data_dict1['score'], predict_img=data_dict1['img'])
    LIST.append(data_dict1)
    data_dict['predict'] = LIST

    LIST = []
    data_dict1 = {}
    data_dict1['name'] = '自回归'
    data_dict1['score'] = arima_score
    data_dict1['img'] = "http://" + ip + '/' + 'static/TimeSeries_' + info + '_1.png'
    data_dict1['最佳模型名称'] = '自回归模型'
    data_dict1['模型框架'] = 'statsmodels框架'
    data_dict1['算法说明'] = '模块使用自回归模型全称差分整合移动平均自回归模型，时间序列的预测方法之一。ARIMA(p，d，q)，p表示自回归项数，d表示差分阶数，q为滑动平均项数。模块中，d统一取1即可获得平稳时间序列，训练获得最佳p、q值。'
    data_dict1['预测因子智能优选方法'] = '不使用预测因子'
    if var == 0:
        data_dict1['数据集'] = '1981-2018年气温资料'
        data_dict1['数据预处理'] = '无'
        data_dict1['训练样本'] = '月平均气温'
    else:
        data_dict1['数据集'] = '1981-2018年降水资料'
        data_dict1['数据预处理'] = '无'
        data_dict1['训练样本'] = '月平均降水'
    LIST.append(data_dict1)
    data_dict1 = {}
    data_dict1['name'] = '平滑滤波+自回归'
    data_dict1['score'] = mf_arima_score
    data_dict1['img'] = "http://" + ip + '/' + 'static/TimeSeries_' + info + '_2.png'
    data_dict1['最佳模型名称'] = '平滑滤波+自回归模型'
    data_dict1['模型框架'] = 'statsmodels框架'
    data_dict1['算法说明'] = '模块使用自回归模型全称差分整合移动平均自回归模型，时间序列的预测方法之一。ARIMA(p，d，q)，p表示自回归项数，d表示差分阶数，q为滑动平均项数。模块中，d统一取1即可获得平稳时间序列，训练获得最佳p、q值。平滑滤波的主要作用是通过对序列邻域求算数平均值而达到消除噪声的目的，模块采用的是连续三个采样求平均。'
    data_dict1['预测因子智能优选方法'] = '不使用预测因子'
    if var == 0:
        data_dict1['数据集'] = '1981-2018年气温资料'
        data_dict1['数据预处理'] = '无'
        data_dict1['训练样本'] = '月平均气温'
    else:
        data_dict1['数据集'] = '1981-2018年降水资料'
        data_dict1['数据预处理'] = '无'
        data_dict1['训练样本'] = '月平均降水'
    LIST.append(data_dict1)
    data_dict1 = {}
    data_dict1['name'] = '小波分析+自回归'
    data_dict1['score'] = wa_arima_score
    data_dict1['img'] = "http://" + ip + '/' + 'static/TimeSeries_' + info + '_3.png'
    data_dict1['最佳模型名称'] = '小波分析+自回归模型'
    data_dict1['模型框架'] = 'statsmodels框架'
    data_dict1['算法说明'] = '模块使用自回归模型全称差分整合移动平均自回归模型，时间序列的预测方法之一。ARIMA(p，d，q)，p表示自回归项数，d表示差分阶数，q为滑动平均项数。模块中，d统一取1即可获得平稳时间序列，训练获得最佳p、q值。小波分析是对原时间序列进行小波分解后滤波，再重构成小波序列，模块选用db1小波进行小波分析。'
    data_dict1['预测因子智能优选方法'] = '不使用预测因子'
    if var == 0:
        data_dict1['数据集'] = '1981-2018年气温资料'
        data_dict1['数据预处理'] = '无'
        data_dict1['训练样本'] = '月平均气温'
    else:
        data_dict1['数据集'] = '1981-2018年降水资料'
        data_dict1['数据预处理'] = '无'
        data_dict1['训练样本'] = '月平均降水'
    LIST.append(data_dict1)
    data_dict['judge'] = LIST
    settings.TRAINING_TIMESERIES = 0

    return HttpResponse(json.dumps({"code": 0, "msg": "success", "data": data_dict}, ensure_ascii=False),
                        content_type="application/json")
