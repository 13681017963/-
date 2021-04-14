from django.shortcuts import render
from django.conf import settings
# from Decisiontree import models
# from Decisiontree.models import
from django.http import HttpResponse
import json
from django.forms.models import model_to_dict
import datetime
import requests
from django.conf import settings
import os
import sys
from datetime import *
import datetime
import dateutil
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
from sklearn.pipeline import Pipeline
from DecisionTree import models
globalIp = '192.168.1.123:8998'


def test(request):
    return HttpResponse('ok2')


def get_data(var, year, method, month, predict_year, station):
    start = "198101"
    end = "201812"
    ind_file = "M_Atm_Nc"
    ind_path = "DecisionTree/static/"
    test_year = year - 2010
    if method == 0:
        t_month = algorithm.get_month_continuous_result(start, end)
        # 读取88项大气环流指数数据作为特征
        df = diag_service.getIndListMonth(t_month, start, end, ind_file, ind_path)  # 按月获取
        odf = df = pd.DataFrame(df)
        df.fillna(method='ffill', inplace=True)  # 前值填充
        df = (df-df.min())/(df.max()-df.min())  # 归一化
        df.fillna(method='bfill', inplace=True)  # 后值填充
        pca = PCA(n_components=0.9)  # 做PCA，取累计贡献率大于等于90%的因子作为预测因子
        df = pca.fit_transform(df)
        print('original shape:', odf.shape)
        print('now shape:', df.shape)
        df = pd.DataFrame(df)

        # 划分训练集
        X_train_origin = np.array(df.head(360))
        if station == 0:
            X_test_origin = np.array(df[slice(360, 360+test_year*12, 1)])
        if station == 1:
            X_test_origin = np.array(df.tail((predict_year-2010)*12))

    if method == 1:
        # 读取88项大气环流指数数据作为特征
        df = diag_service.getIndListMonthYearly(month, start, end, ind_file, ind_path)  # 按月获取
        odf = df = pd.DataFrame(df)
        df.fillna(method='ffill', inplace=True)  # 前值填充
        df = (df-df.min())/(df.max()-df.min())  # 归一化
        df.fillna(method='bfill', inplace=True)  # 后值填充
        df = df.dropna(axis=1, how='any')
        pca = PCA(n_components=0.9)  # 做PCA，取累计贡献率大于等于90%的因子作为预测因子
        df = pca.fit_transform(df)
        print('original shape:', odf.shape)
        print('now shape:', df.shape)
        df = pd.DataFrame(df)

        # 划分训练集
        X_train_origin = np.array(df.head(30))
        if station == 0:
            X_test_origin = np.array(df[slice(30, 30+test_year, 1)])
        if station == 1:
            X_test_origin = np.array(df.tail(predict_year-2010))

    # 读气温降水数据作为标签
    # var 0气温 1降水
    # method 0时间连续 1历年同期
    # station 0北京市站点平均值 1北京市20个站点值
    # 按月读取
    if var == 0 and method == 0 and station == 0:
        y_train_origin = np.array(pd.read_csv('DecisionTree/static/beijing_train_tmean_1981_2010.csv')).reshape(-1, 1)
        y_test_origin = np.array(pd.read_csv('DecisionTree/static/beijing_test_tmean_2011_2018.csv')).reshape(-1, 1)[:test_year*12, :]
    if var == 0 and method == 0 and station == 1:
        y_train_origin = np.array(pd.read_csv('DecisionTree/static/beijing_train_zd_tmean_1981_2010.csv'))
        y_test_origin = np.array(pd.read_csv('DecisionTree/static/beijing_test_zd_tmean_2011_2018.csv'))
    if var == 1 and method == 0 and station == 0:
        y_train_origin = np.array(pd.read_csv('DecisionTree/static/beijing_train_pr_1981_2010.csv')).reshape(-1, 1)
        y_test_origin = np.array(pd.read_csv('DecisionTree/static/beijing_test_pr_2011_2018.csv')).reshape(-1, 1)[:test_year*12, :]
    if var == 1 and method == 0 and station == 1:
        y_train_origin = np.array(pd.read_csv('DecisionTree/static/beijing_train_zd_pr_1981_2010.csv'))
        y_test_origin = np.array(pd.read_csv('DecisionTree/static/beijing_test_zd_pr_2011_2018.csv'))
    if var == 0 and method == 1 and station == 0:
        y_train_origin = np.array(pd.read_csv('DecisionTree/static/beijing_train_tmean_1981_2010.csv')).reshape(-1, 1)[month-1::12, :]
        y_test_origin = np.array(pd.read_csv('DecisionTree/static/beijing_test_tmean_2011_2018.csv')).reshape(-1, 1)[month-1:test_year*12:12, :]
    if var == 0 and method == 1 and station == 1:
        y_train_origin = np.array(pd.read_csv('DecisionTree/static/beijing_train_zd_tmean_1981_2010.csv'))[month-1::12, :]
        y_test_origin = np.array(pd.read_csv('DecisionTree/static/beijing_test_zd_tmean_2011_2018.csv'))[month-1::12, :]
    if var == 1 and method == 1 and station == 0:
        y_train_origin = np.array(pd.read_csv('DecisionTree/static/beijing_train_pr_1981_2010.csv')).reshape(-1, 1)[month-1::12, :]
        y_test_origin = np.array(pd.read_csv('DecisionTree/static/beijing_test_pr_2011_2018.csv')).reshape(-1, 1)[month-1:test_year*12:12, :]
    if var == 1 and method == 1 and station == 1:
        y_train_origin = np.array(pd.read_csv('DecisionTree/static/beijing_train_zd_pr_1981_2010.csv'))[month-1::12, :]
        y_test_origin = np.array(pd.read_csv('DecisionTree/static/beijing_test_zd_pr_2011_2018.csv'))[month-1::12, :]
    X_train = X_train_origin
    X_test = X_test_origin
    y_train = y_train_origin
    y_test = y_test_origin
    return X_train, X_test, y_train, y_test, y_train_origin, y_test_origin


def strToint5(a, b, c, d, e):
    a = int(a)
    b = int(b)
    c = int(c)
    d = int(d)
    e = int(e)
    return a, b, c, d, e


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
    # predict_year = request.GET.get("predict_year")
    return var, year, method, month#, predict_year


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
        OutPicFile1 = "static/" + "DecisionTree_" + info + "_"  + str(i + 1) + ".png"
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


def get_img(DecisionTreeRegressor_score, RandomForestRegressor_score,
            GradientBoostingRegressor_score, AdaBoostRegressor_score):
    choosed = 0
    maxn = DecisionTreeRegressor_score
    if RandomForestRegressor_score > maxn:
        choosed = 1
        maxn = RandomForestRegressor_score
    if GradientBoostingRegressor_score > maxn:
        choosed = 2
        maxn = GradientBoostingRegressor_score
    if AdaBoostRegressor_score > maxn:
        choosed = 3
        maxn = AdaBoostRegressor_score
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


def is_squeeze(nparray):
    if nparray.size != 1:
        nparray = nparray.squeeze()
    return nparray


def DecisionTree(request):
    # global x2
    ip = globalIp
    # settings.TRAINING_DECISIONTREE = 1
    var, year, method, month = get_parameter(request)
    print(var)
    predict_year = "2018"
    if is_illegal(var, year, method, month, predict_year):
        settings.TRAINING_DECISIONTREE = 0
        return HttpResponse(json.dumps({"code": 1, "msg": "必须输入2011-2018间的整数年份", "data": []}, ensure_ascii=False),
                            content_type="application/json")
    info = year + var + method + month
    print(info)
    info_model = models.decisiontree_data.objects.filter(info=info)
    print(info_model)
    if info_model.exists() != False:
        settings.TRAINING_DECISIONTREE = 0
        DecisionTreeRegressor_score = model_to_dict(info_model[0])['dt_score']
        RandomForestRegressor_score = model_to_dict(info_model[0])['rf_score']
        GradientBoostingRegressor_score = model_to_dict(info_model[0])['gb_score']
        AdaBoostRegressor_score = model_to_dict(info_model[0])['ada_score']
        print(info_model)
        LIST = []
        data_dict = {}
        data_dict['name'] = '二元决策树'
        dt_score = np.around(DecisionTreeRegressor_score, 2)
        data_dict['score'] = dt_score
        dt_img = "http://" + ip + '/' + 'static/dt' + info + '.png'
        data_dict['img'] = dt_img
        LIST.append(data_dict)
        data_dict = {}
        data_dict['name'] = '随机森林'
        rf_score = np.around(RandomForestRegressor_score, 2)
        data_dict['score'] = rf_score
        rf_img = "http://" + ip + '/' + 'static/rf' + info + '.png'
        data_dict['img'] = rf_img
        LIST.append(data_dict)
        data_dict = {}
        data_dict['name'] = '梯度提升树'
        gb_score = np.around(GradientBoostingRegressor_score, 2)
        data_dict['score'] = gb_score
        gb_img = "http://" + ip + '/' + 'static/gb' + info + '.png'
        data_dict['img'] = gb_img
        LIST.append(data_dict)
        data_dict = {}
        data_dict['name'] = '自适应增强树'
        ada_score = np.around(AdaBoostRegressor_score, 2)
        data_dict['score'] = ada_score
        ada_img = "http://" + ip + '/' + 'static/ada' + info + '.png'
        data_dict['img'] = ada_img
        LIST.append(data_dict)
        data_dict = {}
        data_dict['train'] = LIST

        LIST = []
        data_dict1 = {}
        data_dict1['name'] = model_to_dict(info_model[0])['predict_name']
        data_dict1['score'] = model_to_dict(info_model[0])['predict_score']
        data_dict1['img'] = model_to_dict(info_model[0])['predict_img']
        LIST.append(data_dict1)
        data_dict['predict'] = LIST

        LIST = []
        data_dict1 = {}
        data_dict1['name'] = '二元决策树'
        data_dict1['score'] = DecisionTreeRegressor_score
        data_dict1['img'] = "http://" + ip + '/' + 'static/DecisionTree_' + info + '_1.png'
        data_dict1['最佳模型名称'] = '二元决策树模型'
        data_dict1['模型框架'] = 'SKlearn框架'
        data_dict1[
            '算法说明'] = '遍历所有数据，尝试每个数据作为分割点，并计算此时左右两侧的数据的离差平方和，并从中找到最小值，然后找到离差平方和最小时对应的数据，它就是最佳分割点。它的两侧作为决策树的左右子树，每进行一次分割决策树深度加一。二元决策树模型仅关心值的分布，不关心值的具体大小，数据不用提前做归一化。'
        data_dict1['数据预处理'] = '大气环流指数最大最小归一化'
        data_dict1[
            '预测因子智能优选方法'] = '主成分分析(PCA)算法，根据相关文献，通常情况下，当主成分方差累积贡献率达到90%时，就能很好第反映相关因子的影响。主成分分析自变量：环流指数累积贡献率＞=90%时，得到预测因子特征。'
        if var == 0:
            data_dict1['数据集'] = '88项大气环流指数和1981-2018年气温资料'
            data_dict1['训练样本'] = '预测因子特征+月平均气温'
        else:
            data_dict1['数据集'] = '88项大气环流指数和1981-2018年降水资料'
            data_dict1['训练样本'] = '预测因子特征+月平均降水'
        LIST.append(data_dict1)
        data_dict1 = {}
        data_dict1['name'] = '随机森林'
        data_dict1['score'] = RandomForestRegressor_score
        data_dict1['img'] = "http://" + ip + '/' + 'static/DecisionTree_' + info + '_2.png'
        data_dict1['最佳模型名称'] = '随机森林模型'
        data_dict1['模型框架'] = 'SKlearn框架'
        data_dict1[
            '算法说明'] = '随机森林由多个决策树组成，并将它们合并到一起提供更加稳定准确的预测。与决策树遍历最佳分割点不同的是，随机森林随机选择特征构建最佳分割，因此随机森林只关心特征的分布，不关心特征的大小，数据不需要做归一化预处理。'
        data_dict1['数据预处理'] = '大气环流指数最大最小归一化'
        data_dict1[
            '预测因子智能优选方法'] = '主成分分析(PCA)算法，根据相关文献，通常情况下，当主成分方差累积贡献率达到90%时，就能很好第反映相关因子的影响。主成分分析自变量：环流指数累积贡献率＞=90%时，得到预测因子特征。'
        if var == 0:
            data_dict1['数据集'] = '88项大气环流指数和1981-2018年气温资料'
            data_dict1['训练样本'] = '预测因子特征+月平均气温'
        else:
            data_dict1['数据集'] = '88项大气环流指数和1981-2018年降水资料'
            data_dict1['训练样本'] = '预测因子特征+月平均降水'
        LIST.append(data_dict1)
        data_dict1 = {}
        data_dict1['name'] = '梯度提升树'
        data_dict1['score'] = GradientBoostingRegressor_score
        data_dict1['img'] = "http://" + ip + '/' + 'static/DecisionTree_' + info + '_3.png'
        data_dict1['最佳模型名称'] = '梯度提升树模型'
        data_dict1['模型框架'] = 'SKlearn框架'
        data_dict1[
            '算法说明'] = '梯度提升树是一种迭代的决策树算法，它基于集成学习中的boosting思想，每次迭代都在减少残差的梯度方向新建立一颗决策树，迭代多少次就会生成多少颗决策树。数据做归一化预处理可以加快在梯度方向迭代的速度，取得更好的结果。'
        data_dict1[
            '预测因子智能优选方法'] = '主成分分析(PCA)算法，根据相关文献，通常情况下，当主成分方差累积贡献率达到90%时，就能很好第反映相关因子的影响。主成分分析自变量：环流指数累积贡献率＞=90%时，得到预测因子特征。'
        if var == 0:
            data_dict1['数据集'] = '88项大气环流指数和1981-2018年气温资料'
            data_dict1['数据预处理'] = '大气环流指数最大最小归一化；气温月数据最大最小归一化'
            data_dict1['训练样本'] = '预测因子特征+月平均气温'
        else:
            data_dict1['数据集'] = '88项大气环流指数和1981-2018年降水资料'
            data_dict1['数据预处理'] = '大气环流指数最大最小归一化；降水月数据最大最小归一化'
            data_dict1['训练样本'] = '预测因子特征+月平均降水'
        LIST.append(data_dict1)
        data_dict1 = {}
        data_dict1['name'] = '自适应增强树'
        data_dict1['score'] = AdaBoostRegressor_score
        data_dict1['img'] = "http://" + ip + '/' + 'static/DecisionTree_' + info + '_4.png'
        data_dict1['最佳模型名称'] = '自适应增强树模型模型'
        data_dict1['模型框架'] = 'SKlearn框架'
        data_dict1[
            '算法说明'] = '自适应增强树实现个弱分类器的加权运算，自适应在于：前一个基本分类器分错的样本会得到加强，加权后的全体样本再次被用来训练下一个基本分类器，同时在每一轮中加入一个新的弱分类器，直到达到某个预定的足够小的错误率或达到预先指定的最大迭代次数。数据归一化预处理取决于选择的弱分类器是否需要，本模块选择二元决策树作为弱分类器，二元决策树的划分仅与值的分布有关，和值的大小无关，不需要归一化操作。'
        data_dict1['数据预处理'] = '大气环流指数最大最小归一化'
        data_dict1[
            '预测因子智能优选方法'] = '主成分分析(PCA)算法，根据相关文献，通常情况下，当主成分方差累积贡献率达到90%时，就能很好第反映相关因子的影响。主成分分析自变量：环流指数累积贡献率＞=90%时，得到预测因子特征。'
        if var == 0:
            data_dict1['数据集'] = '88项大气环流指数和1981-2018年气温资料'
            data_dict1['训练样本'] = '预测因子特征+月平均气温'
        else:
            data_dict1['数据集'] = '88项大气环流指数和1981-2018年降水资料'
            data_dict1['训练样本'] = '预测因子特征+月平均降水'
        LIST.append(data_dict1)
        data_dict['judge'] = LIST

        return HttpResponse(json.dumps({"code": 0, "msg": "success", "data": data_dict}, ensure_ascii=False),
                            content_type="application/json")
    settings.TRAINING_DECISIONTREE = 1
    # 预测年份如果要可修改，把predict_year写进get，去掉get_parameter中predict_year有关注释即可
    var, year, method, month, predict_year = strToint5(var, year, method, month, predict_year)
    X_train, X_test, y_train, y_test, y_train_origin, y_test_origin = get_data(var, year, method, month, predict_year, 0)
    ss = MinMaxScaler()
    X_train_GBRT = ss.fit_transform(X_train)
    X_test_GBRT = ss.fit_transform(X_test)
    y_train_GBRT = ss.fit_transform(y_train)
    y_test_GBRT = ss.fit_transform(y_test)
    List = []
    for i in range(10):
        for j in range(1, 10000):
            if int(j*0.001*360) == i:
                List.append(j*0.001)
                break

    if var == 0 and method == 0:
        model_DecisionTreeRegressor = DecisionTreeRegressor(max_depth=4, max_features='auto',
                                                            max_leaf_nodes=8,
                                                            min_samples_leaf=0.02,
                                                            random_state=10).fit(X_train, y_train)
        model_AdaBoostRegressor = AdaBoostRegressor(base_estimator=model_DecisionTreeRegressor,
                                                    learning_rate=0.9697272727272728,
                                                    n_estimators=20,
                                                    random_state=10).fit(X_train, y_train.ravel())
        model_RandomForestRegressor = ensemble.RandomForestRegressor(max_depth=7, max_leaf_nodes=24,
                                                                     min_samples_leaf=0.006,
                                                                     n_estimators=20,
                                                                     random_state=10).fit(X_train, y_train.ravel())
        model_GradientBoostingRegressor = ensemble.GradientBoostingRegressor(learning_rate=0.18263636363636365,
                                                                             max_depth=4, max_features='auto',
                                                                             max_leaf_nodes=8, min_samples_leaf=0.02,
                                                                             random_state=10).fit(X_train_GBRT, y_train_GBRT.ravel())
    if var == 0 and method == 1:
        tuned_parameters = {'max_features': ['auto', 'log2', 'sqrt'],
                            'max_depth': range(1, 11, 1), 'max_leaf_nodes': range(2, 12, 1),
                            'min_samples_leaf': List}
        grid = GridSearchCV(DecisionTreeRegressor(random_state=10), tuned_parameters, cv=5)
        grid = grid.fit(X_train, y_train)
        aa = grid.best_params_['max_depth']
        bb = grid.best_params_['max_features']
        cc = grid.best_params_['max_leaf_nodes']
        dd = grid.best_params_['min_samples_leaf']
        model_DecisionTreeRegressor = DecisionTreeRegressor(max_depth=aa, max_features=bb,
                                                            max_leaf_nodes=cc, min_samples_leaf=dd,
                                                            random_state=10).fit(X_train, y_train)
        tuned_parameters = {'learning_rate': np.linspace(0.001, 1, 100)}
        grid = GridSearchCV(AdaBoostRegressor(base_estimator=model_DecisionTreeRegressor, random_state=10), tuned_parameters, cv=5)
        grid = grid.fit(X_train, y_train.ravel())
        model_AdaBoostRegressor = AdaBoostRegressor(base_estimator=model_DecisionTreeRegressor,
                                                    learning_rate=grid.best_params_['learning_rate'],
                                                    random_state=10).fit(X_train, y_train.ravel())
        tuned_parameters = {'max_features': ['auto', 'log2', 'sqrt'],
                            'max_depth': range(1, 11, 1), 'max_leaf_nodes': range(2, 12, 1),}
        grid = GridSearchCV(ensemble.RandomForestRegressor(random_state=10), tuned_parameters, cv=5)
        grid = grid.fit(X_train, y_train)
        model_RandomForestRegressor = ensemble.RandomForestRegressor(max_depth=grid.best_params_['max_depth'],
                                                                     max_features=grid.best_params_['max_features'],
                                                                     max_leaf_nodes=grid.best_params_['max_leaf_nodes'],
                                                                     random_state=10).fit(X_train, y_train.ravel())
        GBRT = ensemble.GradientBoostingRegressor(max_depth=aa, max_features=bb,
                                                  max_leaf_nodes=cc, min_samples_leaf=dd,
                                                  random_state=10)
        pipe = Pipeline([("scaler", MinMaxScaler()), ("GBRT", GBRT)])
        tuned_parameters = {'GBRT__learning_rate': np.linspace(0.001, 1, 100)}
        grid = GridSearchCV(pipe, tuned_parameters, cv=5)
        grid = grid.fit(X_train, y_train.ravel())
        model_GradientBoostingRegressor = ensemble.GradientBoostingRegressor(learning_rate=grid.best_params_['GBRT__learning_rate'],
                                                                             max_depth=aa, max_features=bb,
                                                                             max_leaf_nodes=cc, min_samples_leaf=dd,
                                                                             random_state=10).fit(X_train_GBRT, y_train_GBRT.ravel())
    if var == 1 and method == 0:
        model_DecisionTreeRegressor = DecisionTreeRegressor(max_depth=2, max_features='auto', max_leaf_nodes=3,
                                                            min_samples_leaf=0.001,
                                                            random_state=10).fit(X_train, y_train)
        model_AdaBoostRegressor = AdaBoostRegressor(base_estimator=model_DecisionTreeRegressor,
                                                    learning_rate=0.9596363636363637, n_estimators=10,
                                                    random_state=10).fit(X_train, y_train.ravel())
        model_RandomForestRegressor = ensemble.RandomForestRegressor(max_depth=3, max_leaf_nodes=8, n_estimators=80,
                                                                     random_state=10).fit(X_train, y_train.ravel())
        model_GradientBoostingRegressor = ensemble.GradientBoostingRegressor(learning_rate=0.2330909090909091,
                                                                             max_depth=2, max_features='auto',
                                                                             max_leaf_nodes=3, min_samples_leaf=0.001,
                                                                             n_estimators=10, random_state=10).fit(X_train_GBRT, y_train_GBRT.ravel())
    if var == 1 and method == 1:
        tuned_parameters = {'max_features': ['auto', 'log2', 'sqrt'],
                            'max_depth': range(1, 11, 1), 'max_leaf_nodes': range(2, 12, 1),
                            'min_samples_leaf': List}
        grid = GridSearchCV(DecisionTreeRegressor(random_state=10), tuned_parameters, cv=5)
        grid = grid.fit(X_train, y_train)
        aa = grid.best_params_['max_depth']
        bb = grid.best_params_['max_features']
        cc = grid.best_params_['max_leaf_nodes']
        dd = grid.best_params_['min_samples_leaf']
        model_DecisionTreeRegressor = DecisionTreeRegressor(max_depth=aa, max_features=bb,
                                                            max_leaf_nodes=cc, min_samples_leaf=dd,
                                                            random_state=10).fit(X_train, y_train)
        tuned_parameters = {'learning_rate': np.linspace(0.001, 1, 100)}
        grid = GridSearchCV(AdaBoostRegressor(base_estimator=model_DecisionTreeRegressor, random_state=10), tuned_parameters, cv=5)
        grid = grid.fit(X_train, y_train.ravel())
        model_AdaBoostRegressor = AdaBoostRegressor(base_estimator=model_DecisionTreeRegressor,
                                                    learning_rate=grid.best_params_['learning_rate'],
                                                    random_state=10).fit(X_train, y_train.ravel())
        tuned_parameters = {'max_features': ['auto', 'log2', 'sqrt'], 'max_depth': range(1, 11, 1),
                            'max_leaf_nodes': range(2, 12, 1)}
        grid = GridSearchCV(ensemble.RandomForestRegressor(random_state=10), tuned_parameters, cv=5)
        grid = grid.fit(X_train, y_train)
        model_RandomForestRegressor = ensemble.RandomForestRegressor(max_depth=grid.best_params_['max_depth'],
                                                                     max_features=grid.best_params_['max_features'],
                                                                     max_leaf_nodes=grid.best_params_['max_leaf_nodes'],
                                                                     random_state=10).fit(X_train, y_train.ravel())
        GBRT = ensemble.GradientBoostingRegressor(max_depth=aa, max_features=bb,
                                                  max_leaf_nodes=cc, min_samples_leaf=dd,
                                                  random_state=10)
        pipe = Pipeline([("scaler", MinMaxScaler()), ("GBRT", GBRT)])
        tuned_parameters = {'GBRT__learning_rate': np.linspace(0.001, 1, 100)}
        grid = GridSearchCV(pipe, tuned_parameters, cv=5)
        grid = grid.fit(X_train, y_train.ravel())
        model_GradientBoostingRegressor = ensemble.GradientBoostingRegressor(learning_rate=grid.best_params_['GBRT__learning_rate'],
                                                                             max_depth=aa, max_features=bb,
                                                                             max_leaf_nodes=cc, min_samples_leaf=dd,
                                                                             random_state=10).fit(X_train_GBRT, y_train_GBRT.ravel())


    DecisionTreeRegressor_score = model_DecisionTreeRegressor.score(X_test, y_test)
    RandomForestRegressor_score = model_RandomForestRegressor.score(X_test, y_test)
    GradientBoostingRegressor_score = model_GradientBoostingRegressor.score(X_test_GBRT, y_test_GBRT)
    AdaBoostRegressor_score = model_AdaBoostRegressor.score(X_test, y_test)

    predict_DecisionTreeRegressor = model_DecisionTreeRegressor.predict(X_test)
    plt.clf()  # 清理历史绘图
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    ax = plt.gca()  # 获取边框
    ax.spines['top'].set_color("white")
    ax.spines['bottom'].set_color("white")
    ax.spines['left'].set_color("white")
    ax.spines['right'].set_color("white")

    # x1：训练集长度 x2：测试集长度 月
    if method == 0:
        x2 = []
        z = datetime.datetime(2011, 1, 1)
        for i in range((year-2010)*12):
            a = z + dateutil.relativedelta.relativedelta(months=i)
            x2.append(a)
    if method == 1:
        x1 = np.arange(1981, 2011)
        x2 = np.arange(2011, year+1)

    # 绘图 原始数据用柱状图表示，预测数据用折线图表示
    y1 = y_train_origin.squeeze()  # 训练集数据
    y2 = is_squeeze(y_test_origin)  # 测试集数据
    y3 = is_squeeze(predict_DecisionTreeRegressor)  # 预测数据
    # y3 = ss.inverse_transform(y3)
    # y2 = np.array(y2)
    y2 = cal_ano(y2, var, year, method, month)
    y3 = cal_ano(y3, var, year, method, month)
    y2 = y2.squeeze()
    y3 = y3.squeeze()

    # plt.title("二元决策树")
    # plt.bar(x=x1, height=y1, width=0.5, align='center', color='b', edgecolor='b')
    plt.bar(x2, height=y2, width=0.5, align='center', color='#244FFE', edgecolor='#244FFE')
    plt.plot(x2, y2, "-o", color='#244FFE')
    plt.plot(x2, y3, "-o", color='#CD3834')
    plt.tick_params(axis='x', colors='white')
    plt.tick_params(axis='y', colors='white')
    plt.grid()
    currentDir = "static"
    plt.savefig(currentDir+'/dt' + info + '.png', dpi=100, bbox_inches='tight', transparent=True)

    predict_RandomForestRegressor = model_RandomForestRegressor.predict(X_test)
    plt.clf()  # 清理历史绘图
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    # 绘图 原始数据用柱状图表示，预测数据用折线图表示
    y1 = y_train_origin.squeeze()  # 训练集数据
    # y2 = y_test_origin.squeeze()  # 测试集数据
    y3 = is_squeeze(predict_RandomForestRegressor)  # 预测数据
    # y3 = ss.inverse_transform(y3)
    # y2 = cal_ano(y2, var, year, method, month)
    y3 = cal_ano(y3, var, year, method, month)
    y3 = y3.squeeze()

    # plt.title("随机森林")
    # plt.bar(x=x1, height=y1, width=0.5, align='center', color='b', edgecolor='b')
    plt.bar(x2, height=y2, width=0.5, align='center', color='#244FFE', edgecolor='#244FFE')
    plt.plot(x2, y2, "-o", color='#244FFE')
    plt.plot(x2, y3, "-o", color='#CD3834')
    plt.tick_params(axis='x', colors='white')
    plt.tick_params(axis='y', colors='white')
    plt.grid()
    currentDir = "static"
    plt.savefig(currentDir+'/rf' + info + '.png', dpi=100, bbox_inches='tight', transparent=True)

    predict_GradientBoostingRegressor = model_GradientBoostingRegressor.predict(X_test_GBRT)
    plt.clf()  # 清理历史绘图
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    # 绘图 原始数据用柱状图表示，预测数据用折线图表示
    y1 = y_train_origin.squeeze()  # 训练集数据
    # y2 = y_test_origin.squeeze()  # 测试集数据
    y3 = is_squeeze(predict_GradientBoostingRegressor)  # 预测数据
    y3 = ss.inverse_transform(y3.reshape(-1, 1))
    # y2 = cal_ano(y2, var, year, method, month)
    y3 = cal_ano(y3, var, year, method, month)
    y3 = y3.squeeze()

    # plt.title("梯度提升树")
    # plt.bar(x=x1, height=y1, width=0.5, align='center', color='b', edgecolor='b')
    plt.bar(x2, height=y2, width=0.5, align='center', color='#244FFE', edgecolor='#244FFE')
    plt.plot(x2, y2, "-o", color='#244FFE')
    plt.plot(x2, y3, "-o", color='#CD3834')
    plt.tick_params(axis='x', colors='white')
    plt.tick_params(axis='y', colors='white')
    plt.grid()
    currentDir = "static"
    plt.savefig(currentDir + '/gb' + info + '.png', dpi=100, bbox_inches='tight', transparent=True)

    predict_AdaBoostRegressor = model_AdaBoostRegressor.predict(X_test)
    plt.clf()  # 清理历史绘图
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    # 绘图 原始数据用柱状图表示，预测数据用折线图表示
    y1 = y_train_origin.squeeze()  # 训练集数据
    # y2 = y_test_origin.squeeze()  # 测试集数据
    y3 = is_squeeze(predict_AdaBoostRegressor)  # 预测数据
    # y3 = ss.inverse_transform(y3)
    # y2 = cal_ano(y2, var, year, method, month)
    y3 = cal_ano(y3, var, year, method, month)
    y3 = y3.squeeze()

    # plt.title("自适应增强树")
    # plt.bar(x=x1, height=y1, width=0.5, align='center', color='b', edgecolor='b')
    plt.bar(x2, height=y2, width=0.5, align='center', color='#244FFE', edgecolor='#244FFE')
    plt.plot(x2, y2, "-o", color='#244FFE')
    plt.plot(x2, y3, "-o", color='#CD3834')
    plt.tick_params(axis='x', colors='white')
    plt.tick_params(axis='y', colors='white')
    plt.grid()
    currentDir = "static"
    plt.savefig(currentDir + '/ada' + info + '.png', dpi=100, bbox_inches='tight', transparent=True)

    LIST = []
    data_dict = {}
    data_dict['name'] = '二元决策树'
    dt_score = np.around(DecisionTreeRegressor_score, 2)
    data_dict['score'] = dt_score
    dt_img = "http://" + ip + '/' + 'static/dt' + info + '.png'
    data_dict['img'] = dt_img
    LIST.append(data_dict)
    data_dict = {}
    data_dict['name'] = '随机森林'
    rf_score = np.around(RandomForestRegressor_score, 2)
    data_dict['score'] = rf_score
    rf_img = "http://" + ip + '/' + 'static/rf' + info + '.png'
    data_dict['img'] = rf_img
    LIST.append(data_dict)
    data_dict = {}
    data_dict['name'] = '梯度提升树'
    gb_score = np.around(GradientBoostingRegressor_score, 2)
    data_dict['score'] = gb_score
    gb_img = "http://" + ip + '/' + 'static/gb' + info + '.png'
    data_dict['img'] = gb_img
    LIST.append(data_dict)
    data_dict = {}
    data_dict['name'] = '自适应增强树'
    ada_score = np.around(AdaBoostRegressor_score, 2)
    data_dict['score'] = ada_score
    ada_img = "http://" + ip + '/' + 'static/ada' + info + '.png'
    data_dict['img'] = ada_img
    LIST.append(data_dict)
    data_dict = {}
    data_dict['train'] = LIST

    X_train, X_test, y_train, y_test, y_train_origin, y_test_origin = get_data(var, year, method, month, predict_year, 1)

    ss = MinMaxScaler()
    X_train_GBRT = ss.fit_transform(X_train)
    X_test_GBRT = ss.fit_transform(X_test)
    y_train_GBRT = ss.fit_transform(y_train)
    y_test_GBRT = ss.fit_transform(y_test)
    # MultiOutputRegressor()
    # 创建集合梯度提升树AdaBoost模型，弱学习器为决策树回归器
    if var == 0:
        model_DecisionTreeRegressor = MultiOutputRegressor(DecisionTreeRegressor(max_depth=4, max_features='auto',
                                                                                 max_leaf_nodes=8, min_samples_leaf=0.02,
                                                                                 random_state=10)).fit(X_train, y_train)

        ####3.6Adaboost回归####
        model_AdaBoostRegressor = MultiOutputRegressor(
            AdaBoostRegressor(base_estimator=DecisionTreeRegressor(max_depth=4, max_features='auto', max_leaf_nodes=8,
                                                                   min_samples_leaf=0.02, random_state=10),
                              learning_rate=0.9697272727272728, n_estimators=20, random_state=10)).fit(X_train, y_train)

        # ####3.5随机森林回归####
        model_RandomForestRegressor = MultiOutputRegressor(ensemble.RandomForestRegressor(max_depth=7, max_leaf_nodes=24,
                                                                                          min_samples_leaf=0.006,
                                                                                          n_estimators=20, random_state=10))\
            .fit(X_train, y_train)

        ####3.7GBRT回归####
        model_GradientBoostingRegressor = MultiOutputRegressor(ensemble.GradientBoostingRegressor(learning_rate=0.18263636363636365,
                                                                                                  max_depth=4, max_features='auto',
                                                                                                  max_leaf_nodes=8,
                                                                                                  min_samples_leaf=0.02,
                                                                                                  random_state=10))\
            .fit(X_train_GBRT, y_train_GBRT)
    if var == 1:
        model_DecisionTreeRegressor = MultiOutputRegressor(DecisionTreeRegressor(max_depth=2, max_features='auto',
                                                                                 max_leaf_nodes=3, min_samples_leaf=0.001,
                                                                                 random_state=10)).fit(X_train, y_train)

        ####3.6Adaboost回归####
        model_AdaBoostRegressor = MultiOutputRegressor(
            AdaBoostRegressor(base_estimator=DecisionTreeRegressor(max_depth=2, max_features='auto',
                                                                   max_leaf_nodes=3, min_samples_leaf=0.001,
                                                                   random_state=10),
                              learning_rate=0.9596363636363637, n_estimators=10, random_state=10)).fit(X_train, y_train)

        # ####3.5随机森林回归####
        model_RandomForestRegressor = MultiOutputRegressor(ensemble.RandomForestRegressor(max_depth=3, max_leaf_nodes=8,
                                                                                          n_estimators=80, random_state=10))\
            .fit(X_train, y_train)

        ####3.7GBRT回归####
        model_GradientBoostingRegressor = MultiOutputRegressor(
            ensemble.GradientBoostingRegressor(learning_rate=0.2330909090909091, max_depth=2, max_features='auto',
                                               max_leaf_nodes=3, min_samples_leaf=0.001, n_estimators=10,
                                               random_state=10)).fit(X_train_GBRT, y_train_GBRT)

    ####3.8Bagging回归####
    # model_BaggingRegressor = ensemble.BaggingRegressor().fit(X, y.ravel())

    # CART决策树预测值
    predict_DecisionTreeRegressor = model_DecisionTreeRegressor.predict(X_test)
    # DecisionTreeRegressor_score = model_DecisionTreeRegressor.score(X_test, y_test)
    # print("CART决策树的决定系数是:", DecisionTreeRegressor_score)
    #
    predict_RandomForestRegressor = model_RandomForestRegressor.predict(X_test)
    # RandomForestRegressor_score = model_RandomForestRegressor.score(X_test, y_test)
    # print("随机森林的决定系数是:", RandomForestRegressor_score)
    #
    predict_GradientBoostingRegressor = model_GradientBoostingRegressor.predict(X_test_GBRT)
    # predict_GradientBoostingRegressor = ss.inverse_transform(predict_GradientBoostingRegressor)
    # GradientBoostingRegressor_score = model_GradientBoostingRegressor.score(X_test, y_test)
    # print("梯度提升树的决定系数是:", GradientBoostingRegressor_score)
    #
    predict_AdaBoostRegressor = model_AdaBoostRegressor.predict(X_test)
    # model_AdaBoostRegressor_score = model_AdaBoostRegressor.score(X_test, y_test)
    # print("自适应增强树的决定系数是:", model_AdaBoostRegressor_score)

    var = str(var)  # 要素
    area = 'bj'  # 绘制区域 北京：bj，天津：tj，京津冀：jjj，内蒙古：nmg，华北：huabei，山西：shanxi
    num = 5  # 模态个数
    Month = month

    df = pd.read_csv('DecisionTree/static/beijing_test_zd_tmean_2011_2018.csv')
    sta_order_list = df.columns.values
    # print(df)
    # print(predict_DecisionTreeRegressor)

    predict_DecisionTreeRegressor = predict_DecisionTreeRegressor.squeeze()
    predict_RandomForestRegressor = predict_RandomForestRegressor.squeeze()
    predict_GradientBoostingRegressor = predict_GradientBoostingRegressor.squeeze()
    predict_GradientBoostingRegressor = ss.inverse_transform(predict_GradientBoostingRegressor)
    predict_AdaBoostRegressor = predict_AdaBoostRegressor.squeeze()

    # print(predict_DecisionTreeRegressor)
    all_year = predict_year - 2010
    # print(predict_DecisionTreeRegressor[all_year*12-1-(12-Month):all_year*12-(12-Month), :])
    df0 = np.array(pd.read_csv('DecisionTree/static/beijing_tmean_mean_1981_2010.csv')).squeeze()
    df1 = np.array(pd.read_csv('DecisionTree/static/beijing_pr_mean_1981_2010.csv')).squeeze()
    tmean = df0[month-1]
    pr = df1[month-1]
    DRAW = np.zeros(shape=(4, 20))
    method = int(method)
    var = int(var)
    if method == 0 and var == 0:
        # print("ZZ")
        DRAW[0] = np.array(pd.DataFrame(predict_DecisionTreeRegressor[all_year*12-1-(12-Month):all_year*12-(12-Month), :])-tmean).squeeze()
        DRAW[1] = np.array(pd.DataFrame(predict_RandomForestRegressor[all_year*12-1-(12-Month):all_year*12-(12-Month), :])-tmean).squeeze()
        DRAW[2] = np.array(pd.DataFrame(predict_GradientBoostingRegressor[all_year*12-1-(12-Month):all_year*12-(12-Month), :])-tmean).squeeze()
        DRAW[3] = np.array(pd.DataFrame(predict_AdaBoostRegressor[all_year*12-1-(12-Month):all_year*12-(12-Month), :])-tmean).squeeze()
    if method == 0 and var == 1:
        DRAW[0] = np.array(pd.DataFrame(predict_DecisionTreeRegressor[all_year * 12 - 1 - (12 - Month):all_year * 12 - (12 - Month), :])-pr).squeeze()
        DRAW[1] = np.array(pd.DataFrame(predict_RandomForestRegressor[all_year * 12 - 1 - (12 - Month):all_year * 12 - (12 - Month), :])-pr).squeeze()
        DRAW[2] = np.array(pd.DataFrame(predict_GradientBoostingRegressor[all_year * 12 - 1 - (12 - Month):all_year * 12 - (12 - Month), :])-pr).squeeze()
        DRAW[3] = np.array(pd.DataFrame(predict_AdaBoostRegressor[all_year * 12 - 1 - (12 - Month):all_year * 12 - (12 - Month), :])-pr).squeeze()
    if method == 1 and var == 0:
        DRAW[0] = np.array(pd.DataFrame(predict_DecisionTreeRegressor[all_year-1:all_year, :])-tmean).squeeze()
        DRAW[1] = np.array(pd.DataFrame(predict_RandomForestRegressor[all_year-1:all_year, :])-tmean).squeeze()
        DRAW[2] = np.array(pd.DataFrame(predict_GradientBoostingRegressor[all_year-1:all_year, :])-tmean).squeeze()
        DRAW[3] = np.array(pd.DataFrame(predict_AdaBoostRegressor[all_year-1:all_year, :])-tmean).squeeze()
    if method == 1 and var == 1:
        DRAW[0] = np.array(pd.DataFrame(predict_DecisionTreeRegressor[all_year-1:all_year, :])-pr).squeeze()
        DRAW[1] = np.array(pd.DataFrame(predict_RandomForestRegressor[all_year-1:all_year, :])-pr).squeeze()
        DRAW[2] = np.array(pd.DataFrame(predict_GradientBoostingRegressor[all_year-1:all_year, :])-pr).squeeze()
        DRAW[3] = np.array(pd.DataFrame(predict_AdaBoostRegressor[all_year-1:all_year, :])-pr).squeeze()
    DRAW = DRAW.reshape(-1, 20)
    # print(DRAW)
    # print(predict_AdaBoostRegressor)
    # print(np.array(pd.DataFrame(predict_DecisionTreeRegressor[all_year*12-1-(12-Month):all_year*12-(12-Month), :])-tmean).squeeze())
    var = str(var)
    draw_map(var, sta_order_list, DRAW, area, num, Month, info)

    LIST = []
    choosed = get_img(DecisionTreeRegressor_score, RandomForestRegressor_score,
                      GradientBoostingRegressor_score, AdaBoostRegressor_score)
    data_dict1 = {}
    var = int(var)

    if choosed == 0:
        data_dict1['name'] = '二元决策树'
        data_dict1['score'] = np.around(DecisionTreeRegressor_score, 2)
        data_dict1['img'] = "http://" + ip + '/' + 'static/DecisionTree_' + info + '_1.png'
    if choosed == 1:
        data_dict1['name'] = '随机森林'
        data_dict1['score'] = np.around(RandomForestRegressor_score, 2)
        data_dict1['img'] = "http://" + ip + '/' + 'static/DecisionTree_' + info + '_2.png'
    if choosed == 2:
        data_dict1['name'] = '梯度提升树'
        data_dict1['score'] = np.around(GradientBoostingRegressor_score, 2)
        data_dict1['img'] = "http://" + ip + '/' + 'static/DecisionTree_' + info + '_3.png'
    if choosed == 3:
        data_dict1['name'] = '自适应增强树'
        data_dict1['score'] = np.around(AdaBoostRegressor_score, 2)
        data_dict1['img'] = "http://" + ip + '/' + 'static/DecisionTree_' + info + '_4.png'
    models.decisiontree_data.objects.create(
        info=info, dt_score=DecisionTreeRegressor_score, dt_img=dt_img, rf_score=RandomForestRegressor_score,
        rf_img=rf_img, gb_score=GradientBoostingRegressor_score, gb_img=gb_img, ada_score=AdaBoostRegressor_score,
        ada_img=ada_img, predict_name=data_dict1['name'], predict_score=data_dict1['score'],
        predict_img=data_dict1['img'])
    LIST.append(data_dict1)
    data_dict['predict'] = LIST

    LIST = []
    data_dict1 = {}
    data_dict1['name'] = '二元决策树'
    data_dict1['score'] = DecisionTreeRegressor_score
    data_dict1['img'] = "http://" + ip + '/' + 'static/DecisionTree_' + info + '_1.png'
    data_dict1['最佳模型名称'] = '二元决策树模型'
    data_dict1['模型框架'] = 'SKlearn框架'
    data_dict1['算法说明'] = '遍历所有数据，尝试每个数据作为分割点，并计算此时左右两侧的数据的离差平方和，并从中找到最小值，然后找到离差平方和最小时对应的数据，它就是最佳分割点。它的两侧作为决策树的左右子树，每进行一次分割决策树深度加一。二元决策树模型仅关心值的分布，不关心值的具体大小，数据不用提前做归一化。'
    data_dict1['数据预处理'] = '大气环流指数最大最小归一化'
    data_dict1['预测因子智能优选方法'] = '主成分分析(PCA)算法，根据相关文献，通常情况下，当主成分方差累积贡献率达到90%时，就能很好第反映相关因子的影响。主成分分析自变量：环流指数累积贡献率＞=90%时，得到预测因子特征。'
    if var == 0:
        data_dict1['数据集'] = '88项大气环流指数和1981-2018年气温资料'
        data_dict1['训练样本'] = '预测因子特征+月平均气温'
    else:
        data_dict1['数据集'] = '88项大气环流指数和1981-2018年降水资料'
        data_dict1['训练样本'] = '预测因子特征+月平均降水'
    LIST.append(data_dict1)
    data_dict1 = {}
    data_dict1['name'] = '随机森林'
    data_dict1['score'] = RandomForestRegressor_score
    data_dict1['img'] = "http://" + ip + '/' + 'static/DecisionTree_' + info + '_2.png'
    data_dict1['最佳模型名称'] = '随机森林模型'
    data_dict1['模型框架'] = 'SKlearn框架'
    data_dict1['算法说明'] = '随机森林由多个决策树组成，并将它们合并到一起提供更加稳定准确的预测。与决策树遍历最佳分割点不同的是，随机森林随机选择特征构建最佳分割，因此随机森林只关心特征的分布，不关心特征的大小，数据不需要做归一化预处理。'
    data_dict1['数据预处理'] = '大气环流指数最大最小归一化'
    data_dict1['预测因子智能优选方法'] = '主成分分析(PCA)算法，根据相关文献，通常情况下，当主成分方差累积贡献率达到90%时，就能很好第反映相关因子的影响。主成分分析自变量：环流指数累积贡献率＞=90%时，得到预测因子特征。'
    if var == 0:
        data_dict1['数据集'] = '88项大气环流指数和1981-2018年气温资料'
        data_dict1['训练样本'] = '预测因子特征+月平均气温'
    else:
        data_dict1['数据集'] = '88项大气环流指数和1981-2018年降水资料'
        data_dict1['训练样本'] = '预测因子特征+月平均降水'
    LIST.append(data_dict1)
    data_dict1 = {}
    data_dict1['name'] = '梯度提升树'
    data_dict1['score'] = GradientBoostingRegressor_score
    data_dict1['img'] = "http://" + ip + '/' + 'static/DecisionTree_' + info + '_3.png'
    data_dict1['最佳模型名称'] = '梯度提升树模型'
    data_dict1['模型框架'] = 'SKlearn框架'
    data_dict1['算法说明'] = '梯度提升树是一种迭代的决策树算法，它基于集成学习中的boosting思想，每次迭代都在减少残差的梯度方向新建立一颗决策树，迭代多少次就会生成多少颗决策树。数据做归一化预处理可以加快在梯度方向迭代的速度，取得更好的结果。'
    data_dict1['预测因子智能优选方法'] = '主成分分析(PCA)算法，根据相关文献，通常情况下，当主成分方差累积贡献率达到90%时，就能很好第反映相关因子的影响。主成分分析自变量：环流指数累积贡献率＞=90%时，得到预测因子特征。'
    if var == 0:
        data_dict1['数据集'] = '88项大气环流指数和1981-2018年气温资料'
        data_dict1['数据预处理'] = '大气环流指数最大最小归一化；气温月数据最大最小归一化'
        data_dict1['训练样本'] = '预测因子特征+月平均气温'
    else:
        data_dict1['数据集'] = '88项大气环流指数和1981-2018年降水资料'
        data_dict1['数据预处理'] = '大气环流指数最大最小归一化；降水月数据最大最小归一化'
        data_dict1['训练样本'] = '预测因子特征+月平均降水'
    LIST.append(data_dict1)
    data_dict1 = {}
    data_dict1['name'] = '自适应增强树'
    data_dict1['score'] = AdaBoostRegressor_score
    data_dict1['img'] = "http://" + ip + '/' + 'static/DecisionTree_' + info + '_4.png'
    data_dict1['最佳模型名称'] = '自适应增强树模型模型'
    data_dict1['模型框架'] = 'SKlearn框架'
    data_dict1['算法说明'] = '自适应增强树实现个弱分类器的加权运算，自适应在于：前一个基本分类器分错的样本会得到加强，加权后的全体样本再次被用来训练下一个基本分类器，同时在每一轮中加入一个新的弱分类器，直到达到某个预定的足够小的错误率或达到预先指定的最大迭代次数。数据归一化预处理取决于选择的弱分类器是否需要，本模块选择二元决策树作为弱分类器，二元决策树的划分仅与值的分布有关，和值的大小无关，不需要归一化操作。'
    data_dict1['数据预处理'] = '大气环流指数最大最小归一化'
    data_dict1['预测因子智能优选方法'] = '主成分分析(PCA)算法，根据相关文献，通常情况下，当主成分方差累积贡献率达到90%时，就能很好第反映相关因子的影响。主成分分析自变量：环流指数累积贡献率＞=90%时，得到预测因子特征。'
    if var == 0:
        data_dict1['数据集'] = '88项大气环流指数和1981-2018年气温资料'
        data_dict1['训练样本'] = '预测因子特征+月平均气温'
    else:
        data_dict1['数据集'] = '88项大气环流指数和1981-2018年降水资料'
        data_dict1['训练样本'] = '预测因子特征+月平均降水'
    LIST.append(data_dict1)
    data_dict['judge'] = LIST

    settings.TRAINING_DECISIONTREE = 0

    return HttpResponse(json.dumps({"code": 0, "msg": "success", "data": data_dict}, ensure_ascii=False),
                        content_type="application/json")
