# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import os
import sys
from datetime import *
import time
import numpy as np
import src
from src.config import algorithm
from src.config import diag_service
from src.config import station_config
import pandas as pd
import matplotlib.pyplot as plt
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
from sklearn.pipeline import Pipeline
import matplotlib.dates as mdates
from sklearn.pipeline import make_pipeline
import datetime
import dateutil
import dclimate.d_std_lib as d_std_lib
from matplotlib.patches import Ellipse
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from django.http import HttpResponse
import json
globalIp = '192.168.1.123:8998'


def get_data(var, year, method, month, predict_year, station):
    start = "198101"
    end = "201812"
    ind_file = "M_Atm_Nc"
    ind_path = "other_algorithm/static/"
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
        y_train_origin = np.array(pd.read_csv('other_algorithm/static/beijing_train_tmean_1981_2010.csv')).reshape(-1, 1)
        y_test_origin = np.array(pd.read_csv('other_algorithm/static/beijing_test_tmean_2011_2018.csv')).reshape(-1, 1)[:test_year*12, :]
    if var == 0 and method == 0 and station == 1:
        y_train_origin = np.array(pd.read_csv('other_algorithm/static/beijing_train_zd_tmean_1981_2010.csv'))
        y_test_origin = np.array(pd.read_csv('other_algorithm/static/beijing_test_zd_tmean_2011_2018.csv'))
    if var == 1 and method == 0 and station == 0:
        y_train_origin = np.array(pd.read_csv('other_algorithm/static/beijing_train_pr_1981_2010.csv')).reshape(-1, 1)
        y_test_origin = np.array(pd.read_csv('other_algorithm/static/beijing_test_pr_2011_2018.csv')).reshape(-1, 1)[:test_year*12, :]
    if var == 1 and method == 0 and station == 1:
        y_train_origin = np.array(pd.read_csv('other_algorithm/static/beijing_train_zd_pr_1981_2010.csv'))
        y_test_origin = np.array(pd.read_csv('other_algorithm/static/beijing_test_zd_pr_2011_2018.csv'))
    if var == 0 and method == 1 and station == 0:
        y_train_origin = np.array(pd.read_csv('other_algorithm/static/beijing_train_tmean_1981_2010.csv')).reshape(-1, 1)[month-1::12, :]
        y_test_origin = np.array(pd.read_csv('other_algorithm/static/beijing_test_tmean_2011_2018.csv')).reshape(-1, 1)[month-1:test_year*12:12, :]
    if var == 0 and method == 1 and station == 1:
        y_train_origin = np.array(pd.read_csv('other_algorithm/static/beijing_train_zd_tmean_1981_2010.csv'))[month-1::12, :]
        y_test_origin = np.array(pd.read_csv('other_algorithm/static/beijing_test_zd_tmean_2011_2018.csv'))[month-1::12, :]
    if var == 1 and method == 1 and station == 0:
        y_train_origin = np.array(pd.read_csv('other_algorithm/static/beijing_train_pr_1981_2010.csv')).reshape(-1, 1)[month-1::12, :]
        y_test_origin = np.array(pd.read_csv('other_algorithm/static/beijing_test_pr_2011_2018.csv')).reshape(-1, 1)[month-1:test_year*12:12, :]
    if var == 1 and method == 1 and station == 1:
        y_train_origin = np.array(pd.read_csv('other_algorithm/static/beijing_train_zd_pr_1981_2010.csv'))[month-1::12, :]
        y_test_origin = np.array(pd.read_csv('other_algorithm/static/beijing_test_zd_pr_2011_2018.csv'))[month-1::12, :]
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
def draw_map(var, sta_order_list, res, area, num, month):
    for i in range(num-1):
        result = res[i]
        Region_ID = area
        res_list = algorithm.make_station_list(sta_order_list, result)
        # OutPicFile1 = "d:\\eof_" + var + "_" + area +"_"+str(i+1)+ ".png"  # 图片输出路径
        OutPicFile1 = "static/" + "other_algorithm_" + var + "_" + area + "_" + str(i + 1) + ".png"
        # print("!!!", root_path)
        LevelFile = 'src\\config\\LEV\\eof\\maplev_descisiontree.LEV'
        Region_Dict2 = algorithm.get_RegionID_by_XML('src\\config\\sky_region_config_utf8_2.xml', Region_ID)
        title = get_title(area, var, i, month)
        d_std_lib.DrawMapMain_XML_CFG(Region_Dict2, res_list, Levfile=LevelFile, \
                                      Title=title, imgfile=OutPicFile1,
                                      bShowImage=False, bDrawNumber=False, bDrawColorbar=True,
                                      format1='%1d')  # ,Title='')
        print('ok')


def get_img(knn_score, lr_score, ridge_score, svr_score):
    choosed = 0
    maxn = knn_score
    if lr_score > maxn:
        choosed = 1
        maxn = lr_score
    if ridge_score > maxn:
        choosed = 2
        maxn = ridge_score
    if svr_score > maxn:
        choosed = 3
        maxn = svr_score
    return choosed


def cal_ano(y2, var, year, method, month):
    if var == 0 and method == 0:
        df = np.array(pd.read_csv('other_algorithm/static/beijing_tmean_mean_1981_2010.csv')).squeeze()
        for i in range(12):
            for j in range(year-2010):
                y2[i + j * 12] -= df[i]
    if var == 0 and method == 1:
        df = np.array(pd.read_csv('other_algorithm/static/beijing_tmean_mean_1981_2010.csv')).squeeze()
        for j in range(year-2010):
            y2[j] -= df[month-1]
    if var == 1 and method == 0:
        df = np.array(pd.read_csv('other_algorithm/static/beijing_pr_mean_1981_2010.csv')).squeeze()
        for i in range(12):
            for j in range(year-2010):
                y2[i + j * 12] -= df[i]
    if var == 1 and method == 1:
        df = np.array(pd.read_csv('other_algorithm/static/beijing_pr_mean_1981_2010.csv')).squeeze()
        for j in range(year-2010):
            y2[j] -= df[month-1]
    return y2


def is_squeeze(nparray):
    if nparray.size != 1:
        nparray = nparray.squeeze()
    return nparray


def other_algorithm(request):
    ip = globalIp
    var, year, method, month = get_parameter(request)
    # 预测年份如果要可修改，把predict_year写进get，去掉get_parameter中predict_year有关注释即可
    predict_year = "2018"
    if is_illegal(var, year, method, month, predict_year):
        return HttpResponse(json.dumps({"code": 1, "msg": "参数不合符要求", "data": []}, ensure_ascii=False),
                            content_type="application/json")
    var, year, method, month, predict_year = strToint5(var, year, method, month, predict_year)
    X_train, X_test, y_train, y_test, y_train_origin, y_test_origin = get_data(var, year, method, month, predict_year, 0)

    X_train_origin = X_train
    ss = MinMaxScaler()
    X_train = ss.fit_transform(X_train)
    X_test = ss.fit_transform(X_test)
    y_train = ss.fit_transform(y_train)
    y_test = ss.fit_transform(y_test)

    if var == 0 and method == 0:
        model_knn = KNeighborsRegressor(n_neighbors=8).fit(X_train, y_train)
        model_lr = LinearRegression().fit(X_train, y_train)
        model_ridge = Ridge(alpha=0.13).fit(X_train, y_train)
        model_svr = SVR(C=10, gamma=0.3).fit(X_train, y_train.ravel())
    if var == 0 and method == 1:
        knn = KNeighborsRegressor()
        pipe = Pipeline([("scaler", MinMaxScaler()), ("knn", knn)])  # 标准语法
        tuned_parameters = {'knn__n_neighbors': range(1, 25)}
        grid = GridSearchCV(pipe, tuned_parameters, cv=5)
        grid = grid.fit(X_train_origin, y_train_origin)
        model_knn = KNeighborsRegressor(n_neighbors=grid.best_params_['knn__n_neighbors']).fit(X_train, y_train)
        model_lr = LinearRegression().fit(X_train, y_train)
        ridge = Ridge()
        pipe = Pipeline([("scaler", MinMaxScaler()), ("ridge", ridge)])  # 标准语法
        tuned_parameters = {'ridge__alpha': np.linspace(0.01, 1, 100)}
        grid = GridSearchCV(pipe, tuned_parameters, cv=5)
        grid = grid.fit(X_train_origin, y_train_origin)
        model_ridge = Ridge(alpha=grid.best_params_['ridge__alpha']).fit(X_train, y_train)
        svr = SVR()
        pipe = Pipeline([("scaler", MinMaxScaler()), ("svr", svr)])  # 标准语法
        tuned_parameters = {'svr__C': np.linspace(0.1, 10, 10), 'svr__gamma': np.linspace(0.1, 10, 10)}
        grid = GridSearchCV(pipe, tuned_parameters, cv=5)
        grid = grid.fit(X_train_origin, y_train_origin.ravel())
        model_svr = SVR(C=grid.best_params_['svr__C'], gamma=grid.best_params_['svr__gamma']).fit(X_train, y_train.ravel())
    if var == 1 and method == 0:
        model_knn = KNeighborsRegressor(n_neighbors=24).fit(X_train, y_train)
        model_lr = LinearRegression().fit(X_train, y_train)
        model_ridge = Ridge().fit(X_train, y_train)
        model_svr = SVR(C=5.5, gamma=0.3).fit(X_train, y_train.ravel())
    if var == 1 and method == 1:
        knn = KNeighborsRegressor()
        pipe = Pipeline([("scaler", MinMaxScaler()), ("knn", knn)])  # 标准语法
        tuned_parameters = {'knn__n_neighbors': range(1, 25)}
        grid = GridSearchCV(pipe, tuned_parameters, cv=5)
        grid = grid.fit(X_train_origin, y_train_origin)
        model_knn = KNeighborsRegressor(n_neighbors=grid.best_params_['knn__n_neighbors']).fit(X_train, y_train)
        model_lr = LinearRegression().fit(X_train, y_train)
        ridge = Ridge()
        pipe = Pipeline([("scaler", MinMaxScaler()), ("ridge", ridge)])  # 标准语法
        tuned_parameters = {'ridge__alpha': np.linspace(0.01, 1, 100)}
        grid = GridSearchCV(pipe, tuned_parameters, cv=5)
        grid = grid.fit(X_train_origin, y_train_origin)
        model_ridge = Ridge(alpha=grid.best_params_['ridge__alpha']).fit(X_train, y_train)
        svr = SVR()
        pipe = Pipeline([("scaler", MinMaxScaler()), ("svr", svr)])  # 标准语法
        tuned_parameters = {'svr__C': np.linspace(0.1, 10, 10), 'svr__gamma': np.linspace(0.1, 10, 10)}
        grid = GridSearchCV(pipe, tuned_parameters, cv=5)
        grid = grid.fit(X_train_origin, y_train_origin.ravel())
        model_svr = SVR(C=grid.best_params_['svr__C'], gamma=grid.best_params_['svr__gamma']).fit(X_train, y_train.ravel())

    knn_score = model_knn.score(X_test, y_test)
    lr_score = model_lr.score(X_test, y_test)
    ridge_score = model_ridge.score(X_test, y_test)
    svr_score = model_svr.score(X_test, y_test)

    predict_knn = model_knn.predict(X_test)
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
        for i in range((year - 2010) * 12):
            a = z + dateutil.relativedelta.relativedelta(months=i)
            x2.append(a)
    if method == 1:
        x1 = np.arange(1981, 2011)
        x2 = np.arange(2011, year + 1)

    # 绘图 原始数据用柱状图表示，预测数据用折线图表示
    y1 = y_train_origin.squeeze()  # 训练集数据
    y2 = is_squeeze(y_test_origin)  # 测试集数据
    y3 = is_squeeze(predict_knn)  # 预测数据
    y3 = ss.inverse_transform(y3.reshape(-1, 1))
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
    plt.savefig(currentDir + '/knn.png', dpi=100, bbox_inches='tight', transparent=True)

    predict_lr = model_lr.predict(X_test)
    plt.clf()  # 清理历史绘图
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    # 绘图 原始数据用柱状图表示，预测数据用折线图表示
    y1 = y_train_origin.squeeze()  # 训练集数据
    # y2 = y_test_origin.squeeze()  # 测试集数据
    y3 = is_squeeze(predict_lr)  # 预测数据
    y3 = ss.inverse_transform(y3.reshape(-1, 1))
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
    plt.savefig(currentDir + '/lr.png', dpi=100, bbox_inches='tight', transparent=True)

    predict_ridge = model_ridge.predict(X_test)
    plt.clf()  # 清理历史绘图
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    # 绘图 原始数据用柱状图表示，预测数据用折线图表示
    y1 = y_train_origin.squeeze()  # 训练集数据
    # y2 = y_test_origin.squeeze()  # 测试集数据
    y3 = is_squeeze(predict_ridge)  # 预测数据
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
    plt.savefig(currentDir + '/ridge.png', dpi=100, bbox_inches='tight', transparent=True)

    predict_svr = model_svr.predict(X_test)
    plt.clf()  # 清理历史绘图
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    # 绘图 原始数据用柱状图表示，预测数据用折线图表示
    y1 = y_train_origin.squeeze()  # 训练集数据
    # y2 = y_test_origin.squeeze()  # 测试集数据
    y3 = is_squeeze(predict_svr)  # 预测数据
    y3 = ss.inverse_transform(y3.reshape(-1, 1))
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
    plt.savefig(currentDir + '/svr.png', dpi=100, bbox_inches='tight', transparent=True)

    LIST = []
    data_dict = {}
    data_dict['name'] = 'k近邻'
    data_dict['score'] = np.around(knn_score, 2)
    data_dict['img'] = "http://" + ip + '/' + 'static/knn.png'
    LIST.append(data_dict)
    data_dict = {}
    data_dict['name'] = '线性回归'
    data_dict['score'] = np.around(lr_score, 2)
    data_dict['img'] = "http://" + ip + '/' + 'static/lr.png'
    LIST.append(data_dict)
    data_dict = {}
    data_dict['name'] = '岭回归'
    data_dict['score'] = np.around(ridge_score, 2)
    data_dict['img'] = "http://" + ip + '/' + 'static/ridge.png'
    LIST.append(data_dict)
    data_dict = {}
    data_dict['name'] = '支持向量机'
    data_dict['score'] = np.around(svr_score, 2)
    data_dict['img'] = "http://" + ip + '/' + 'static/svr.png'
    LIST.append(data_dict)
    data_dict = {}
    data_dict['train'] = LIST

    X_train, X_test, y_train, y_test, y_train_origin, y_test_origin = get_data(var, year, method, month, predict_year, 1)

    X_train_origin = X_train
    ss = MinMaxScaler()
    X_train = ss.fit_transform(X_train)
    X_test = ss.fit_transform(X_test)
    y_train = ss.fit_transform(y_train)
    y_test = ss.fit_transform(y_test)
    # MultiOutputRegressor()
    # 创建集合梯度提升树AdaBoost模型，弱学习器为决策树回归器
    if var == 0:
        model_knn = MultiOutputRegressor(KNeighborsRegressor(n_neighbors=8)).fit(X_train, y_train)
        model_lr = MultiOutputRegressor(LinearRegression()).fit(X_train, y_train)
        model_ridge = MultiOutputRegressor(Ridge(alpha=0.13)).fit(X_train, y_train)
        model_svr = MultiOutputRegressor(SVR(C=10, gamma=0.3)).fit(X_train, y_train)
    if var == 1:
        model_knn = MultiOutputRegressor(KNeighborsRegressor(n_neighbors=24)).fit(X_train, y_train)
        model_lr = MultiOutputRegressor(LinearRegression()).fit(X_train, y_train)
        model_ridge = MultiOutputRegressor(Ridge()).fit(X_train, y_train)
        model_svr = MultiOutputRegressor(SVR(C=5.5, gamma=0.3)).fit(X_train, y_train)

    ####3.8Bagging回归####
    # model_BaggingRegressor = ensemble.BaggingRegressor().fit(X, y.ravel())

    # CART决策树预测值
    predict_knn = model_knn.predict(X_test)
    # DecisionTreeRegressor_score = model_DecisionTreeRegressor.score(X_test, y_test)
    # print("CART决策树的决定系数是:", DecisionTreeRegressor_score)
    #
    predict_lr = model_lr.predict(X_test)
    # RandomForestRegressor_score = model_RandomForestRegressor.score(X_test, y_test)
    # print("随机森林的决定系数是:", RandomForestRegressor_score)
    #
    predict_ridge = model_ridge.predict(X_test)
    # predict_GradientBoostingRegressor = ss.inverse_transform(predict_GradientBoostingRegressor)
    # GradientBoostingRegressor_score = model_GradientBoostingRegressor.score(X_test, y_test)
    # print("梯度提升树的决定系数是:", GradientBoostingRegressor_score)
    #
    predict_svr = model_svr.predict(X_test)
    # model_AdaBoostRegressor_score = model_AdaBoostRegressor.score(X_test, y_test)
    # print("自适应增强树的决定系数是:", model_AdaBoostRegressor_score)

    var = str(var)  # 要素
    area = 'bj'  # 绘制区域 北京：bj，天津：tj，京津冀：jjj，内蒙古：nmg，华北：huabei，山西：shanxi
    num = 5  # 模态个数
    Month = month

    df = pd.read_csv('other_algorithm/static/beijing_test_zd_tmean_2011_2018.csv')
    sta_order_list = df.columns.values
    # print(df)
    # print(predict_DecisionTreeRegressor)

    predict_knn = predict_knn.squeeze()
    predict_knn = ss.inverse_transform(predict_knn)
    predict_lr = predict_lr.squeeze()
    predict_lr = ss.inverse_transform(predict_lr)
    predict_ridge = predict_ridge.squeeze()
    predict_ridge = ss.inverse_transform(predict_ridge)
    predict_svr = predict_svr.squeeze()
    predict_svr = ss.inverse_transform(predict_svr)

    # print(predict_DecisionTreeRegressor)
    all_year = predict_year - 2010
    # print(predict_DecisionTreeRegressor[all_year*12-1-(12-Month):all_year*12-(12-Month), :])
    df0 = np.array(pd.read_csv('other_algorithm/static/beijing_tmean_mean_1981_2010.csv')).squeeze()
    df1 = np.array(pd.read_csv('other_algorithm/static/beijing_pr_mean_1981_2010.csv')).squeeze()
    tmean = df0[month - 1]
    pr = df1[month - 1]
    DRAW = np.zeros(shape=(4, 20))
    method = int(method)
    var = int(var)
    if method == 0 and var == 0:
        # print("ZZ")
        DRAW[0] = np.array(pd.DataFrame(
            predict_knn[all_year * 12 - 1 - (12 - Month):all_year * 12 - (12 - Month),
            :]) - tmean).squeeze()
        DRAW[1] = np.array(pd.DataFrame(
            predict_lr[all_year * 12 - 1 - (12 - Month):all_year * 12 - (12 - Month),
            :]) - tmean).squeeze()
        DRAW[2] = np.array(pd.DataFrame(
            predict_ridge[all_year * 12 - 1 - (12 - Month):all_year * 12 - (12 - Month),
            :]) - tmean).squeeze()
        DRAW[3] = np.array(pd.DataFrame(
            predict_svr[all_year * 12 - 1 - (12 - Month):all_year * 12 - (12 - Month),
            :]) - tmean).squeeze()
    if method == 0 and var == 1:
        DRAW[0] = np.array(pd.DataFrame(
            predict_knn[all_year * 12 - 1 - (12 - Month):all_year * 12 - (12 - Month),
            :]) - pr).squeeze()
        DRAW[1] = np.array(pd.DataFrame(
            predict_lr[all_year * 12 - 1 - (12 - Month):all_year * 12 - (12 - Month),
            :]) - pr).squeeze()
        DRAW[2] = np.array(pd.DataFrame(
            predict_ridge[all_year * 12 - 1 - (12 - Month):all_year * 12 - (12 - Month),
            :]) - pr).squeeze()
        DRAW[3] = np.array(pd.DataFrame(
            predict_svr[all_year * 12 - 1 - (12 - Month):all_year * 12 - (12 - Month), :]) - pr).squeeze()
    if method == 1 and var == 0:
        DRAW[0] = np.array(pd.DataFrame(predict_knn[all_year - 1:all_year, :]) - tmean).squeeze()
        DRAW[1] = np.array(pd.DataFrame(predict_lr[all_year - 1:all_year, :]) - tmean).squeeze()
        DRAW[2] = np.array(pd.DataFrame(predict_ridge[all_year - 1:all_year, :]) - tmean).squeeze()
        DRAW[3] = np.array(pd.DataFrame(predict_svr[all_year - 1:all_year, :]) - tmean).squeeze()
    if method == 1 and var == 1:
        DRAW[0] = np.array(pd.DataFrame(predict_knn[all_year - 1:all_year, :]) - pr).squeeze()
        DRAW[1] = np.array(pd.DataFrame(predict_lr[all_year - 1:all_year, :]) - pr).squeeze()
        DRAW[2] = np.array(pd.DataFrame(predict_ridge[all_year - 1:all_year, :]) - pr).squeeze()
        DRAW[3] = np.array(pd.DataFrame(predict_svr[all_year - 1:all_year, :]) - pr).squeeze()
    DRAW = DRAW.reshape(-1, 20)
    # print(DRAW)
    # print(predict_AdaBoostRegressor)
    # print(np.array(pd.DataFrame(predict_DecisionTreeRegressor[all_year*12-1-(12-Month):all_year*12-(12-Month), :])-tmean).squeeze())
    var = str(var)
    draw_map(var, sta_order_list, DRAW, area, num, Month)

    LIST = []
    choosed = get_img(knn_score, lr_score,
                      ridge_score, svr_score)
    data_dict1 = {}
    var = int(var)

    if var == 0:
        if choosed == 0:
            data_dict1['name'] = 'k近邻'
            data_dict1['score'] = np.around(knn_score, 2)
            data_dict1['img'] = "http://" + ip + '/' + 'static/other_algorithm_0_bj_1.png'
        if choosed == 1:
            data_dict1['name'] = '线性回归'
            data_dict1['score'] = np.around(lr_score, 2)
            data_dict1['img'] = "http://" + ip + '/' + 'static/other_algorithm_0_bj_2.png'
        if choosed == 2:
            data_dict1['name'] = '岭回归'
            data_dict1['score'] = np.around(ridge_score, 2)
            data_dict1['img'] = "http://" + ip + '/' + 'static/other_algorithm_0_bj_3.png'
        if choosed == 3:
            data_dict1['name'] = '支持向量机'
            data_dict1['score'] = np.around(svr_score, 2)
            data_dict1['img'] = "http://" + ip + '/' + 'static/other_algorithm_0_bj_4.png'
        LIST.append(data_dict1)
        data_dict['predict'] = LIST

    if var == 1:
        if choosed == 0:
            data_dict1['name'] = 'k近邻'
            data_dict1['score'] = np.around(knn_score, 2)
            data_dict1['img'] = "http://" + ip + '/' + 'static/other_algorithm_1_bj_1.png'
        if choosed == 1:
            data_dict1['name'] = '线性回归'
            data_dict1['score'] = np.around(lr_score, 2)
            data_dict1['img'] = "http://" + ip + '/' + 'static/other_algorithm_1_bj_2.png'
        if choosed == 2:
            data_dict1['name'] = '岭回归'
            data_dict1['score'] = np.around(ridge_score, 2)
            data_dict1['img'] = "http://" + ip + '/' + 'static/other_algorithm_1_bj_3.png'
        if choosed == 3:
            data_dict1['name'] = '支持向量机'
            data_dict1['score'] = np.around(svr_score, 2)
            data_dict1['img'] = "http://" + ip + '/' + 'static/other_algorithm_1_bj_4.png'
        LIST.append(data_dict1)
        data_dict['predict'] = LIST

    LIST = []
    data_dict1 = {}
    data_dict1['name'] = 'k近邻'
    data_dict1['score'] = knn_score
    if var == 0:
        data_dict1['img'] = "http://" + ip + '/' + 'static/other_algorithm_0_bj_1.png'
    else:
        data_dict1['img'] = "http://" + ip + '/' + 'static/other_algorithm_1_bj_1.png'
    data_dict1['最佳模型名称'] = 'k近邻模型'
    data_dict1['模型框架'] = 'SKlearn框架'
    data_dict1['算法说明'] = 'k近邻模型通过找出一个样本的k个最近邻居，将这些邻居的某个（些）属性的平均值赋给该样本，就可以得到该样本对应属性的值。'
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
    data_dict1['name'] = '线性回归'
    data_dict1['score'] = lr_score
    if var == 0:
        data_dict1['img'] = "http://" + ip + '/' + 'static/other_algorithm_0_bj_2.png'
    else:
        data_dict1['img'] = "http://" + ip + '/' + 'static/other_algorithm_1_bj_2.png'
    data_dict1['最佳模型名称'] = '线性回归模型'
    data_dict1['模型框架'] = 'SKlearn框架'
    data_dict1['算法说明'] = '线性回归模型能够用一条直线较为精确地描述数据之间的关系。通用公式可以表达为：h(w)=w0+w1x1+w2x2+...=w^Tx,x1、x2...表示样本的不同属性。'
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
    data_dict1['name'] = '岭回归'
    data_dict1['score'] = ridge_score
    if var == 0:
        data_dict1['img'] = "http://" + ip + '/' + 'static/other_algorithm_0_bj_3.png'
    else:
        data_dict1['img'] = "http://" + ip + '/' + 'static/other_algorithm_1_bj_3.png'
    data_dict1['最佳模型名称'] = '岭回归模型'
    data_dict1['模型框架'] = 'SKlearn框架'
    data_dict1['算法说明'] = '岭回归是一种专用于线性数据分析的有偏估计回归方法，实质上是一种改良的最小二乘估计法，通过放弃最小二乘法的无偏性，以损失部分信息、降低精度为代价获得回归系数更为符合实际、更可靠的回归方法，对病态数据的拟合要强于最小二乘法。当数据集中存在共线性的时候，岭回归就会有用。'
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
    data_dict1['name'] = '支持向量机'
    data_dict1['score'] = svr_score
    if var == 0:
        data_dict1['img'] = "http://" + ip + '/' + 'static/other_algorithm_0_bj_4.png'
    else:
        data_dict1['img'] = "http://" + ip + '/' + 'static/other_algorithm_1_bj_4.png'
    data_dict1['最佳模型名称'] = '支持向量机模型'
    data_dict1['模型框架'] = 'SKlearn框架'
    data_dict1['算法说明'] = '支持向量机是一种二分类模型，其基本模型定义为特征空间上的间隔最大的线性分类器，即支持向量机的学习策略便是间隔最大化，最终可转化为一个凸二次规划问题的求解。'
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
    data_dict['judge'] = LIST

    return HttpResponse(json.dumps({"code": 0, "msg": "success", "data": data_dict}, ensure_ascii=False),
                        content_type="application/json")
