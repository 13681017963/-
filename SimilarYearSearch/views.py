from django.shortcuts import render
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
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.multioutput import MultiOutputRegressor
import dclimate.d_std_lib as d_std_lib
from mttkinter import mtTkinter as tk
from django.conf import settings
globalIp = '192.168.1.123:8998'


def get_data(method, month, predict_year):
    start = "198101"
    end = str(predict_year)+"12"
    ind_file = "M_Atm_Nc"
    ind_num = -999
    ind_path = "SimilarYearSearch/static/"
    if method == 0:
        t_year = algorithm.get_year_result(start, end)
        # 读取88项大气环流指数数据作为特征
        df = diag_service.getIndList(t_year, start, end, ind_file, int(ind_num), ind_path)  # 按年获取
    if method == 1:
        df = diag_service.getIndListMonthYearly(month, start, end, ind_file, ind_path)  # 按月获取
    odf = df = pd.DataFrame(df)
    df.fillna(method='ffill', inplace=True)  # 前值填充
    df = (df-df.min())/(df.max()-df.min())  # 归一化
    df.fillna(method='ffill', inplace=True)  # 前值填充
    df.fillna(method='bfill', inplace=True)  # 后值填充
    df = df.dropna(axis=1, how='any')
    pca = PCA(n_components=2)  # 做PCA，取累计贡献率大于等于90%的因子作为预测因子
    df = pca.fit_transform(df)
    print('original shape:', odf.shape)
    print('now shape:', df.shape)
    return df


def strToint3(a, b, c):
    a = int(a)
    b = int(b)
    c = int(c)
    return a, b, c


def is_illegal(method, month):
    if not (method.isdigit()) or not (month.isdigit()) or int(month) < 1 or int(month) > 12:
        return True
    return False


def get_parameter(request):
    method = request.GET.get("method")
    month = request.GET.get("month")
    return method, month


# k-means
def cal_km(df, predict_year):
    KM = KMeans(n_clusters=29, random_state=10)
    y_pred = KM.fit_predict(df)
    predicted = KM.predict(df)

    df = pd.DataFrame(df)
    L1 = df[0]
    L2 = df[1]

    plt.clf()  # 清理历史绘图
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.scatter(L1, L2, c=predicted)
    plt.tick_params(axis='x', colors='white')
    plt.tick_params(axis='y', colors='white')
    LIST = []
    for i in range(1981, predict_year+1):
        LIST.append(str(i)[2:])
    for i, txt in enumerate(LIST):
        plt.annotate(txt, (L1[i], L2[i]), color='white')
    currentDir = "static"  # 当前工作路径

    ax = plt.gca()  # 获取边框

    ax.spines['top'].set_color("white")

    ax.spines['bottom'].set_color("white")

    ax.spines['left'].set_color("white")

    ax.spines['right'].set_color("white")

    plt.savefig(currentDir + '/k-means.png', dpi=100, bbox_inches='tight', transparent=True)
    # print("模型最终得分是：", metrics.calinski_harabasz_score(df, y_pred))
    # tuned_parameters = {'n_clusters': range(2, 39, 1), 'algorithm': ['auto', 'full', 'elkan']}
    # grid = GridSearchCV(KMeans(random_state=10), tuned_parameters, cv=5, n_jobs=-1)
    # grid = grid.fit(df)
    # print(grid.best_estimator_)
    return np.around(metrics.calinski_harabasz_score(df, y_pred), 2)


# 欧氏距离
def cal_Ed(df, predict_year):
    mp = np.empty((predict_year-1981+1, predict_year-1981+1), dtype=float)
    list_year = []
    list_val = []
    for i in range(predict_year-1981+1):
        for j in range(predict_year-1981+1):
            if i == j:
                continue
            a, b, c, d = df[i, 0], df[i, 1], df[j, 0], df[j, 1]
            dis = pow(pow((a-c), 2) + pow((b-d), 2), 0.5)
            mp[i, j] = dis
        min_year = 0
        min_val = 999999
        for j in range(predict_year-1981+1):
            if i == j:
                continue
            if mp[i, j] < min_val:
                min_val = mp[i, j]
                min_year = j + 1981
        list_year.append(min_year)
        list_val.append(np.around(min_val, 6))
    return list_year, list_val


def cal_Ched(df, predict_year):
    mp = np.empty((predict_year-1981+1, predict_year-1981+1), dtype=float)
    list_year = []
    list_val = []
    for i in range(predict_year-1981+1):
        for j in range(predict_year-1981+1):
            if i == j:
                continue
            a, b, c, d = df[i, 0], df[i, 1], df[j, 0], df[j, 1]
            dis = max(abs(a-c), abs(b-d))
            mp[i, j] = dis
        min_year = 0
        min_val = 999999
        for j in range(predict_year-1981+1):
            if i == j:
                continue
            if mp[i, j] < min_val:
                min_val = mp[i, j]
                min_year = j + 1981
        list_year.append(min_year)
        list_val.append(np.around(min_val, 6))
    return list_year, list_val


def cal_Coss(df, predict_year):
    mp = np.empty((predict_year-1981+1, predict_year-1981+1), dtype=float)
    list_year = []
    list_val = []
    for i in range(predict_year-1981+1):
        for j in range(predict_year-1981+1):
            if i == j:
                continue
            a, b, c, d = df[i, 0], df[i, 1], df[j, 0], df[j, 1]
            dis = (a*c+b*d)/(pow(pow(a,2)+pow(b,2),0.5)*pow(pow(c,2)+pow(d,2),0.5))
            mp[i, j] = dis
        max_year = 0
        max_val = -1
        for j in range(predict_year-1981+1):
            if i == j:
                continue
            if mp[i, j] > max_val:
                max_val = mp[i, j]
                max_year = j + 1981
        list_year.append(max_year)
        list_val.append(np.around(max_val, 6))
    return list_year, list_val


def SimilarYearSearch(request):
    ip = globalIp
    # settings.TRAINING_SIMILARYEAR = 1
    method, month = get_parameter(request)
    predict_year = "2018"
    if is_illegal(method, month):
        settings.TRAINING_SIMILARYEAR = 0
        return HttpResponse(json.dumps({"code": 1, "msg": "参数不合符要求", "data": []}, ensure_ascii=False),
                            content_type="application/json")
    settings.TRAINING_SIMILARYEAR = 1
    method, month, predict_year = strToint3(method, month, predict_year)
    df = get_data(method, month, predict_year)
    km_score = cal_km(df, predict_year)
    list_year_Ed, list_val_Ed = cal_Ed(df, predict_year)
    list_year_Ched, list_val_Ched = cal_Ched(df, predict_year)
    list_year_Coss, list_val_Coss = cal_Coss(df, predict_year)

    LIST = []
    data_dict = {}
    data_dict['name'] = 'k-means聚类'
    data_dict['score'] = km_score
    data_dict['img'] = "http://" + ip + '/' + 'static/k-means.png'
    LIST.append(data_dict)
    data_dict = {}
    dic = []
    # for i in range(1981, predict_year+1):
    #     dic_temp = {i: list_year_Ed[i-1981]}
    #     dic.append(dic_temp)
    data_dict['name'] = '欧氏距离'
    # data_dict['最相似年份'] = dic
    # dic = []
    for i in range(1981, predict_year+1):
        # dic_temp = {i: list_val_Ed[i-1981]}
        dic_temp = {"year": i, "similar_year": list_year_Ed[i-1981], "value": list_val_Ed[i - 1981]}
        dic.append(dic_temp)
    data_dict['data'] = dic
    LIST.append(data_dict)

    data_dict = {}
    data_dict['name'] = '切比雪夫距离'
    dic = []
    for i in range(1981, predict_year+1):
        dic_temp = {"year": i, "similar_year": list_year_Ched[i-1981], "value": list_val_Ched[i - 1981]}
        dic.append(dic_temp)
    data_dict['data'] = dic
    # dic = []
    # for i in range(1981, predict_year+1):
    #     dic_temp = {i: list_val_Ched[i-1981]}
    #     dic.append(dic_temp)
    # data_dict['相似值'] = dic
    LIST.append(data_dict)

    data_dict = {}
    data_dict['name'] = '余弦相似度'
    dic = []
    for i in range(1981, predict_year+1):
        dic_temp = {"year": i, "similar_year": list_year_Coss[i-1981], "value": list_val_Coss[i - 1981]}
        dic.append(dic_temp)
    data_dict['data'] = dic
    # dic = []
    # for i in range(1981, predict_year+1):
    #     dic_temp = {i: list_val_Coss[i-1981]}
    #     dic.append(dic_temp)
    # data_dict['相似值'] = dic
    LIST.append(data_dict)
    settings.TRAINING_SIMILARYEAR = 0

    return HttpResponse(json.dumps({"code": 0, "msg": "success", "data": LIST}, ensure_ascii=False),
                        content_type="application/json")
