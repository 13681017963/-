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
globalIp = '192.168.1.123:8998'


def checkTrain(request):
    ip = globalIp
    data_dict = {}
    data_dict['DecisionTree'] = settings.TRAINING_DECISIONTREE
    data_dict['TimeSeries'] = settings.TRAINING_TIMESERIES
    data_dict['SimilarYear'] = settings.TRAINING_SIMILARYEAR
    data_dict['RecommendationSystem'] = settings.TRAINING_RECOMMENDATIONSYSTEM
    if settings.TRAINING_DECISIONTREE == 1 or settings.TRAINING_TIMESERIES == 1 \
            or settings.TRAINING_SIMILARYEAR == 1 or settings.TRAINING_RECOMMENDATIONSYSTEM == 1:
        return HttpResponse(json.dumps({"code": 0, "msg": "训练中", "data": data_dict}, ensure_ascii=False),
                        content_type="application/json")
    else:
        return HttpResponse(json.dumps({"code": 1, "msg": "训练结束", "data": {"DecisionTree": 0, "TimeSeries": 0, "SimilarYear": 0, "RecommendationSystem": 0}}, ensure_ascii=False),
                        content_type="application/json")
