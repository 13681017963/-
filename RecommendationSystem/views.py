from django.shortcuts import render
from django.http import HttpResponse
import json
import datetime
import requests
from django.conf import settings
import numpy as np
from django.conf import settings
from RecommendationSystem import models
from RecommendationSystem.models import recommendation_data
from django.forms.models import model_to_dict

globalIp = '192.168.1.123:8998'
from apscheduler.schedulers.background import BackgroundScheduler
from django_apscheduler.jobstores import DjangoJobStore, register_job


def get_data(var, year, method, algorithm, ip, List):
    month = datetime.datetime.now().month
    month = str(month)
    urls = "http://" + ip + "/" + algorithm + "?" + "var=" + var + "&year=" + year + "&method=" + method + "&month=" + month
    dic = requests.get(urls).json()

    for i in dic['data']['judge']:
        data_dict = {}
        data_dict['name'] = i['name']
        data_dict['score'] = i['score']
        data_dict['img'] = i['img']
        data_dict['最佳模型名称'] = i['最佳模型名称']
        data_dict['模型框架'] = i['模型框架']
        data_dict['算法说明'] = i['算法说明']
        data_dict['数据预处理'] = i['数据预处理']
        data_dict['预测因子智能优选方法'] = i['预测因子智能优选方法']
        data_dict['数据集'] = i['数据集']
        data_dict['训练样本'] = i['训练样本']
        List.append(data_dict)
    return List


# 实例化调度器
scheduler = BackgroundScheduler()
# 调度器使用默认的DjangoJobStore()
#scheduler.add_jobstore(DjangoJobStore(), 'default')


# @register_job(scheduler, 'cron', id='test', day=1, hour=0, minute=0, args=[])
# 每个月1号0点0分执行这个任务
# @register_job(scheduler, 'interval', id='test', seconds=10, args=[])
# @register_job(scheduler, 'date', id='test', run_date='2021-04-02 9:33:00', args=["2018"])
@register_job(scheduler, 'cron', id='test', day=1, hour=0, minute=0, args=["2018"])
def test(predict_year):
    # 具体要执行的代码

    print("test")

    ip = globalIp
    global_data_dict = {}

    settings.TRAINING_RECOMMENDATIONSYSTEM = 1

    list_tmean = []
    list_tmean = get_data("0", predict_year, "0", "DecisionTree", ip, list_tmean)
    list_tmean = get_data("0", predict_year, "0", "TimeSeries", ip, list_tmean)
    list_tmean = get_data("0", predict_year, "0", "other_algorithm", ip, list_tmean)
    list_pr = []
    list_pr = get_data("1", predict_year, "0", "DecisionTree", ip, list_pr)
    list_pr = get_data("1", predict_year, "0", "TimeSeries", ip, list_pr)
    list_pr = get_data("1", predict_year, "0", "other_algorithm", ip, list_pr)

    list_tmean = sorted(list_tmean, key=lambda x: x['score'], reverse=True)
    list_pr = sorted(list_pr, key=lambda x: x['score'], reverse=True)
    global_data_dict['tmean'] = list_tmean
    global_data_dict['pr'] = list_pr
    print(global_data_dict)

    r = recommendation_data()
    r.data = global_data_dict
    print(r.data)
    r.item = 1
    r.save()

    settings.TRAINING_RECOMMENDATIONSYSTEM = 0
    print("Complete")
    return global_data_dict


# 注册定时任务并开始
scheduler.start()


def RecommendationSystem(request):
    ip = globalIp
    global_data_dict = recommendation_data.objects.filter(item=1)
    if global_data_dict.exists() == False:
        global_data_dict = {}
    else:
        global_data_dict = model_to_dict(global_data_dict[0])['data']
    return HttpResponse(json.dumps({"code": 0, "msg": "success", "data": global_data_dict}, ensure_ascii=False),
                        content_type="application/json")
