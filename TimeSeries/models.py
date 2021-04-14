from django.db import models

# 2018117 year+var+method+month
class timeseries_data(models.Model):
    info = models.CharField('参数信息', max_length=100)
    arima_score = models.FloatField('自回归分数')
    arima_img = models.CharField('自回归图片地址', max_length=100)
    mfarima_score = models.FloatField('平滑滤波+自回归分数')
    mfarima_img = models.CharField('平滑滤波+自回归图片地址', max_length=100)
    waarima_score = models.FloatField('小波分析+自回归分数')
    waarima_img = models.CharField('小波分析+自回归图片地址', max_length=100)
    predict_name = models.CharField('最优预测算法名称', max_length=100)
    predict_score = models.FloatField('最优预测算法分数')
    predict_img = models.CharField('最优预测算法图片地址', max_length=100)

    class Meta:
        verbose_name = '时间序列'
        verbose_name_plural = verbose_name

    def __str__(self):
        return str(self.info)