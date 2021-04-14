from django.db import models

# 2018117 year+var+method+month
class decisiontree_data(models.Model):
    info = models.CharField('参数信息', max_length=100)
    dt_score = models.FloatField('二元决策树分数')
    dt_img = models.CharField('二元决策树图片地址', max_length=100)
    rf_score = models.FloatField('随机森林分数')
    rf_img = models.CharField('随机森林图片地址', max_length=100)
    gb_score = models.FloatField('梯度提升树分数')
    gb_img = models.CharField('梯度提升树图片地址', max_length=100)
    ada_score = models.FloatField('自适应增强树分数')
    ada_img = models.CharField('自适应增强树图片地址', max_length=100)
    predict_name = models.CharField('最优预测算法名称', max_length=100)
    predict_score = models.FloatField('最优预测算法分数')
    predict_img = models.CharField('最优预测算法图片地址', max_length=100)

    class Meta:
        verbose_name = '决策树'
        verbose_name_plural = verbose_name

    def __str__(self):
        return str(self.info)