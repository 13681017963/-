from django.db import models

# Create your models here.
class recommendation_data(models.Model):
    data = models.CharField('推荐数据', max_length=1000000)
    item = models.IntegerField('条目', primary_key=True)

    class Meta:
        verbose_name = '推荐数据'
        verbose_name_plural = verbose_name

    def __str__(self):
        return str(self.item)
