# Generated by Django 2.2.10 on 2021-02-02 05:10

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('RecommendationSystem', '0001_initial'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='recommendation_data',
            name='id',
        ),
        migrations.AlterField(
            model_name='recommendation_data',
            name='item',
            field=models.IntegerField(max_length=500, primary_key=True, serialize=False, verbose_name='条目'),
        ),
    ]
