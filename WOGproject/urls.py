"""WOGproject URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from DecisionTree import views as DecisionTreeViews
from TimeSeries import views as TimeSeriesViews
from SimilarYearSearch import views as SimilarYearSearchViews
from RecommendationSystem import views as RecommendationSystemViews
from checkTrain import views as checkTrainViews
from other_algorithm import views as other_algorithm_views
from django.conf.urls import url
from django.conf import settings
from django.conf.urls.static import static
from django.views.static import serve

urlpatterns = [
    path('admin/', admin.site.urls),
    url(r'^test', DecisionTreeViews.test),
    url(r'^DecisionTree', DecisionTreeViews.DecisionTree),
    url(r'^TimeSeries', TimeSeriesViews.TimeSeries),
    url(r'^SimilarYearSearch', SimilarYearSearchViews.SimilarYearSearch),
    url(r'^other_algorithm', other_algorithm_views.other_algorithm),
    url(r'^RecommendationSystem', RecommendationSystemViews.RecommendationSystem),
    url(r'^checkTrain', checkTrainViews.checkTrain),
    url(r'^static/(?P<path>.*)$', serve, {'document_root': settings.MEDIA_ROOT}),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
