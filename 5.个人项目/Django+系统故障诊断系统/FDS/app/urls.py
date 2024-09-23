from django.urls import path
from . import views

app_name = 'app'

urlpatterns = [
    path('douploadTrainForm/',views.douploadTrainForm,name='douploadTrainForm'),
    path('douploadTestForm/',views.douploadTestForm,name='douploadTestForm'),
    path('homepage/',views.homepage,name='homepage'),
    path('downloadModel1/',views.downloadModel1,name='downloadModel1'),
    path('downloadModel2/',views.downloadModel2,name='downloadModel2'),
    path('downloadTestResult/',views.downloadTestResult,name='downloadTestResult'),
]