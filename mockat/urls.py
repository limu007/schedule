from django.contrib import admin
from django.urls import include, path
from . import views 

urlpatterns = [
    path('', views.index, name='index'),
    path('<int:pk>/', views.observe, name='src'),
    path('schedule/<int:pk>/', views.schedule, name='sched'),
    path('plot/<int:pk>/', views.plot, name='schplot'),
]
