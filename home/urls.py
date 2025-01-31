from django.urls import path
from home import views
from django.urls import path, include

urlpatterns = [
    path('', views.index, name='home'),
    path('About', views.about, name='about'),
    path('Prediction', views.Prediction, name='Prediction')
    
]
