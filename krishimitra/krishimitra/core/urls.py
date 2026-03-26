from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('features/', views.features, name='features'),
    path('about/', views.about, name='about'),
    path('contact/', views.contact, name='contact'),
    path('mandi/', views.mandi_prices, name='mandi_prices'),
    path('detect/', views.detect_disease, name='detect'),
]


