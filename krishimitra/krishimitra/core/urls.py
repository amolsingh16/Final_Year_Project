from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('features/', views.features, name='features'),
    path('about/', views.about, name='about'),
    path('contact/', views.contact, name='contact'),
    path('mandi/', views.mandi_prices, name='mandi_prices'),
    path('detect/', views.detect_disease, name='detect'),
    path('crop-recommendation/', views.crop_recommendation, name='crop_recommendation'),
    path('weather/', views.weather_advisory, name='weather'),
    path('schemes/', views.government_schemes, name='schemes'),
    path('voice-assistant/', views.voice_assistant, name='voice_assistant'),
    path('api/voice-assistant/', views.voice_assistant_api, name='voice_assistant_api'),
]
