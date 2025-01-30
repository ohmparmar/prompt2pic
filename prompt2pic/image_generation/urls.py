from django.urls import path
from . import views
app_name = 'image_generation'  

urlpatterns = [
    path('generate/', views.generate_image, name='generate_image'),
    path('dashboard/', views.dashboard, name='dashboard')]
