from django.urls import path
from .views import generate_image, dashboard
app_name = 'image_generation'  

urlpatterns = [
    path('generate/', generate_image, name='generate_image'),
    path('dashboard/', dashboard, name='dashboard'),
]
