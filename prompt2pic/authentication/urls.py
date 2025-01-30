from django.urls import path
from . import views
app_name = 'authentication'
urlpatterns = [
    path('signup/', views.signup, name='signup'),
    path('otp-verification/', views.otp_verification, name='otp_verification')
    # Add other authentication URLs here
]
