from django.urls import path
from . import views
app_name = 'authentication'
urlpatterns = [
    path('signup/', views.signup, name='signup'),
    path('login/', views.login, name='login'),  # Add this line for login
    path('otp-verification/', views.otp_verification, name='otp_verification'),
    path('logout/', views.user_logout, name='logout'),
    # Add other authentication URLs here
]
