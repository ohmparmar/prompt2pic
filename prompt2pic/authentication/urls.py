from django.urls import path
from . import views
app_name = 'authentication'
urlpatterns = [
    path('signup/', views.signup, name='signup'),
    path('login/', views.login, name='login'),  # Add this line for login
    path('otp-verification/', views.otp_verification, name='otp_verification'),
    path('logout/', views.user_logout, name='logout'),
    path('forgot-password/', views.forgot_password, name='forgot_password'),
    path('reset-password/', views.reset_password, name='reset_password'),  # New URL

    # Add other authentication URLs here
]
