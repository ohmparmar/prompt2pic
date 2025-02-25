from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from .models import CustomUser

@admin.register(CustomUser)
class CustomUserAdmin(UserAdmin):
    list_display = ('username', 'email', 'otp_verified', 'is_staff', 'is_active')
    list_filter = ('otp_verified', 'is_staff', 'is_active')
    search_fields = ('username', 'email')
    fieldsets = UserAdmin.fieldsets + (
        ('OTP Information', {'fields': ('otp', 'otp_verified')}),
    )
