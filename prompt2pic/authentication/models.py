from django.db import models
from django.contrib.auth.models import AbstractUser
from django.db import models

# # Create your models here.

# class User(models.Model):
#     username = models.CharField(max_length=150, unique=True)
#     email = models.EmailField(unique=True)
#     otp = models.CharField(max_length=6, null=True, blank=True)
#     otp_verified = models.BooleanField(default=False)
#     password = models.CharField(max_length=128)
#     created_at = models.DateTimeField(auto_now_add=True)
#     updated_at = models.DateTimeField(auto_now=True)

#     def __str__(self):
#         return self.username


class CustomUser(AbstractUser):
    otp = models.CharField(max_length=6, null=True, blank=True)
    otp_verified = models.BooleanField(default=False)
    is_paid = models.BooleanField(
        default=False
    )  # New field to indicate if the user is a paid user

    def __str__(self):
        return self.username
