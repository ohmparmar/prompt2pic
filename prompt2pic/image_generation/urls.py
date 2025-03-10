from django.urls import path
from . import views

app_name = "image_generation"

urlpatterns = [
    path("generate/", views.generate_image, name="generate_image"),
    path("dashboard/", views.dashboard, name="dashboard"),
    path("guest-dashboard/", views.guest_dashboard, name="guest_dashboard"),
    path("create-chat/", views.create_chat, name="create_chat"),
    path("delete-chat/<int:chat_id>/", views.delete_chat, name="delete_chat"),
    path("rename-chat/<int:chat_id>/", views.rename_chat, name="rename_chat"),
    path("change-password/", views.change_password, name="change_password"),
    path("dashboard/<int:chat_id>/", views.dashboard, name="dashboard_with_chat"),  # Chat selected
    # path("plans/", views.plans_view, name="plans"),  # Add this line
    # path(
    #     "create-payment-intent/",
    #     views.create_payment_intent,
    #     name="create_payment_intent",
    # ),  # Add this line
    # path("payment-success/", views.payment_success, name="payment_success"),
    # path(
    #     "process-payment/", views.process_payment, name="process_payment"
    # ),  # New endpoint
]
