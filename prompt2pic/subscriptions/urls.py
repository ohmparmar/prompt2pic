from django.urls import path
from . import views

app_name = "subscriptions"

urlpatterns = [
    path("plans/", views.plans_view, name="plans"),
    path(
        "create-payment-intent/",
        views.create_payment_intent,
        name="create_payment_intent",
    ),
    path("payment-success/", views.payment_success, name="payment_success"),
    path("process-payment/", views.process_payment, name="process_payment"),
    path(
        "dashboard/", views.subscription_dashboard, name="subscription_dashboard"
    ),  # New endpoint
]
