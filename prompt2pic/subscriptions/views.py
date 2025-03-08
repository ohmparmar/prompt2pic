# Create your views here.
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from .models import Subscription, Transaction
from django.utils import timezone
from authentication.models import CustomUser
from django.views.decorators.csrf import csrf_exempt  # Add this import
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
import stripe
from django.utils import timezone
from datetime import timedelta
import json
from django.core.mail import send_mail
from django.template.loader import render_to_string
from django.conf import settings
from django.contrib import messages

@login_required
def subscription_dashboard(request):
    subscriptions = Subscription.objects.filter(user=request.user).order_by(
        "-start_date"
    )
    transactions = Transaction.objects.filter(user=request.user).order_by("-created_at")
    return render(
        request,
        "subscriptions/subscription_dashboard.html",
        {
            "username":request.user.username,
            "subscriptions": subscriptions,
            "transactions": transactions,
        },
    )




def plans_view(request):
    context = {
        "login": 0,
    }
    if request.user.is_authenticated:
        context["login"] = 1
    return render(
        request, "subscriptions/plans.html", context
    )  # Adjust the path as necessary


def create_stripe_payment_intent():
    # Create a payment intent with Stripe
    try:
        payment_intent = stripe.PaymentIntent.create(
            amount=1000,  # Amount in cents (e.g., $10.00)
            currency="usd",  # Currency code
            payment_method_types=["card"],  # Specify payment method types
        )
        return payment_intent
    except Exception as e:
        print(f"Error creating payment intent: {str(e)}")
        return None  # Handle error appropriately




@csrf_exempt  # Assuming this decorator is still in use
def create_payment_intent(request):
    if not request.user.is_authenticated:
        return JsonResponse({"error": "User not authenticated"}, status=401)

    plan = request.POST.get("plan")
    # Set up Stripe session (adjust parameters as per your setup)
    session = stripe.checkout.Session.create(
        payment_method_types=["card"],
        line_items=[{
            "price_data": {
                "currency": "usd",
                "product_data": {
                    "name": f"{plan} Plan",
                },
                "unit_amount": 1000,  # Example amount in cents, adjust accordingly
            },
            "quantity": 1,
        }],
        mode="payment",
        success_url=request.build_absolute_uri(reverse("subscriptions:payment_success")),
        cancel_url=request.build_absolute_uri(reverse("subscriptions:plans")),
        metadata={"plan": plan},
    )
    print(f"Created Stripe session with success_url: {session.success_url}")
    # Return the session URL instead of the session ID
    return JsonResponse({"url": session.url})


stripe.api_key = "sk_test_51QwSWdL05RboHoHGndmL3eTEDkrf9Rm2pQdkqyKr5CrCNcxtgaubBSo6MOGSYZy6JOjW1eYoSkTGG5qEnZ4NRCoR00elj8RQ8U"

@login_required
def payment_success(request):
    payment_intent_id = request.GET.get("payment_intent_id")
    plan = request.GET.get("plan", "unknown")
    amount = float(request.GET.get("amount", 0)) / 100

    if not payment_intent_id:
        return render(request, "subscriptions/payment_error.html", {"error": "No payment intent provided."})

    try:
        payment_intent = stripe.PaymentIntent.retrieve(payment_intent_id)
        if payment_intent.status == "succeeded":
            user = request.user
            transaction = Transaction.objects.create(
                user=user,
                amount_paid=amount,
                status=1,
                transaction_id=payment_intent_id,
            )

            if plan == "weekly":
                end_date = timezone.now() + timedelta(days=7)
            elif plan == "monthly":
                end_date = timezone.now() + timedelta(days=30)
            elif plan == "annual":
                end_date = timezone.now() + timedelta(days=365)
            else:
                end_date = timezone.now()

            Subscription.objects.create(
                user=user,
                transaction=transaction,
                plan_type=plan.capitalize(),
                end_date=end_date,
            )

            user.is_paid = True
            user.save()

            # Send success email
            subject = "Prompt2Pic - Payment Receipt"
            html_message = render_to_string('subscriptions/receipt_success.html', {
                'amount': f"{amount:.2f}",
                'plan_type': plan.capitalize(),
                'transaction_id': payment_intent_id,
                'user': user,
            })
            send_mail(
                subject,
                "Thank you for your payment! Here are your receipt details.",  # Plain text fallback
                settings.DEFAULT_FROM_EMAIL,
                [user.email],
                html_message=html_message,
                fail_silently=False,
            )
            messages.success(request, 'Payment successful!')

            return redirect("image_generation:dashboard")
        else:
            # Send failure email
            subject = "Prompt2Pic - Payment Failed"
            html_message = render_to_string('subscriptions/receipt_failed.html', {
                'transaction_id': payment_intent_id,
                'user': request.user,
                'reason': "Payment did not succeed.",
            })
            send_mail(
                subject,
                "We encountered an issue processing your payment.",  # Plain text fallback
                settings.DEFAULT_FROM_EMAIL,
                [request.user.email],
                html_message=html_message,
                fail_silently=False,
            )
            return render(request, "subscriptions/receipt_failed.html", {"message": "Payment did not succeed."})
    except stripe.error.StripeError as e:
        # Send failure email for Stripe error
        subject = "Prompt2Pic - Payment Failed"
        html_message = render_to_string('receipt_failed.html', {
            'transaction_id': payment_intent_id or "Unknown",
            'user': request.user,
            'reason': str(e),
        })
        send_mail(
            subject,
            "We encountered an issue processing your payment.",  # Plain text fallback
            settings.DEFAULT_FROM_EMAIL,
            [request.user.email],
            html_message=html_message,
            fail_silently=False,
        )
        return render(request, "subscriptions/payment_error.html", {"error": str(e)})
@csrf_exempt
def process_payment(request):
    if request.method != "POST":
        return JsonResponse({"error": "Invalid request method"}, status=405)

    if not request.user.is_authenticated:
        return JsonResponse({"error": "User not authenticated"}, status=401)

    try:
        data = json.loads(request.body)
        payment_method_id = data.get("payment_method_id")
        plan = data.get("plan")
        amount = data.get("amount")

        if not all([payment_method_id, plan, amount]):
            return JsonResponse({"error": "Missing required fields"}, status=400)

        # Create Payment Intent with only card payments allowed
        payment_intent = stripe.PaymentIntent.create(
            amount=amount,  # Amount in cents
            currency="usd",
            payment_method=payment_method_id,
            payment_method_types=['card'],  # Restrict to card payments only
            confirmation_method="manual",
            confirm=True,
            metadata={
                "plan": plan,
                "user_id": str(request.user.id),
            },
        )

        # Check if payment succeeded
        if payment_intent.status == "succeeded":
            return JsonResponse({"success": True, "payment_intent_id": payment_intent.id})
        else:
            return JsonResponse({"error": "Payment failed: " + payment_intent.last_payment_error.get("message", "Unknown error") if payment_intent.last_payment_error else "Unknown error"}, status=400)

    except stripe.error.CardError as e:
        return JsonResponse({"error": f"Card error: {e.user_message}"}, status=400)
    except stripe.error.StripeError as e:
        return JsonResponse({"error": f"Stripe error: {str(e)}"}, status=500)
    except Exception as e:
        return JsonResponse({"error": f"Unexpected error: {str(e)}"}, status=500)