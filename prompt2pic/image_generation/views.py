from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.contrib.auth import logout
from django.contrib import messages
import os
import uuid
from django.http import JsonResponse
from django.shortcuts import render, redirect
from diffusers import StableDiffusionPipeline
import torch
from .models import Chat, Agent, ChatMessage

from subscriptions.models import Transaction, Subscription
from django.urls import reverse
from django.core.exceptions import ObjectDoesNotExist
import openai
from django.conf import settings
from django.core.files.base import ContentFile
from io import BytesIO
import requests
import re
from django.contrib.auth.decorators import login_required
from django.contrib.auth import update_session_auth_hash
import os
import json
from imagepig import ImagePig
from django.utils import timezone
from datetime import timedelta
import stripe
from django.views.decorators.csrf import csrf_exempt  # Add this import

imagepig = ImagePig(settings.IMAGEPIG_API_KEY)

client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)  # Explicitly passing API key

stripe.api_key = settings.STRIPE_SECRET_KEY  # Ensure you have your secret key set


@login_required
def dashboard(request):

    try:
        chat_id = request.GET.get("chat_id")
        active_chat = Chat.objects.get(id=chat_id, user=request.user)

    except (ObjectDoesNotExist, ValueError):
        active_chat = Chat.objects.filter(user=request.user).first()

    # Check if user has an active subscription for paid models
    has_active_subscription = Subscription.objects.filter(
        user=request.user,
        end_date__gte=timezone.now()
    ).exists()
    if has_active_subscription:
        has_subscribed = True  # All available models
    else:
        has_subscribed = False  # Only free models

    context = {
        "history": Chat.objects.filter(user=request.user).order_by("-created_at"),
        "models": Agent.objects.filter(is_available=True),
        "active_chat": active_chat,
        "error": messages.get_messages(request),
        "prompt_input": request.POST.get("prompt", ""),
        "selected_model": Agent.objects.filter(id=request.POST.get("model")).first(),
        "has_subscribed": has_subscribed
    }

    return render(request, "image_generation/dashboard.html", context)
def guest_dashboard(request):
    generated_image_url = None

    if request.method == "POST":
        prompt = request.POST.get("prompt", "")

        # Show loader during generation
        MODEL_ID = "CompVis/stable-diffusion-v1-4"
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load model (consider caching this in production)
        pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_ID, torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )
        pipe = pipe.to(device)

        # Generate image
        result = pipe(prompt)
        generated_image = result.images[0]

        # Save image to BytesIO
        img_io = BytesIO()
        generated_image.save(img_io, format="PNG")
        img_io.seek(0)  # Move back to the start of the stream

        # Generate a unique filename
        filename = f"{uuid.uuid4()}.png"

        # Construct the path for saving the file in media/generated_images
        save_dir = os.path.join(settings.MEDIA_ROOT, "generated_images")
        os.makedirs(save_dir, exist_ok=True)  # Ensure directory exists
        file_path = os.path.join(save_dir, filename)

        # Write file to disk
        with open(file_path, "wb") as f:
            f.write(img_io.getvalue())

        # Construct the URL to access this image
        generated_image_url = os.path.join(
            settings.MEDIA_URL, "generated_images", filename
        )

    return render(
        request,
        "image_generation/guest_user_dashboard.html",
        {"generated_image": generated_image_url},
    )


# Initialize your model once when the server starts (optional but recommended)
# You can do this in a module-level variable so that it's not reloaded on every request.
MODEL_ID = "CompVis/stable-diffusion-v1-4"  # or any other compatible model
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_ID, torch_dtype=torch.float16 if device == "cuda" else torch.float32
)
pipe = pipe.to(device)
# Load the model once (e.g., at server startup, not in the view itself)
# pipe = StableDiffusionXLPipeline.from_pretrained(
#     "stabilityai/stable-diffusion-xl-base-1.0",
#     torch_dtype=torch.float16
# )
# pipe = pipe.to("cuda")  # M


@login_required
def generate_image(request):
    if request.method == "POST":
        try:
            prompt = request.POST.get("prompt", "").strip()
            model_id = request.POST.get("model")
            chat_id = request.POST.get("chat_id")
            if not prompt:
                messages.error(request, "Please enter a prompt")
                return redirect("image_generation:dashboard")

            if not model_id:
                messages.error(request, "Please select a model")
                return redirect("image_generation:dashboard")

            # Get selected model
            try:
                model = Agent.objects.get(id=model_id, is_available=True)
            except ObjectDoesNotExist:
                messages.error(request, "Selected model is not available")
                return redirect("image_generation:dashboard")
            if chat_id:
                try:
                    active_chat = Chat.objects.get(id=chat_id, user=request.user)
                except Chat.DoesNotExist:
                    active_chat = Chat.objects.create(
                        user=request.user,
                        title=prompt[:50] + ("..." if len(prompt) > 50 else ""),
                    )
            # Generate image using the selected model
            generated_image = None
            print("model.name.lower()", model.name.lower())

            if model.name.lower() == "chat-gpt":
                print("inside chat-gpt")
                response = client.images.generate(
                    model="dall-e-2",  # or "dall-e-3"
                    prompt=prompt,
                    n=1,
                    size="1024x1024",
                )

                image_url = response.data[0].url
                print("image_url", image_url)
                response_content = requests.get(image_url).content
                generated_image = ContentFile(
                    response_content, name=f"{uuid.uuid4()}.png"
                )

            elif model.name.lower() == "stability-ai":
                print("Using Stability AI API for image generation")
                url = "https://api.stability.ai/v2beta/stable-image/generate/ultra"

                headers = {
                    "Authorization": f"Bearer {settings.STABILITYAI_API_KEY}",
                    "Accept": "image/*",
                }

                files = {
                    "prompt": (None, prompt),
                    "output_format": (None, "png"),
                    "width": (None, "1024"),
                    "height": (None, "1024"),
                    "samples": (None, "1"),
                }

                response = requests.post(url, headers=headers, files=files)

                if response.status_code == 200:
                    generated_image = ContentFile(
                        response.content, name=f"{uuid.uuid4()}.png"
                    )
                else:
                    error_message = response.json().get("message", "Unknown error")
                    messages.error(request, f"Stability AI Error: {error_message}")
                    return redirect("image_generation:dashboard")

            elif model.name.lower() == "image-pig":
                print("Using ImagePig API")
                result = imagepig.default(prompt)

                # Define media folder path
                media_folder = "generated_images"
                media_path = os.path.join(settings.MEDIA_ROOT, media_folder)
                # Generate a unique filename
                filename = f"{uuid.uuid4()}.png"
                image_path = os.path.join(media_path, filename)

                # Save image to media folder
                result.save(image_path)

                # Convert saved file into a Django `ContentFile`
                with open(image_path, "rb") as img_file:
                    generated_image = ContentFile(img_file.read(), name=filename)
            elif model.name.lower() == "ai-girl":
                print("Using ai-girl.site API for image generation")
                url = "https://ai-girl.site/api/workerai"
                payload = json.dumps({"prompt": prompt})  # Convert dict to JSON string
                headers = {
                    "accept": "*/*",
                    "accept-language": "en-GB,en;q=0.9",
                    "content-type": "text/plain;charset=UTF-8",
                    "origin": "https://ai-girl.site",
                    "priority": "u=1, i",
                    "referer": "https://ai-girl.site/",
                    "sec-ch-ua": '"Not(A:Brand";v="99", "Google Chrome";v="133", "Chromium";v="133"',
                    "sec-ch-ua-mobile": "?0",
                    "sec-ch-ua-platform": '"Windows"',
                    "sec-fetch-dest": "empty",
                    "sec-fetch-mode": "cors",
                    "sec-fetch-site": "same-origin",
                    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36",
                }
                response = requests.post(url, headers=headers, data=payload)

                if response.status_code == 200:
                    # Assuming the response is image binary data
                    generated_image = ContentFile(
                        response.content, name=f"{uuid.uuid4()}.png"
                    )
                else:
                    error_message = response.text or "Unknown error from ai-girl API"
                    messages.error(request, f"ai-girl API Error: {error_message}")
                    return redirect("image_generation:dashboard")
            elif model.name.lower() == "hugging-face":
                print("Using Hugging Face Inference API")
                API_URL = "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5"
                headers = {"Authorization": f"Bearer {settings.HUGGING_FACE_API_KEY}"}
                response = requests.post(
                    API_URL, headers=headers, json={"inputs": prompt}
                )
                print(response)
                print(response.content)
                if response.status_code == 200:
                    generated_image = ContentFile(
                        response.content, name=f"{uuid.uuid4()}.png"
                    )
            else:
                result = pipe(prompt)
                generated_image = result.images[0]
                img_io = BytesIO()
                generated_image.save(img_io, format="PNG")
                generated_image = ContentFile(
                    img_io.getvalue(), name=f"{uuid.uuid4()}.png"
                )
                # print("Using Stable Diffusion XL locally")
                # result = pipe(prompt)  # Generate the image
                # generated_image = result.images[0]  # Get the first image
                # img_io = BytesIO()
                # generated_image.save(img_io, format="PNG")
                # generated_image = ContentFile(
                #     img_io.getvalue(), name=f"{uuid.uuid4()}.png"
                # )

            # Store image in chat history

            chat_message = ChatMessage.objects.create(
                chat=active_chat,
                user_prompt=prompt,
                agent=model,
                image_generated=generated_image,
            )

            return redirect(
                f'{reverse("image_generation:dashboard")}?chat_id={active_chat.id}'
            )

        except Exception as e:
            print(str(e))
            messages.error(request, f"Error generating image: {str(e)}")
            return redirect("image_generation:dashboard")

    return redirect("image_generation:dashboard")


# views.py
from django.db import transaction


@login_required
def create_chat(request):
    if not request.user.is_authenticated:
        return redirect("authentication:login")

    with transaction.atomic():
        # Create new chat with default title
        new_chat = Chat.objects.create(user=request.user, title="New Chat")

    return redirect(f'{reverse("image_generation:dashboard")}?chat_id={new_chat.id}')


@login_required
def delete_chat(request, chat_id):
    if not request.user.is_authenticated:
        return redirect("authentication:login")

    try:
        chat = Chat.objects.get(id=chat_id, user=request.user)
        chat.delete()
        messages.success(request, "Chat deleted successfully")
    except ObjectDoesNotExist:
        messages.error(request, "Chat not found")

    return redirect("image_generation:dashboard")


@login_required
def rename_chat(request, chat_id):
    if not request.user.is_authenticated:
        return JsonResponse({"status": "error", "message": "Unauthorized"}, status=401)

    if request.method == "POST":
        try:
            chat = Chat.objects.get(id=chat_id, user=request.user)
            new_title = request.POST.get("title", "")[:50]  # Get title or empty string
            chat.title = new_title or f"Chat {chat.id}"  # Set default if empty
            chat.save()
            return redirect("image_generation:dashboard")
        except Chat.DoesNotExist:
            return JsonResponse(
                {"status": "error", "message": "Chat not found"}, status=404
            )
        except Exception as e:
            return JsonResponse({"status": "error", "message": str(e)}, status=400)

    return JsonResponse({"status": "error", "message": "Invalid method"}, status=400)


PASSWORD_REGEX = re.compile(
    r'^(?=.*[A-Z])(?=.*\d)(?=.*[!@#$%^&*()_+\-=[\]{};\'":\\|,.<>/?]).{6,}$'
)


@login_required
def change_password(request):
    if request.method == "POST":
        new_password = request.POST.get("new_password", "")
        confirm_password = request.POST.get("confirm_password", "")

        if not PASSWORD_REGEX.match(new_password):
            error = "Password must have one capital letter, one number, one special character, and be at least 6 characters long."
            return render(
                request, "image_generation/change_password.html", {"error": error}
            )

        if new_password != confirm_password:
            error = "Passwords do not match."
            return render(
                request, "image_generation/change_password.html", {"error": error}
            )

        # Update password and keep user logged in
        user = request.user
        user.set_password(new_password)
        user.save()
        update_session_auth_hash(request, user)
        messages.success(request, "Password changed successfully.")
        return redirect("image_generation:dashboard")

    return render(request, "image_generation/change_password.html")


# def plans_view(request):
#     context = {
#         "login": 0,
#     }
#     if request.user.is_authenticated:
#         context["login"] = 1
#     return render(
#         request, "image_generation/plans.html", context
#     )  # Adjust the path as necessary


# def create_stripe_payment_intent():
#     # Create a payment intent with Stripe
#     try:
#         payment_intent = stripe.PaymentIntent.create(
#             amount=1000,  # Amount in cents (e.g., $10.00)
#             currency="usd",  # Currency code
#             payment_method_types=["card"],  # Specify payment method types
#         )
#         return payment_intent
#     except Exception as e:
#         print(f"Error creating payment intent: {str(e)}")
#         return None  # Handle error appropriately




# @csrf_exempt  # Assuming this decorator is still in use
# def create_payment_intent(request):
#     if not request.user.is_authenticated:
#         return JsonResponse({"error": "User not authenticated"}, status=401)

#     plan = request.POST.get("plan")
#     # Set up Stripe session (adjust parameters as per your setup)
#     session = stripe.checkout.Session.create(
#         payment_method_types=["card"],
#         line_items=[{
#             "price_data": {
#                 "currency": "usd",
#                 "product_data": {
#                     "name": f"{plan} Plan",
#                 },
#                 "unit_amount": 1000,  # Example amount in cents, adjust accordingly
#             },
#             "quantity": 1,
#         }],
#         mode="payment",
#         success_url=request.build_absolute_uri(reverse("image_generation:payment_success")),
#         cancel_url=request.build_absolute_uri(reverse("image_generation:plans")),
#         metadata={"plan": plan},
#     )
#     print(f"Created Stripe session with success_url: {session.success_url}")
#     # Return the session URL instead of the session ID
#     return JsonResponse({"url": session.url})

# @login_required
# def payment_success(request):
#     payment_intent_id = request.GET.get("payment_intent_id")
#     plan = request.GET.get("plan", "unknown")
#     amount = float(request.GET.get("amount", 0)) / 100

#     if not payment_intent_id:
#         return render(request, "image_generation/payment_error.html", {"error": "No payment intent provided."})

#     try:
#         payment_intent = stripe.PaymentIntent.retrieve(payment_intent_id)
#         if payment_intent.status == "succeeded":
#             user = request.user
#             transaction = Transaction.objects.create(
#                 user=user,
#                 amount_paid=amount,
#                 transaction_id=payment_intent_id,
#             )

#             if plan == "weekly":
#                 end_date = timezone.now() + timedelta(days=7)
#             elif plan == "monthly":
#                 end_date = timezone.now() + timedelta(days=30)
#             elif plan == "annual":
#                 end_date = timezone.now() + timedelta(days=365)
#             else:
#                 end_date = timezone.now()

#             Subscription.objects.create(
#                 user=user,
#                 transaction=transaction,
#                 plan_type=plan,
#                 end_date=end_date,
#             )

#             user.is_paid = True
#             user.save()
#             return redirect("image_generation:dashboard")
#         else:
#             return render(request, "image_generation/payment_failed.html", {"message": "Payment did not succeed."})
#     except stripe.error.StripeError as e:
#         return render(request, "image_generation/payment_error.html", {"error": str(e)})
# @csrf_exempt
# def process_payment(request):
#     if request.method != "POST":
#         return JsonResponse({"error": "Invalid request method"}, status=405)

#     if not request.user.is_authenticated:
#         return JsonResponse({"error": "User not authenticated"}, status=401)

#     try:
#         data = json.loads(request.body)
#         payment_method_id = data.get("payment_method_id")
#         plan = data.get("plan")
#         amount = data.get("amount")

#         if not all([payment_method_id, plan, amount]):
#             return JsonResponse({"error": "Missing required fields"}, status=400)

#         # Create Payment Intent with only card payments allowed
#         payment_intent = stripe.PaymentIntent.create(
#             amount=amount,  # Amount in cents
#             currency="usd",
#             payment_method=payment_method_id,
#             payment_method_types=['card'],  # Restrict to card payments only
#             confirmation_method="manual",
#             confirm=True,
#             metadata={
#                 "plan": plan,
#                 "user_id": str(request.user.id),
#             },
#         )

#         # Check if payment succeeded
#         if payment_intent.status == "succeeded":
#             return JsonResponse({"success": True, "payment_intent_id": payment_intent.id})
#         else:
#             return JsonResponse({"error": "Payment failed: " + payment_intent.last_payment_error.get("message", "Unknown error") if payment_intent.last_payment_error else "Unknown error"}, status=400)

#     except stripe.error.CardError as e:
#         return JsonResponse({"error": f"Card error: {e.user_message}"}, status=400)
#     except stripe.error.StripeError as e:
#         return JsonResponse({"error": f"Stripe error: {str(e)}"}, status=500)
#     except Exception as e:
#         return JsonResponse({"error": f"Unexpected error: {str(e)}"}, status=500)