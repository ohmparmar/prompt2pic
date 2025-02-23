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
from django.urls import reverse
from django.core.exceptions import ObjectDoesNotExist
import openai
from django.conf import settings
from django.core.files.base import ContentFile
from io import BytesIO
import requests
import json
import re
from django.contrib.auth.decorators import login_required
from django.contrib.auth import update_session_auth_hash

client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)  # Explicitly passing API key


def dashboard(request):
    if not request.user.is_authenticated:
        return redirect("authentication:login")

    try:
        chat_id = request.GET.get("chat_id")
        active_chat = Chat.objects.get(id=chat_id, user=request.user)

    except (ObjectDoesNotExist, ValueError):
        active_chat = Chat.objects.filter(user=request.user).first()

    context = {
        "history": Chat.objects.filter(user=request.user).order_by("-created_at"),
        "models": Agent.objects.filter(is_available=True),
        "active_chat": active_chat,
        "error": messages.get_messages(request),
        "prompt_input": request.POST.get("prompt", ""),
        "selected_model": Agent.objects.filter(id=request.POST.get("model")).first(),
    }

    return render(request, "image_generation/dashboard.html", context)


# def guest_dashboard(request):
#     # if request.user.is_authenticated:
#     #     return render(request, 'image_generation/dashboard.html')
#     return render(request, "image_generation/guest_user_dashboard.html")


import os
import uuid
from io import BytesIO
from django.conf import settings
from django.core.files.base import ContentFile
from django.shortcuts import render
import torch
from diffusers import StableDiffusionPipeline


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
# You can do this in a module-level variable so that it’s not reloaded on every request.
MODEL_ID = "CompVis/stable-diffusion-v1-4"  # or any other compatible model
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_ID, torch_dtype=torch.float16 if device == "cuda" else torch.float32
)
pipe = pipe.to(device)

# def generate_image(request):
#     if not request.user.is_authenticated:
#         return redirect('authentication:login')

#     if request.method == 'POST':
#         try:
#             prompt = request.POST.get('prompt', '').strip()
#             model_id = request.POST.get('model')

#             # Validate inputs
#             if not prompt:
#                 messages.error(request, 'Please enter a prompt')
#                 return redirect('image_generation:dashboard')

#             if not model_id:
#                 messages.error(request, 'Please select a model')
#                 return redirect('image_generation:dashboard')

#             # Check for bad words
#             bad_words = ['nsfw', 'violent', 'explicit']  # Customize as needed
#             if any(word in prompt.lower() for word in bad_words):
#                 messages.error(request, 'Prompt contains inappropriate content')
#                 return redirect('image_generation:dashboard')

#             # Get selected model
#             try:
#                 model = Agent.objects.get(id=model_id, is_available=True)
#             except ObjectDoesNotExist:
#                 messages.error(request, 'Selected model is not available')
#                 return redirect('image_generation:dashboard')

#             # Create or get active chat and generate image inside a transaction
#             with transaction.atomic():
#                 active_chat = Chat.objects.filter(user=request.user).first()
#                 if not active_chat:
#                     active_chat = Chat.objects.create(
#                         user=request.user,
#                         title=prompt[:50] + ('...' if len(prompt) > 50 else '')
#                     )

#                 # Generate the image using Stable Diffusion
#                 try:
#                     result = pipe(prompt)
#                     generated_image = result.images[0]
#                 except Exception as e:
#                     messages.error(request, f'Image generation failed: {str(e)}')
#                     return redirect('image_generation:dashboard')

#                 # Save the generated image to the media directory
#                 filename = f"{uuid.uuid4()}.png"
#                 media_dir = os.path.join("media", "generated_images")
#                 os.makedirs(media_dir, exist_ok=True)
#                 image_path = os.path.join(media_dir, filename)
#                 generated_image.save(image_path)

#                 # Optionally, store a relative path (or use Django's FileField handling)
#                 relative_image_path = os.path.join('generated_images', filename)

#                 # Create chat message with the image path
#                 ChatMessage.objects.create(
#                     chat=active_chat,
#                     user_prompt=prompt,
#                     agent=model,
#                     image_generated=relative_image_path  # Store the relative path
#                 )

#             return redirect(f'{reverse("image_generation:dashboard")}?chat_id={active_chat.id}')

#         except Exception as e:
#             messages.error(request, f'Error generating image: {str(e)}')
#             return redirect('image_generation:dashboard')

#     return redirect('image_generation:dashboard')


# def generate_image(request):
#     if not request.user.is_authenticated:
#         return redirect('authentication:login')

#     if request.method == 'POST':
#         try:
#             prompt = request.POST.get('prompt', '').strip()
#             model_id = request.POST.get('model')

#             if not prompt:
#                 messages.error(request, 'Please enter a prompt')
#                 return redirect('image_generation:dashboard')

#             if not model_id:
#                 messages.error(request, 'Please select a model')
#                 return redirect('image_generation:dashboard')

#             # Get selected model
#             try:
#                 model = Agent.objects.get(id=model_id, is_available=True)
#             except ObjectDoesNotExist:
#                 messages.error(request, 'Selected model is not available')
#                 return redirect('image_generation:dashboard')

#             # Generate image using the selected model
#             image_url = None
#             if model.name.lower() == "chat-gpt":
#                 # Call OpenAI's API for image generation
#                 response = openai.Image.create(
#                     prompt=prompt,
#                     n=1,
#                     size="1024x1024"
#                 )
#                 image_url = response["data"][0]["url"]

#             else:
#                 # Use Stable Diffusion
#                 result = pipe(prompt)
#                 generated_image = result.images[0]

#                 filename = f"{uuid.uuid4()}.png"
#                 media_dir = os.path.join(settings.BASE_DIR,  'image_generation','static', 'image_generation','images')
#                 os.makedirs(media_dir, exist_ok=True)
#                 image_path = os.path.join(media_dir, filename)
#                 generated_image.save(image_path)

#                 image_url = f'/static/image_generation/images/{filename}'

#             # Store image in chat history
#             active_chat = Chat.objects.filter(user=request.user).first()
#             if not active_chat:
#                 active_chat = Chat.objects.create(
#                     user=request.user,
#                     title=prompt[:50] + ('...' if len(prompt) > 50 else '')
#                 )

#             ChatMessage.objects.create(
#                 chat=active_chat,
#                 user_prompt=prompt,
#                 agent=model,
#                 image_generated=image_url
#             )

#             return redirect(f'{reverse("image_generation:dashboard")}?chat_id={active_chat.id}')

#         except Exception as e:
#             messages.error(request, f'Error generating image: {str(e)}')
#             return redirect('image_generation:dashboard')

#     return redirect('image_generation:dashboard')


@login_required
def generate_image(request):
    if request.method == "POST":
        try:
            prompt = request.POST.get("prompt", "").strip()
            model_id = request.POST.get("model")

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
                print("response_content", response_content)
                generated_image = ContentFile(
                    response_content, name=f"{uuid.uuid4()}.png"
                )
            elif model.name.lower() == "stability-ai":
                print(
                    "Using Stability AI API for image generation",
                    settings.STABILITYAI_API_KEY,
                )
                url = "https://api.stability.ai/v2beta/stable-image/generate/ultra"

                headers = {
                    "Authorization": f"Bearer {settings.STABILITYAI_API_KEY}",
                    "Accept": "image/*",
                }

                # Match --form parameters from cURL
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
                    print("response", response)
                    error_message = response.json().get("message", "Unknown error")
                    messages.error(request, f"Stability AI Error: {error_message}")
                    return redirect("image_generation:dashboard")
            else:
                result = pipe(prompt)
                generated_image = result.images[0]
                img_io = BytesIO()
                generated_image.save(img_io, format="PNG")
                generated_image = ContentFile(
                    img_io.getvalue(), name=f"{uuid.uuid4()}.png"
                )

            # Store image in chat history
            active_chat, _ = Chat.objects.get_or_create(
                user=request.user,
                defaults={"title": prompt[:50] + ("..." if len(prompt) > 50 else "")},
            )

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


def create_chat(request):
    if not request.user.is_authenticated:
        return redirect("authentication:login")

    with transaction.atomic():
        # Create new chat with default title
        new_chat = Chat.objects.create(user=request.user, title="New Chat")

    return redirect(f'{reverse("image_generation:dashboard")}?chat_id={new_chat.id}')


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
