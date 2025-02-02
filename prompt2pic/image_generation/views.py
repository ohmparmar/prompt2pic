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
from .models import Chat, Agent,ChatMessage

# # Create your views here.

# def generate_image(request):
#     if request.method == 'POST':
#         prompt = request.POST.get('prompt')
#         # Logic for image generation based on the prompt
#         # This is a placeholder for the actual image generation logic
#         image_url = f'http://example.com/generated_images/{prompt}.png'  # Replace with actual image generation logic
#         return JsonResponse({'image_url': image_url})
#     return render(request, 'image_generation/generate_image.html')
def dashboard(request):
    if not request.user.is_authenticated:
        return redirect('authentication:login')
    
    try:
        chat_id = request.GET.get('chat_id')
        active_chat = Chat.objects.get(id=chat_id, user=request.user)
    except (ObjectDoesNotExist, ValueError):
        active_chat = Chat.objects.filter(user=request.user).first()

    context = {
        'history': Chat.objects.filter(user=request.user).order_by('-created_at'),
        'models': Agent.objects.filter(is_available=True),
        'active_chat': active_chat,
        'error': messages.get_messages(request),
        'prompt_input': request.POST.get('prompt', ''),
        'selected_model': Agent.objects.filter(id=request.POST.get('model')).first()
    }
    return render(request, 'image_generation/dashboard.html', context)

# def dashboard(request):
#     if request.user.is_authenticated:
#          # Fetch chat history for the authenticated user
#         history = Chat.objects.filter(user=request.user).order_by('-created_at')
#         # Fetch all available agents
#         models = Agent.objects.filter(is_available=True)
#         return render(request, 'image_generation/dashboard.html', {'history': history, 'models': models})
#     return redirect('authentication:login')  # Redirect to login if not authenticated
def guest_dashboard(request):
    # if request.user.is_authenticated:
    #     return render(request, 'image_generation/dashboard.html')
    return render(request, 'image_generation/guest_user_dashboard.html')


# Initialize your model once when the server starts (optional but recommended)
# You can do this in a module-level variable so that it’s not reloaded on every request.
MODEL_ID = "CompVis/stable-diffusion-v1-4"  # or any other compatible model
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16 if device == "cuda" else torch.float32)
pipe = pipe.to(device)

# def generate_image(request):

#     if request.method == 'POST':
#         prompt = request.POST.get('prompt')

#         # Check if prompt is provided
#         if not prompt:
#             return JsonResponse({'error': 'No prompt provided.'}, status=400)
        
#         # Check for bad words (add your list of bad words here)
#         bad_words = ['badword1', 'badword2', 'anotherbadword']  # Example bad words list
#         if any(bad_word in prompt.lower() for bad_word in bad_words):
#             return JsonResponse({'error': 'Your prompt contains inappropriate content.'}, status=400)
        
#         # Ensure the user has an associated chat history
#         chat_entry, created = Chat.objects.get_or_create(user=request.user)
        
#         # Save the user's prompt to chat history
#         chat_entry.prompt = prompt
#         chat_entry.save()

#         # Generate the image using Stable Diffusion
#         try:
#             result = pipe(prompt)
#             image = result.images[0]
#         except Exception as e:
#             return JsonResponse({'error': f'Image generation failed: {str(e)}'}, status=500)

#         # Save the image to the media directory
#         filename = f"{uuid.uuid4()}.png"
#         media_dir = os.path.join("media", "generated_images")
#         os.makedirs(media_dir, exist_ok=True)
#         image_path = os.path.join(media_dir, filename)
#         image.save(image_path)

#         # Save the image URL in chat history
#         chat_entry.image_url = f"/media/generated_images/{filename}"
#         chat_entry.save()

#         # Return the image URL to the frontend
#         return JsonResponse({'image_url': request.build_absolute_uri(f"/media/generated_images/{filename}")})
    
#     return render(request, 'image_generation/generate_image.html')


from django.db import transaction
from django.core.exceptions import ObjectDoesNotExist
def generate_image(request):
    if not request.user.is_authenticated:
        return redirect('authentication:login')

    if request.method == 'POST':
        try:
            prompt = request.POST.get('prompt', '').strip()
            model_id = request.POST.get('model')

            # Validate inputs
            if not prompt:
                messages.error(request, 'Please enter a prompt')
                return redirect('image_generation:dashboard')
                
            if not model_id:
                messages.error(request, 'Please select a model')
                return redirect('image_generation:dashboard')

            # Check for bad words
            bad_words = ['nsfw', 'violent', 'explicit']  # Customize as needed
            if any(word in prompt.lower() for word in bad_words):
                messages.error(request, 'Prompt contains inappropriate content')
                return redirect('image_generation:dashboard')

            # Get selected model
            try:
                model = Agent.objects.get(id=model_id, is_available=True)
            except ObjectDoesNotExist:
                messages.error(request, 'Selected model is not available')
                return redirect('image_generation:dashboard')

            # Create or get active chat and generate image inside a transaction
            with transaction.atomic():
                active_chat = Chat.objects.filter(user=request.user).first()
                if not active_chat:
                    active_chat = Chat.objects.create(
                        user=request.user,
                        title=prompt[:50] + ('...' if len(prompt) > 50 else '')
                    )

                # Generate the image using Stable Diffusion
                try:
                    result = pipe(prompt)
                    generated_image = result.images[0]
                except Exception as e:
                    messages.error(request, f'Image generation failed: {str(e)}')
                    return redirect('image_generation:dashboard')

                # Save the generated image to the media directory
                filename = f"{uuid.uuid4()}.png"
                media_dir = os.path.join("media", "generated_images")
                os.makedirs(media_dir, exist_ok=True)
                image_path = os.path.join(media_dir, filename)
                generated_image.save(image_path)

                # Optionally, store a relative path (or use Django's FileField handling)
                relative_image_path = os.path.join('generated_images', filename)

                # Create chat message with the image path
                ChatMessage.objects.create(
                    chat=active_chat,
                    user_prompt=prompt,
                    agent=model,
                    image_generated=relative_image_path  # Store the relative path
                )

            return redirect(f'{reverse("image_generation:dashboard")}?chat_id={active_chat.id}')

        except Exception as e:
            messages.error(request, f'Error generating image: {str(e)}')
            return redirect('image_generation:dashboard')

    return redirect('image_generation:dashboard')