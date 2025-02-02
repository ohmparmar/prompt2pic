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
from .models import Chat, Agent

# Create your views here.

def generate_image(request):
    if request.method == 'POST':
        prompt = request.POST.get('prompt')
        # Logic for image generation based on the prompt
        # This is a placeholder for the actual image generation logic
        image_url = f'http://example.com/generated_images/{prompt}.png'  # Replace with actual image generation logic
        return JsonResponse({'image_url': image_url})
    return render(request, 'image_generation/generate_image.html')

def dashboard(request):
    if request.user.is_authenticated:
         # Fetch chat history for the authenticated user
        history = Chat.objects.filter(user=request.user).order_by('-created_at')
        # Fetch all available agents
        models = Agent.objects.filter(is_available=True)
        return render(request, 'image_generation/dashboard.html', {'history': history, 'models': models})
    return redirect('authentication:login')  # Redirect to login if not authenticated
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

def generate_image(request):
    if request.method == 'POST':
        prompt = request.POST.get('prompt')
        if not prompt:
            return JsonResponse({'error': 'No prompt provided.'}, status=400)
        
        try:
            # Generate image from the prompt
            result = pipe(prompt)
            image = result.images[0]
        except Exception as e:
            return JsonResponse({'error': f'Image generation failed: {str(e)}'}, status=500)

        # Save the image to your media directory (make sure this directory exists and is served properly)
        filename = f"{uuid.uuid4()}.png"
        media_dir = os.path.join("media", "generated_images")
        os.makedirs(media_dir, exist_ok=True)
        image_path = os.path.join(media_dir, filename)
        image.save(image_path)

        # Build an absolute URL for the saved image
        image_url = request.build_absolute_uri(f"/media/generated_images/{filename}")
        return JsonResponse({'image_url': image_url})
    
    return render(request, 'image_generation/generate_image.html')
