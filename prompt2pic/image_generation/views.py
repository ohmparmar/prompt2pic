from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.contrib.auth import logout
from django.contrib import messages
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
    # if request.user.is_authenticated:
    #     return render(request, 'image_generation/dashboard.html')
    return render(request, 'image_generation/dashboard.html')

