from django.db import models

# Create your models here.

class Image(models.Model):
    prompt = models.TextField()
    image_file = models.ImageField(upload_to='images/')
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Image generated from prompt: {self.prompt[:50]}"
