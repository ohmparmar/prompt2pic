from django.db import models

# Create your models here.

class Image(models.Model):
    prompt = models.TextField()
    prompt_title = models.CharField(max_length=255, null=True, blank=True)  # Allow null values
    image_file = models.ImageField(upload_to='images/')
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['id', 'prompt_title', 'prompt', 'image_file', 'created_at']  # Specify the order of fields

    def __str__(self):
        return f"Image generated from prompt: {self.prompt_title[:50]}"

