from django.db import models
from django.conf import settings
# Create your models here.


class Agent(models.Model):
    """
    Model to keep track of agents.
    """
 
    name = models.CharField(max_length=255, unique=True)
    is_available = models.BooleanField(default=True)

    def __str__(self):
        return self.name


class Chat(models.Model):
    """
    Main table representing a chat session.
    """
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='chats'
    )
    title = models.CharField(max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.title


class ChatMessage(models.Model):
    """
    Table to store individual messages (or prompts) within a chat session.
    Each record is linked to a Chat via a ForeignKey, and may also be associated
    with an Agent.
    """
    # Reference to the main chat table (parent chat).
    chat = models.ForeignKey(
        Chat,
        on_delete=models.CASCADE,
        related_name="messages"
    )
    
    # The prompt entered by the user.
    user_prompt = models.TextField()
    agent = models.ForeignKey(
        Agent,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="chat_messages"
    )
    # The generated image; adjust the upload path as needed.
    image_generated = models.ImageField(
        upload_to='generated_images/',
        null=True,
        blank=True
    )
    
    # Reference to an agent, if one was involved in generating the answer.
 
    
    # Timestamp for when the message was created.
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        # Display a simple string with the chat title and a snippet of the prompt.
        return f"{self.chat.title} - {self.user_prompt[:20]}..."
