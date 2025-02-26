from django.db import models
from django.conf import settings

# Create your models here.


class Agent(models.Model):
    """
    Model to keep track of agents.
    """

    name = models.CharField(max_length=255, unique=True)
    is_available = models.BooleanField(default=True)
    is_paid = models.BooleanField(
        default=False
    )  # Field to indicate if the model is paid

    def __str__(self):
        return self.name


class Chat(models.Model):
    """
    Main table representing a chat session.
    """

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="chats"
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
    chat = models.ForeignKey(Chat, on_delete=models.CASCADE, related_name="messages")

    # The prompt entered by the user.
    user_prompt = models.TextField()
    agent = models.ForeignKey(
        Agent,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="chat_messages",
    )
    # The generated image; adjust the upload path as needed.
    image_generated = models.ImageField(
        upload_to="generated_images/", null=True, blank=True
    )

    # Reference to an agent, if one was involved in generating the answer.

    # Timestamp for when the message was created.
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        # Display a simple string with the chat title and a snippet of the prompt.
        return f"{self.chat.title} - {self.user_prompt[:20]}..."


class Transaction(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    amount_paid = models.DecimalField(max_digits=10, decimal_places=2)
    transaction_id = models.CharField(max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Transaction {self.transaction_id} by {self.user.username}"


class Subscription(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    transaction = models.ForeignKey(Transaction, on_delete=models.CASCADE)
    plan_type = models.CharField(max_length=50)
    start_date = models.DateTimeField(auto_now_add=True)
    end_date = models.DateTimeField()  # New field for subscription end date

    def __str__(self):
        return f"{self.plan_type} subscription for {self.user.username}"
