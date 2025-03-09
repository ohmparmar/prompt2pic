from django.contrib import admin
from .models import Agent, Chat, ChatMessage

@admin.register(Agent)
class AgentAdmin(admin.ModelAdmin):
    list_display = ('name', 'is_available','is_paid')
    list_filter = ('is_available','is_paid')
    search_fields = ('name',)

@admin.register(Chat)
class ChatAdmin(admin.ModelAdmin):
    list_display = ('title', 'user', 'created_at')
    list_filter = ('created_at',)
    search_fields = ('title', 'user__username')

@admin.register(ChatMessage)
class ChatMessageAdmin(admin.ModelAdmin):
    list_display = ('chat', 'user_prompt', 'agent', 'created_at')
    list_filter = ('created_at', 'agent')
    search_fields = ('chat__title', 'user_prompt')
