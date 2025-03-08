from django.contrib import admin
from .models import Transaction, Subscription

# Register Transaction model
@admin.register(Transaction)
class TransactionAdmin(admin.ModelAdmin):
    list_display = ('transaction_id', 'user', 'amount_paid', 'status', 'created_at')  # Fields to display in the list view
    list_filter = ('status', 'created_at')  # Filters for the sidebar
    search_fields = ('transaction_id', 'user__username')  # Searchable fields
    ordering = ('-created_at',)  # Order by creation date (newest first)
    date_hierarchy = 'created_at'  # Adds a date-based drilldown navigation

    # Optional: Make some fields read-only
    readonly_fields = ('created_at',)

    def get_queryset(self, request):
        # Optimize queryset by selecting related user
        return super().get_queryset(request).select_related('user')


# Register Subscription model
@admin.register(Subscription)
class SubscriptionAdmin(admin.ModelAdmin):
    list_display = ('plan_type', 'user', 'start_date', 'end_date', 'is_active')  # Fields to display in the list view
    list_filter = ('plan_type', 'start_date', 'end_date')  # Filters for the sidebar
    search_fields = ('user__username', 'plan_type')  # Searchable fields
    ordering = ('-start_date',)  # Order by start date (newest first)
    date_hierarchy = 'start_date'  # Adds a date-based drilldown navigation

    # Optional: Make some fields read-only
    readonly_fields = ('start_date',)

    def get_queryset(self, request):
        # Optimize queryset by selecting related user and transaction
        return super().get_queryset(request).select_related('user', 'transaction')

    # Optional: Customize how the is_active status is displayed
    def is_active(self, obj):
        return obj.is_active()
    is_active.boolean = True  # Displays as a green checkmark or red cross
    is_active.short_description = 'Active?'  # Column header in admin