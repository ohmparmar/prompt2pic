# Generated by Django 5.1.5 on 2025-02-28 17:11

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ("image_generation", "0006_agent_is_paid_transaction_subscription"),
    ]

    operations = [
        migrations.RemoveField(
            model_name="transaction",
            name="user",
        ),
        migrations.DeleteModel(
            name="Subscription",
        ),
        migrations.DeleteModel(
            name="Transaction",
        ),
    ]
