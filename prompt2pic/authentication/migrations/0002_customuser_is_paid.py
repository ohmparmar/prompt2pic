# Generated by Django 5.1.5 on 2025-02-26 18:22

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("authentication", "0001_initial"),
    ]

    operations = [
        migrations.AddField(
            model_name="customuser",
            name="is_paid",
            field=models.BooleanField(default=False),
        ),
    ]
