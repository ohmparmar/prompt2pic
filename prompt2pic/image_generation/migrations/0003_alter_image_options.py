# Generated by Django 5.1.5 on 2025-01-31 16:30

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ("image_generation", "0002_image_prompt_title"),
    ]

    operations = [
        migrations.AlterModelOptions(
            name="image",
            options={
                "ordering": ["id", "prompt_title", "prompt", "image_file", "created_at"]
            },
        ),
    ]
