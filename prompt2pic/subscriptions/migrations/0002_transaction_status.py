# Generated by Django 5.1.5 on 2025-02-28 17:13

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("subscriptions", "0001_initial"),
    ]

    operations = [
        migrations.AddField(
            model_name="transaction",
            name="status",
            field=models.IntegerField(default=1),
            preserve_default=False,
        ),
    ]
