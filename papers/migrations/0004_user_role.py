# Generated by Django 5.1.3 on 2024-11-29 09:01

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('papers', '0003_user_identity'),
    ]

    operations = [
        migrations.AddField(
            model_name='user',
            name='role',
            field=models.CharField(blank=True, max_length=255, null=True),
        ),
    ]