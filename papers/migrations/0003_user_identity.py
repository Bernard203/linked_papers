# Generated by Django 5.1.3 on 2024-11-26 17:54

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('papers', '0002_user_nickname_alter_user_email'),
    ]

    operations = [
        migrations.AddField(
            model_name='user',
            name='identity',
            field=models.CharField(blank=True, max_length=255, null=True),
        ),
    ]
