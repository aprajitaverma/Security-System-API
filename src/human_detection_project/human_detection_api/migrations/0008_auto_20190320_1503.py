# Generated by Django 2.1.4 on 2019-03-20 09:33

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('human_detection_api', '0007_auto_20190320_1502'),
    ]

    operations = [
        migrations.AlterField(
            model_name='imageclass',
            name='created_on',
            field=models.DateField(auto_now_add=True),
        ),
    ]
