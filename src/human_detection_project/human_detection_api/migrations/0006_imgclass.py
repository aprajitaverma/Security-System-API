# Generated by Django 2.1.4 on 2019-03-19 10:50

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('human_detection_api', '0005_imageclass_created_on'),
    ]

    operations = [
        migrations.CreateModel(
            name='ImgClass',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('im_title', models.CharField(max_length=200)),
            ],
        ),
    ]