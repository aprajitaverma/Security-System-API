# Generated by Django 2.1.4 on 2019-03-26 07:15

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('human_detection_api', '0008_auto_20190320_1503'),
    ]

    operations = [
        migrations.AddField(
            model_name='imageclass',
            name='im_url',
            field=models.CharField(default='', max_length=1000),
        ),
        migrations.AlterField(
            model_name='imageclass',
            name='im_photo',
            field=models.ImageField(default='', editable=False, upload_to='capture/'),
        ),
    ]