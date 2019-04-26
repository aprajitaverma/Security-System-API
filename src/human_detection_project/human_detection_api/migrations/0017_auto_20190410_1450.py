# Generated by Django 2.1.4 on 2019-04-10 09:20

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('human_detection_api', '0016_imageclass_im_type'),
    ]

    operations = [
        migrations.AlterField(
            model_name='imageclass',
            name='im_type',
            field=models.CharField(choices=[('HD', 'human_detected'), ('FD', 'fire_detected')], default=None, max_length=2, null=True),
        ),
    ]
