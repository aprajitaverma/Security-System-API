from django.db import models

# Create your models here.


class ImageClass(models.Model):
    """To save images from the detected feed."""

    id = models.AutoField(primary_key=True)
    im_title = models.CharField(max_length=200)
    im_photo = models.ImageField(upload_to='capture', default="")
    created_on = models.DateField(auto_now_add=True)

    HumanDetected = 'HD'
    FireDetected = 'FD'
    PersonMissing = 'PM'
    TYPE_CHOICES = (
        (HumanDetected, 'Human Detected'),
        (FireDetected, 'Fire Detected'),
        (PersonMissing, 'Person Missing')
    )
    im_type = models.CharField(
        blank=True,
        null=True,
        max_length=3,
        choices=TYPE_CHOICES,
        default=" ",
    )

    # def set_image_path(self, path):
    #     self.im_path = path

    def __str__(self):
        return '%s   %s   %s  %s' % (self.im_title, self.im_photo, self.im_type, self.created_on)


class InputFormForOperations(models.Model):
    """To take input for all the operations to be performed on the given video"""

    id = models.AutoField(primary_key=True)
    flag = models.BooleanField(default=False)
    fire_checker = models.BooleanField(default=False)
    fr_start_time = models.DateTimeField(default=None)
    fr_end_time = models.DateTimeField(default=None, blank=True, null=True)
    human_checker_not_present = models.BooleanField(default=False)
    hcnp_start_time = models.DateTimeField(default=None)
    hcnp_end_time = models.DateTimeField(default=None, blank=True, null=True)
    human_checker_present = models.BooleanField(default=False)
    hcp_start_time = models.DateTimeField(default=None, blank=True, null=True)
    hcp_end_time = models.DateTimeField(default=None, blank=True, null=True)

