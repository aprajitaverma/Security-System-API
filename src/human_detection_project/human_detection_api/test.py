from .models import ImageClass
import os
import cv2
from django.conf import settings
from PIL import Image
import numpy as np
from apscheduler.scheduler import Scheduler

sched = Scheduler()
sched.start()


def run_test(f):
    print(str(f) + "this is a test function")


def test():
    try:
        print("m")
    except (ValueError, AttributeError) as e:
        print("Unexpected error:", str(e))


def hit_screenshot_api():
    screenshot = 'capture/sam6556.jpg'
    ImageClass.objects.create(im_title="helloimage", im_photo=screenshot)


def check_hit():
    img = Image.open('D:/User/Images/test_images/img1.jpg')
    pix = np.array(img)
    image_name = "helloimage- 99 - today"
    obj = ImageClass.objects.create(im_title=image_name)
    new_name = 'detected_image' + str(obj.id) + ".jpg"
    cv2.imwrite(os.path.join(settings.BASE_DIR, 'capture', new_name), pix)
    obj.im_photo = os.path.join('capture', new_name)
    obj.save()


def check_sch():
    sched.add_date_job(run_test, '2019-04-23 14:49:15', ['yeehaw!'])