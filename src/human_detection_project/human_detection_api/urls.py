from django.conf.urls import url, include


from rest_framework import routers

from . import views

router = routers.DefaultRouter()
router.register(r'postimage', views.PostImageAPI, basename="getimage")
router.register(r'run_detections', views.RunDetection)
# router.register(r'getimagebydate', views.GetImageAPI, basename="getimage")
urlpatterns = [
    url('human_detection/', views.DetectAPI.as_view()),
    # url('get_image/', views.GetImageAPI.as_view()),
    url('hit_test/', views.HitTestAPI.as_view()),
    url('fire_detection/', views.DetectFireAPI.as_view()),
    url('person_identification/', views.CheckPersonAPI.as_view()),
    url('all_operations/', views.TestAllOperations.as_view()),
    url('run_detection_settings/', views.RunDetectionSettings.as_view()),
    url(r'^', include(router.urls)),
]