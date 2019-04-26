import threading

from rest_framework.parsers import JSONParser, FormParser, MultiPartParser
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework import status
from rest_framework import viewsets

from .models import InputFormForOperations
from .models import ImageClass
from . import schedule_detection
from . import serializers
from . import detection_files
from . import test

on_off_cft = "global"  # cft: check for trespassers
on_off_cff = "global"  # cff: check for fire
on_off_fr = "global"   # fr: check if person present

global start_stop
start_stop = True


class DetectAPI(APIView):
    """For detecting human"""

    def get(self, r):
        """Returns API view"""

        return Response({'API: Run human detection'})

    serializer_class = serializers.DetectionSerializer

    def post(self, request):
        """Start the script"""
        # url = 'http://192.168.10.73:8000/human_detection/detect/'
        serializer = serializers.DetectionSerializer(data=request.data)

        if serializer.is_valid():
            global on_off_cft
            on_off_cft = serializer.data.get('checker')
            if on_off_cft is True:
                t1 = threading.Thread(target=detection_files.check_for_trespassers)
                t1.start()
                return Response({"message": "Running human detection", "status": "True"}, status=status.HTTP_200_OK)
            else:
                return Response({"message": "Stopped human detection", "status": "False"}, status=status.HTTP_200_OK)
        else:
            return Response(
                serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class DetectFireAPI(APIView):
    """For detecting human"""

    def get(self, r):
        """Returns API view"""

        return Response({'API: Run fire detection'})

    serializer_class = serializers.DetectionSerializer

    def post(self, request):
        """Start the script"""
        # url = 'http://192.168.10.73:8000/human_detection/detect/'
        serializer = serializers.DetectionSerializer(data=request.data)

        if serializer.is_valid():
            global on_off_cff
            on_off_cff = serializer.data.get('checker')
            print(on_off_cff)
            if on_off_cff is True:
                t1 = threading.Thread(target=detection_files.check_for_fire)
                t1.start()
                return Response({"message": "Running Fire detection", "status": "True"}, status=status.HTTP_200_OK)
            else:
                return Response({"message": "Stopped Fire detection", "status": "False"}, status=status.HTTP_200_OK)
        else:
            return Response(
                serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class CheckPersonAPI(APIView):
    """For identifying and watching a person"""

    def get(self, r):
        """Returns API view"""

        return Response({'API: Run person Identification and tracking'})

    # serializer_class = serializers.DetectionSerializer
    serializer_class = serializers.PersonGetterSerializer

    def post(self, request):
        """Start the script"""
        # url = 'http://192.168.10.73:8000/human_detection/detect/'
        serializer = serializers.PersonGetterSerializer(data=request.data)
        # check_serializer = serializers.DetectionSerializer(data=request.data)

        if serializer.is_valid():
            global on_off_fr
            on_off_fr = serializer.data.get('checker')
            print(on_off_fr)
            if on_off_fr is True:
                name = serializer.data.get('person_name')
                time_duration = serializer.data.get('time_duration')
                t1 = threading.Thread(target=detection_files.check_if_person_present, args=(name, time_duration))
                t1.start()
                return Response({"message": "Running Person Identification and tracking", "status": "True"}, status=status.HTTP_200_OK)
            else:
                return Response({"message": "Stopped Person Identification and tracking", "status": "False"}, status=status.HTTP_200_OK)
        else:
            return Response(
                serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class PostImageAPI(viewsets.ModelViewSet):

    """Model view set for post image"""

    queryset = ImageClass.objects.all()
    serializer_class = serializers.ImageObjectSerializer
    parser_classes = (JSONParser, FormParser, MultiPartParser)


class HitTestAPI(APIView):
    def get(self, request):

            try:
                test.check_sch()
                return Response(
                    "Test Run Complete"
                )
            except Exception as e:
                return Response(
                    "Unexpected Error" + str(e)
                )


class TestAllOperations(APIView):
    """"""

    def get(self, r):
        """Returns API view"""

        return Response({'API: check all operations'})

    serializer_class = serializers.AllOperationFormSerializer

    def post(self, request):

        serializer = serializers.AllOperationFormSerializer(data=request.data)
        if serializer.is_valid():

            global start_stop
            start_stop = serializer.data.get('start_stop')
            if start_stop is True:
                check_fire = serializer.data.get('fire_checker')
                human_checker_not_present = serializer.data.get('human_checker_not_present')

                t1 = threading.Thread(target=detection_files.all_operations, args=(check_fire, human_checker_not_present))
                t1.start()
                return Response({"message": "Running given operations", "status": "True"}, status=status.HTTP_200_OK)
            else:
                return Response({"message": "Stopped", "status": "False"}, status=status.HTTP_200_OK)


class RunDetection(viewsets.ModelViewSet):

    queryset = InputFormForOperations.objects.all()
    serializer_class = serializers.InputFormSerializer


class RunDetectionSettings(APIView):

    def get(self, r):
        """Returns API view"""

        return Response({'API: check all operations in settings'})

    serializer_class = serializers.DetectionInputSerializer

    def post(self, request):

        serializer = serializers.DetectionInputSerializer(data=request.data)

        if serializer.is_valid():
            global check_fire
            check_fire = serializer.data.get('fire_checker')
            fc_start_time = serializer.data.get('fc_start_time')
            fc_end_time = serializer.data.get('fc_end_time')
            human_checker_not_present = serializer.data.get('human_checker_not_present')
            hcnp_start_time = serializer.data.get('hcnp_start_time')
            hcnp_end_time = serializer.data.get('hcnp_end_time')
            human_checker_present = serializer.data.get('human_checker_present')
            hcp_start_time = serializer.data.get('hcp_start_time')
            hcp_end_time = serializer.data.get('hcp_end_time')
            person_present_checker = serializer.data.get('person_present_checker')
            ppc_start_time = serializer.data.get('ppc_start_time')
            ppc_end_time = serializer.data.get('ppc_end_time')
            person_name = serializer.data.get('person_name')
            time_constraint = serializer.data.get('time_constraint')
            object_detection = serializer.data.get('object_detection')
            object_name = serializer.data.get('object_name')
            od_start_time = serializer.data.get('od_start_time')
            od_end_time = serializer.data.get('od_end_time')

            t1 = threading.Thread(target=schedule_detection.run_scheduler, args=(check_fire, fc_start_time, fc_end_time,
                                                                                 human_checker_not_present,
                                                                                 hcnp_start_time, hcnp_end_time,
                                                                                 human_checker_present, hcp_start_time,
                                                                                 hcp_end_time, person_present_checker,
                                                                                 ppc_start_time, ppc_end_time,
                                                                                 person_name, time_constraint,
                                                                                 object_detection, object_name,
                                                                                 od_start_time, od_end_time))

            t1.start()
            return Response({"message": "Saved given operation settings", "status": "True"}, status=status.HTTP_200_OK)

        else:
            return Response(
                serializer.errors, status=status.HTTP_400_BAD_REQUEST)