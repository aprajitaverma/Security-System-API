from rest_framework import serializers
from .models import ImageClass
from .models import InputFormForOperations


class DetectionSerializer(serializers.Serializer):
    """Serializes the input"""

    checker = serializers.BooleanField()


class DateSerializer(serializers.Serializer):
    """Serializes the input for Image Response to the client."""
    date = serializers.DateField()


class ImageObjectSerializer(serializers.ModelSerializer):
    class Meta:
        model = ImageClass
        fields = "id", "im_title", "im_photo", "created_on", "im_type"


class PersonGetterSerializer(serializers.Serializer):
    """Serializes the input for person present"""

    person_name = serializers.CharField(max_length=100, required=False)
    time_duration = serializers.IntegerField(required=False)
    checker = serializers.BooleanField()


class AllOperationFormSerializer(serializers.Serializer):

    fire_checker = serializers.BooleanField()
    human_checker_not_present = serializers.BooleanField()
    start_stop = serializers.BooleanField()


class InputFormSerializer(serializers.ModelSerializer):

    class Meta:
        model = InputFormForOperations
        fields = "id", "flag", "fire_checker", "fr_start_time", "fr_end_time", "human_checker_not_present", \
                 "hcnp_start_time", "hcnp_end_time"


class DetectionInputSerializer(serializers.Serializer):

    fire_checker = serializers.BooleanField()
    fc_start_time = serializers.DateTimeField(required=False)
    fc_end_time = serializers.DateTimeField(required=False, allow_null=True)
    human_checker_not_present = serializers.BooleanField()
    hcnp_start_time = serializers.DateTimeField(required=False)
    hcnp_end_time = serializers.DateTimeField(required=False, allow_null=True)
    human_checker_present = serializers.BooleanField()
    hcp_start_time = serializers.DateTimeField(required=False)
    hcp_end_time = serializers.DateTimeField(required=False, allow_null=True)
    person_present_checker = serializers.BooleanField()
    ppc_start_time = serializers.DateTimeField(required=False)
    ppc_end_time = serializers.DateTimeField(required=False, allow_null=True)
    person_name = serializers.CharField(max_length=200, required=False)
    time_constraint = serializers.IntegerField(required=False)



