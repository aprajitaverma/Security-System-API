# Code adapted from Tensorflow Object Detection Framework
# https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb
# Tensorflow Object Detection Detector

import face_recognition
import tensorflow as tf
import numpy as np
import datetime
import time
import cv2
import os

from . import schedule_detection
from django.conf import settings
from .models import ImageClass
from . import views


class DetectorAPI:
    def __init__(self, path_to_ckpt):
        self.path_to_ckpt = path_to_ckpt

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.path_to_ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.default_graph = self.detection_graph.as_default()
        self.sess = tf.Session(graph=self.detection_graph)

        # Definite input and output Tensors for detection_graph
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def process_frame(self, image):
        # Expand dimensions since the trained_model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image, axis=0)
        # Actual detection.
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})

        im_height, im_width, _ = image.shape
        boxes_list = [None for a in range(boxes.shape[1])]
        for j in range(boxes.shape[1]):
            boxes_list[j] = (int(boxes[0, j, 0] * im_height),
                             int(boxes[0, j, 1]*im_width),
                             int(boxes[0, j, 2] * im_height),
                             int(boxes[0, j, 3]*im_width)
                             )

        return boxes_list, scores[0].tolist(), [int(x) for x in classes[0].tolist()], int(num[0])

    def close(self):
        self.sess.close()
        self.default_graph.close()


def select_frame_in_frame(captured_frame, x, y, w, h):

    color = (0, 100, 0)             # BGR 0-255
    stroke = 2
    end_cord_x = x + w
    end_cord_y = y + h
    cv2.rectangle(captured_frame, (x, y), (end_cord_x, end_cord_y), color, stroke)
    cropped_frame = captured_frame[y:h, x:w]

    return cropped_frame


def save_in_database(name, captured_photo, type_image):
    obj = ImageClass.objects.create(im_title=name)
    new_name = 'detected_image' + str(obj.id) + ".jpg"
    cv2.imwrite(os.path.join(settings.BASE_DIR, 'capture', new_name), captured_photo)
    obj.im_photo = os.path.join('capture', new_name)
    if type_image == "cft":
        obj.im_type = "HD"
    elif type_image == "cff":
        obj.im_type = "FD"
    elif type_image == "fr":
        obj.im_type = "PM"
    obj.save()


def check_for_trespassers():
    start_time = 0
    model_path = 'human_detection_api/frozen_inference_graph.pb'

    od_api = DetectorAPI(path_to_ckpt=model_path)
    threshold = 0.7
    cap = cv2.VideoCapture(0)
    # rtsp://192.168.10.10:554/user=admin_password=tlJwpbo6_channel=1_stream=0.sdp?real_stream
    while True:

        if views.on_off_cft is True:
            r, img = cap.read()
            img = cv2.resize(img, (1280, 720))
            # selected_frame = select_frame_in_frame(img, 0, 0, 400, 720)
            boxes, scores, classes, num = od_api.process_frame(img)

            # Visualization of the results of a detection.

            for i in range(len(boxes)):
                print(classes.__str__())
                # Class 1 represents human
                if classes[i] == 1 and scores[i] > threshold:

                    end_time = time.time()
                    if end_time - start_time > 10:
                        start_time = time.time()
                        image_name = "human_detected"
                        # save_in_database(image_name, img, type_image="cft")
                        time.sleep(10)
                        break
                        # box = boxes[i]
                        # cv2.rectangle(img, (box[1], box[0]), (box[3], box[2]), (255, 0, 0), 4)
                        # break


        else:
            cap.release()
            cv2.destroyAllWindows()
            break


def check_for_fire():
    # video_file = "video_1.mp4"
    video = cv2.VideoCapture(0)

    while True:
        if views.on_off_cff is True:
            (grabbed, frame) = video.read()
            if not grabbed:
                break

            blur = cv2.GaussianBlur(frame, (21, 21), 0)
            hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

            lower = [18, 50, 50]
            upper = [35, 255, 255]
            lower = np.array(lower, dtype="uint8")
            upper = np.array(upper, dtype="uint8")
            mask = cv2.inRange(hsv, lower, upper)

            no_red = cv2.countNonZero(mask)
            # print("output:", frame)
            if int(no_red) > 20000:
                print("fire detected")
                image_name = "fire_detected-"
                save_in_database(image_name, frame, type_image="cff")
            # print(int(no_red))
            # print("output:".format(mask))

        else:
            cv2.destroyAllWindows()
            video.release()


def check_if_person_present(person, time_constraint):
    picture = person + ".jpg"
    image = face_recognition.load_image_file(os.path.join("faces", picture))

    image_face_encoding = face_recognition.face_encodings(image)[0]
    known_face_encodings = [image_face_encoding,
                            ]
    known_face_names = [person,
                        ]
    video_capture = cv2 .VideoCapture(0)
    process_this_frame = True
    global timer
    timer = time.time()
    global this_frame
    this_frame = False
    face_locations = []
    face_names = []

    while True:
        if views.on_off_fr:

            # Grab a frame of the video
            ret, frame = video_capture.read()
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, str(datetime.datetime.now()), (20, 20), font, 0.5, (255, 255, 255), 1)
            # Resize frame of video to smaller size for faster  processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame[:, :, ::-1]

            # Only process every other frame of video to save time
            if process_this_frame:
                # Find all the faces and face encodings in the current frame of video
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

                if this_frame is False:
                    if time.time() - timer > time_constraint:
                        print("hey1")
                        image_name = person + "_missing-"
                        save_in_database(image_name, frame, type_image="fr")
                        # print("Person missing")
                        this_frame = True
                for face_encoding in face_encodings:
                    # See if the face is a match for the known face(s)
                    match = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.55)
                    # Tolerance determines the accuracy of face identification.
                    # Lower the tolerance, higher is the accuracy.

                    # If a match was found in known_face_encodings, just use the first one.

                    if True in match:
                        # if timer == 0:
                        #     print("hey1")
                        #     print("Person not present in the first place")
                        # else:
                        this_frame = False
                        first_match_index = match.index(True)
                        name = known_face_names[first_match_index]
                        face_names.append(name)
                        # print("hey2")
                        timer = time.time()
                    else:
                        name = "Unknown"
                        print(name)
                        face_names.append(name)

            process_this_frame = not process_this_frame

            # Display the results
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
            face_names = []
            # Display the resulting image
            cv2.imshow('Video', frame)

            # Hit 'q' on the keyboard to quit!
            if cv2.waitKey(1) & 0xFF == ord('q'):
                video_capture.release()
                break
        else:
            video_capture.release()
            cv2.destroyAllWindows()

    video_capture.release()
    cv2.destroyAllWindows()


known_face_encodings = []
known_face_names = []
timer = 0


def all_operations():

    # ===================================================================
    if schedule_detection.time_constraint != 0:
        print(str(schedule_detection.time_constraint) + "I'm unstoppable, I'm a porche with no brakes")
        picture = schedule_detection.person + ".jpg"
        image = face_recognition.load_image_file(os.path.join("faces", picture))

        image_face_encoding = face_recognition.face_encodings(image)[0]

        global known_face_encodings
        known_face_encodings = [image_face_encoding,
                                ]
        global known_face_names
        known_face_names = [schedule_detection.person,
                            ]

        global timer
        timer = time.time()
        global ignore_this_frame
        ignore_this_frame = False
        face_locations = []
        face_names = []
        process_this_frame = True
    # ===================================================================

    print(schedule_detection.check_fire, "__________", schedule_detection.human_checker_not_present)
    start_time = 0
    st_time = 0
    model_path = 'human_detection_api/frozen_inference_graph.pb'
    print("The time has come")
    label_number = 0

    od_api = DetectorAPI(path_to_ckpt=model_path)
    threshold = 0.7
    video = cv2.VideoCapture(0)
    # rtsp://192.168.10.10:554/user=admin_password=tlJwpbo6_channel=1_stream=0.sdp?real_stream
    while True:
        if views.start_stop is True:
            # print("in condition (if views.start_stop is True:)")
            r, img = video.read()
            if schedule_detection.human_checker_not_present is True:
                # print("running person detection")
                img = cv2.resize(img, (1280, 720))
                # selected_frame = select_frame_in_frame(img, 0, 0, 400, 720)
                boxes, scores, classes, num = od_api.process_frame(img)

                # Visualization of the results of a detection.

                for i in range(len(boxes)):

                    # Class 1 represents human
                    if classes[i] == 1 and scores[i] > threshold:
                        end_time = time.time()
                        if end_time - start_time > 10:
                            start_time = time.time()
                            image_name = "human_detected"
                            # save_in_database(image_name, img, type_image="cft")
                            print(image_name)
                            time.sleep(10)
                            break
                            # box = boxes[i]
                            # cv2.rectangle(img, (box[1], box[0]), (box[3], box[2]), (255, 0, 0), 4)
                            # break
            else:
                # print("not running person detection yet")
                pass

            if schedule_detection.human_checker_present is True:
                # print("running person detection")
                img = cv2.resize(img, (1280, 720))
                # selected_frame = select_frame_in_frame(img, 0, 0, 400, 720)
                boxes, scores, classes, num = od_api.process_frame(img)

                # Visualization of the results of a detection.
                human_found = False
                for i in range(len(boxes)):

                    # Class 1 represents human
                    if classes[i] == 1 and scores[i] > threshold:
                        human_found = True
                        break

                if human_found is False:
                    end_time = time.time()
                    if end_time - st_time > 10:
                        st_time = time.time()
                        image_name = "human_not_present"
                        # save_in_database(image_name, img, type_image="cft")
                        print(image_name)
                        time.sleep(10)
                else:
                    pass

            else:
                # print("not running person detection yet")
                pass

            if schedule_detection.check_fire is True:
                print("running FIRE DETECTION")
                blur = cv2.GaussianBlur(img, (21, 21), 0)
                hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

                lower = [18, 50, 50]
                upper = [35, 255, 255]
                lower = np.array(lower, dtype="uint8")
                upper = np.array(upper, dtype="uint8")
                mask = cv2.inRange(hsv, lower, upper)

                no_red = cv2.countNonZero(mask)
                # print("output:", frame)
                if int(no_red) > 20000:
                    print("fire detected")
                    image_name = "fire_detected-"
                    # save_in_database(image_name, img, type_image="cff")
                    print(image_name)
                # print(int(no_red))
                # print("output:".format(mask))

            else:
                pass

            # FOR PERSON IDENTIFICATION
            # ==========================================================================================================
            if schedule_detection.person_present_checker:

                # Grab a frame of the video
                ret, frame = video.read()
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, str(datetime.datetime.now()), (20, 20), font, 0.5, (255, 255, 255), 1)
                # Resize frame of video to smaller size for faster  processing
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

                # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
                rgb_small_frame = small_frame[:, :, ::-1]

                # Only process every other frame of video to save time
                if process_this_frame:
                    # Find all the faces and face encodings in the current frame of video
                    face_locations = face_recognition.face_locations(rgb_small_frame)
                    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

                    if ignore_this_frame is False:

                        if time.time() - timer > schedule_detection.time_constraint and timer != 0:
                            image_name = schedule_detection.person + "_missing-"
                            # save_in_database(image_name, frame, type_image="fr")
                            print("Person missing - " + schedule_detection.person)
                            ignore_this_frame = True
                    for face_encoding in face_encodings:
                        # See if the face is a match for the known face(s)
                        match = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.55)
                        # Tolerance determines the accuracy of face identification.
                        # Lower the tolerance, higher is the accuracy.

                        # If a match was found in known_face_encodings, just use the first one.

                        if True in match:
                            # if timer == 0:
                            #     print("hey1")
                            #     print("Person not present in the first place")
                            # else:
                            ignore_this_frame = False
                            first_match_index = match.index(True)
                            name = known_face_names[first_match_index]
                            face_names.append(name)
                            print("hey " + name)
                            timer = time.time()
                        else:
                            name = "Unknown"
                            print(name)
                            face_names.append(name)

                process_this_frame = not process_this_frame

                # Display the results
                # for (top, right, bottom, left), name in zip(face_locations, face_names):
                #     # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                #     top *= 4
                #     right *= 4
                #     bottom *= 4
                #     left *= 4
                #
                #     # Draw a box around the face
                #     cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                #
                #     # Draw a label with a name below the face
                #     cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                #     cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
                # face_names = []
                # # Display the resulting image
                # cv2.imshow('Video', frame)
                #
                # # Hit 'q' on the keyboard to quit!
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     video.release()
                #     break
            # else:
            #     video.release()
            #     cv2.destroyAllWindows()
            # ==========================================================================================================
            if schedule_detection.object_detection is True:
                # print("running person detection")
                img = cv2.resize(img, (1280, 720))
                # selected_frame = select_frame_in_frame(img, 0, 0, 400, 720)
                boxes, scores, classes, num = od_api.process_frame(img)

                # Visualization of the results of a detection.
                if schedule_detection.object_name == "laptop":
                    label_number = 73
                elif schedule_detection.object_name == "cell phone":
                    label_number = 77
                elif schedule_detection.object_name == "bottle":
                    label_number = 44
                elif schedule_detection.object_name == "chair":
                    label_number = 62
                elif schedule_detection.object_name == "clock":
                    label_number = 85

                object_found = False
                for i in range(len(boxes)):

                    # Class 1 represents human
                    if classes[i] == label_number and scores[i] > threshold:
                        object_found = True
                        break

                if object_found is False:
                    end_time = time.time()
                    if end_time - st_time > 10:
                        st_time = time.time()
                        image_name = str(schedule_detection.object_name) + " not_present"
                        # save_in_database(image_name, img, type_image="cft")
                        print(image_name)
                        time.sleep(10)
                else:
                    pass

        elif views.start_stop is False:
            break

    cv2.destroyAllWindows()
    video.release()
