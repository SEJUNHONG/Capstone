import tensorflow as tf
import src.align.detect_face
from facenet.src import facenet
import cv2
import numpy as np
import glob
import pickle
import collections
import os
from image_commons2 import nparray_as_image, draw_with_alpha, image_as_nparray
#from pytube import YouTube
from moviepy.editor import VideoFileClip,AudioFileClip
# import ffmpeg

def _load_emoticons(emotions):
    """
    Loads emotions images from graphics folder.
    :param emotions: Array of emotions names.
    :return: Array of emotions graphics.
    """
    return [nparray_as_image(cv2.imread('graphics/%s.png' % emotion, -1), mode=None) for emotion in emotions]

def main(filename,tmp_result,known_name):
    args = lambda: None
    args.video = True
    args.youtube_video_url = ''
    args.video_speedup = 3
    args.webcam = False
    emotions = ['neutral', 'anger', 'disgust', 'happy', 'sadness', 'surprise']
    emoticons = _load_emoticons(emotions)

    # load model
    model_emoji = cv2.face.FisherFaceRecognizer_create()
    model_emoji.read(r'C:/Users/mmlab/PycharmProjects/UI_pyqt/models/emotion_detection_model.xml')

    a = 0

    for i in range(0, len(
            os.listdir('C:/Users/mmlab/PycharmProjects/UI_pyqt/cluster_people'))):
        # print(len(os.listdir('C:/Users/mmlab/PycharmProjects/facenet-pytorch-master/facenet-pytorch-master/models/clusteringfolder/{}'.format(i))))
        a += len(os.listdir(
            'C:/Users/mmlab/PycharmProjects/UI_pyqt/cluster_people/{}'.format(
                i) + 'human'))
    folder = []
    folder_name = []
    folder_in_file = []
    under_folder = []
    for i in range(0, len(
            os.listdir('C:/Users/mmlab/PycharmProjects/UI_pyqt/cluster_people'))):
        b = len(os.listdir(
            'C:/Users/mmlab/PycharmProjects/UI_pyqt/cluster_people/{}'.format(
                i) + 'human'))
        d = int(b / a * 100)
        print(d)
        folder.append('C:/Users/mmlab/PycharmProjects/UI_pyqt/cluster_people/{}'.format(
            i) + 'human')
        folder_name.append(i)
        if d <30:
            under_folder.append(str(i) + 'human')
    for i in range(len(under_folder)):
        print(under_folder[i])
    for i in range(0, len(folder)):
        print(folder[i])
        print(folder_name[i])
        if int(folder_name[i]) > 0:
            file = os.path.join(
                'C:/Users/mmlab/PycharmProjects/UI_pyqt/cluster_people/{}'.format(
                    int(folder_name[i])) + 'human', str(folder_name[i]) + 'human1.png')
            folder_in_file.append(file)
            # print(folder_in_file[i])

    for i in range(0, len(folder_in_file)-1):
        cv = cv2.imread(folder_in_file[i], cv2.IMREAD_COLOR)
        #cv2.imwrite('model{}.png'.format(i), cv)
    minsize = 20
    threshold = [0.6, 0.7, 0.7]
    factor = 0.709
    image_size = 182
    input_image_size = 160

    # comment out these lines if you do not want video recording
    # USE FOR RECORDING VIDEO
    fourcc = cv2.VideoWriter_fourcc(*'FMP4')

    # Get the path of the classifier and load it
    project_root_folder = os.path.join(os.path.abspath(__file__), "C:/Users/mmlab/PycharmProjects/UI_pyqt/")
    classifier_path = project_root_folder + 'trained_classifier/video_new_name_test4.pkl'
    print (classifier_path)
    with open(classifier_path, 'rb') as f:
        (model, class_names) = pickle.load(f)
        print("Loaded classifier file")

    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            # Bounding box
            pnet, rnet, onet = src.align.detect_face.create_mtcnn(sess, project_root_folder + "src/align")
            # Get the path of the facenet model and load it
            facenet_model_path = project_root_folder + "20180402-114759/20180402-114759.pb"
            facenet.load_model(facenet_model_path)

            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            # Start video capture
            people_detected = set()

            person_detected = collections.Counter()

            if args.webcam is True:
                video_capture = cv2.VideoCapture(0)
            else:
                video_path = project_root_folder
                video_name = "vlog1"
                full_original_video_path_name = filename
                video_capture_path = full_original_video_path_name
                if not os.path.isfile(full_original_video_path_name):
                    print('Video not found at path ' + full_original_video_path_name + '. Commencing download from YouTube')
                    # Note if the video ever gets removed this will cause issues
                    #YouTube(args.youtube_video_url).streams.first().download(output_path =video_path, filename=video_name)
                video_capture = cv2.VideoCapture(full_original_video_path_name)
            width = video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
            height = video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float

            videoclip = VideoFileClip(full_original_video_path_name)
            audioclip = videoclip.audio
            video_recording = cv2.VideoWriter(project_root_folder + 'final_video_emoji.avi', fourcc, 10,(int(width), int(height)))
            output_video_name = project_root_folder + 'final_video_emoji.avi'

            total_frames_passed = 0
            face_frame = np.zeros(200, dtype=np.int32)
            while True:
                try:
                    ret, frame = video_capture.read()
                except Exception as e:
                    break
                if ret:
                    # Skip frames if video is to be sped up
                    if args.video_speedup:
                        total_frames_passed += 1
                        if total_frames_passed % args.video_speedup != 0:
                            continue

                    bounding_boxes, _ = src.align.detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
                    faces_found = bounding_boxes.shape[0]
                    #known_name=under_folder
                    #number = len(under_folder)
                    #for n in range(number):
                    #    known_name[n]=under_folder[n]
                    #known_name=['2human']


                    if faces_found > 0:
                        det = bounding_boxes[:, 0:4]

                        bb = np.zeros((faces_found, 4), dtype=np.int32)
                        for i in range(faces_found):
                            bb[i][0] = det[i][0]
                            bb[i][1] = det[i][1]
                            bb[i][2] = det[i][2]
                            bb[i][3] = det[i][3]

                            # inner exception
                            if bb[i][0] <= 0 or bb[i][1] <= 0 or bb[i][2] >= len(frame[0]) or bb[i][3] >= len(frame):
                                print('face is inner of range!')
                                continue

                            cropped = frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :]
                            scaled = cv2.resize(cropped, (input_image_size, input_image_size), interpolation=cv2.INTER_CUBIC)
                            # cv2.imshow("Cropped and scaled", scaled)
                            # cv2.waitKey(1)
                            scaled = facenet.prewhiten(scaled)
                            # cv2.imshow("\"Whitened\"", scaled)
                            # cv2.waitKey(1)
                            scaled_emoji = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
                            scaled_emoji = cv2.resize(scaled_emoji,(48,48))

                            scaled_reshape = scaled.reshape(-1, input_image_size, input_image_size, 3)
                            feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}
                            emb_array = sess.run(embeddings, feed_dict=feed_dict)
                            predictions = model.predict_proba(emb_array)
                            best_class_indices = np.argmax(predictions, axis=1)
                            best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                            best_name = class_names[best_class_indices[0]]
                            print("Name: {}, Probability: {}".format(best_name, best_class_probabilities))
                            if best_class_probabilities > 0.09:
                                #cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)    #boxing face

                                frame_speed = 20

                                for y in range(0,len(known_name)):
                                    if class_names[best_class_indices[0]] == known_name[y]:

                                        if face_frame[y] % frame_speed == 0:
                                            predictions_emoji = model_emoji.predict(scaled_emoji)
                                            if cv2.__version__ != '3.1.0':
                                                predictions_emoji = predictions_emoji[0]

                                        image_to_draw = emoticons[predictions_emoji]

                                        draw_with_alpha(frame, image_to_draw, (bb[i][0], bb[i][1], bb[i][2] - bb[i][0], bb[i][3] - bb[i][1]))

                                        face_frame[y] += 1
                                        print(face_frame[y])


                                #cv2.putText(frame, class_names[best_class_indices[0]], (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                 #           1, (0, 0, 255), thickness=1, lineType=2)
                                person_detected[best_name] += 1


                        # if total_frames_passed == 2:
                        for person, count in person_detected.items():
                            if count > 4:
                                print("Personc Detected: {}, Count: {}".format(person, count))
                                people_detected.add(person)
                        # person_detected.clear()
                        # total_frames_passed = 0


                    #cv2.putText(frame, "People detected so far:", (20, 20), cv2.FONT_HERSHEY_PLAIN,
                                #1, (255, 0, 0), thickness=1, lineType=2)
                    currentYIndex = 40
                    # for idx, name in enumerate(people_detected):
                    #     cv2.putText(frame, name, (20, currentYIndex + 20 * idx), cv2.FONT_HERSHEY_PLAIN,
                    #                 1, (0, 0, 255), thickness=1, lineType=2)
                    cv2.imshow("Face Detection and Identification", frame)
                    video_recording.write(frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    break
    video_recording.release()
    video_capture.release()
    cv2.destroyAllWindows()
    videoclip2 = VideoFileClip(output_video_name)
    videoclip2.audio = audioclip
    videoclip2.write_videofile(tmp_result)

if __name__ == "__main__":
    args = 1





