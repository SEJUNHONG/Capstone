import tensorflow as tf
import src.align.detect_face
from facenet.src import facenet
import cv2
import numpy as np
import glob
import pickle
import collections
import os

#from pytube import YouTube
# import ffmpeg

from moviepy.editor import VideoFileClip,AudioFileClip

def check(filename,tmp_result,known_name):
    args = lambda : None
    args.video = True
    args.youtube_video_url = ''
    args.video_speedup = 2
    args.webcam = False
    a = 0
    b = 0
    c=0
    for i in range(0, len(
            os.listdir('C:/Users/mmlab/PycharmProjects/UI_pyqt/cluster_people'))):
        # print(len(os.listdir('C:/Users/mmlab/PycharmProjects/facenet-pytorch-master/facenet-pytorch-master/models/clusteringfolder/{}'.format(i))))
        a += len(os.listdir(
            'C:/Users/mmlab/PycharmProjects/UI_pyqt/cluster_people/{}'.format(
                i) + 'human'))
    print("asdfasdfdsa")
    print(len(
            os.listdir('C:/Users/mmlab/PycharmProjects/UI_pyqt/cluster_people')))
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
            #print(folder_in_file[i])

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
    project_root_folder = os.path.join(os.path.abspath(__file__), 'C:/Users/mmlab/PycharmProjects/UI_pyqt/')
    classifier_path = project_root_folder + 'trained_classifier/video_new_name_test4.pkl'
    print(classifier_path)
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


            video_path = project_root_folder
            video_name = filename # 영상
            full_original_video_path_name = filename
            print(filename)
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
            #저장파일 이름
            video_recording = cv2.VideoWriter(project_root_folder + 'final_video_mosaic.avi', fourcc,12,(int(width), int(height)))
            output_video_name = project_root_folder + 'final_video_mosaic.avi'

            total_frames_passed = 0

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
                    frame_track=[]
                    faces_found = bounding_boxes.shape[0]
                    #number = len(under_folder)
                    #for n in range(number):
                    #    known_name[n]=under_folder[n]
                    #known_name=['1human']
                    print(known_name)
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

                            scaled_reshape = scaled.reshape(-1, input_image_size, input_image_size, 3)
                            feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}
                            emb_array = sess.run(embeddings, feed_dict=feed_dict)
                            predictions = model.predict_proba(emb_array)
                            best_class_indices = np.argmax(predictions, axis=1)
                            best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                            best_name = class_names[best_class_indices[0]]
                            #print("Name: {}, Probability: {}".format(best_name, best_class_probabilities))

                            if best_class_probabilities > 0.09:
                                #cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)    #boxing face


                                for y in range(0,len(known_name)):
                                    if class_names[best_class_indices[0]] == known_name[y]:
                                        frame[bb[i][1] + 5: bb[i][3] - 5, bb[i][0] + 2: bb[i][2] - 2] =cv2.blur(frame[bb[i][1] + 5: bb[i][3] - 5, bb[i][0] + 2: bb[i][2] - 2], (50,50))
                                            #cv2.blur(frame[bb[i][1] + 5: bb[i][3] - 5, bb[i][0] + 2: bb[i][2] - 2], (23,23))
                                            # cv2.bilateralFilter(frame[bb[i][1] + 5: bb[i][3] - 5, bb[i][0] + 2: bb[i][2] - 2], 12, 300, 300)

                                    for j in range(100):
                                        c=+1

                                #cv2.putText(frame, class_names[best_class_indices[0]], (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                 #           1, (0, 0, 255), thickness=1, lineType=2)
                                person_detected[best_name] += 1

                        # total_frames_passed += 1
                        # if total_frames_passed == 2:

                        for person, count in person_detected.items():
                            if count > 4:
                                print("Person Detected: {}, Count: {}".format(person, count))
                                people_detected.add(person)

                        # person_detected.clear()
                        # total_frames_passed = 0

                    #cv2.putText(frame, "People detected so far:", (20, 20), cv2.FONT_HERSHEY_PLAIN,
                                #1, (255, 0, 0), thickness=1, lineType=2)
                    '''
                    currentYIndex = 40
                    for idx, name in enumerate(people_detected):
                        cv2.putText(frame, name, (20, currentYIndex + 20 * idx), cv2.FONT_HERSHEY_PLAIN,
                                    1, (0, 0, 255), thickness=1, lineType=2)
                    '''
                    cv2.imshow("Face Detection and Identification", frame)
                    video_recording.write(frame)
                    frame_track.append(frame)
                    #if cv2.waitKey(1) & 0xFF == ord('q'):
                    #    break
                else:
                    break
    print("mosaiced")
    video_recording.release()
    video_capture.release()
    cv2.destroyAllWindows()
    videoclip2 = VideoFileClip(output_video_name)
    videoclip2.audio = audioclip
    #저장파일 일므
    videoclip2.write_videofile("tmp_result.mp4")
    print("done")

def main(filename,tmp_result):
    args = lambda : None
    args.video = True
    args.youtube_video_url = ''
    args.video_speedup = 2
    args.webcam = False
    a = 0
    b = 0
    c=0
    for i in range(0, len(
            os.listdir('C:/Users/mmlab/PycharmProjects/UI_pyqt/cluster_people'))):
        # print(len(os.listdir('C:/Users/mmlab/PycharmProjects/facenet-pytorch-master/facenet-pytorch-master/models/clusteringfolder/{}'.format(i))))
        a += len(os.listdir(
            'C:/Users/mmlab/PycharmProjects/UI_pyqt/cluster_people/{}'.format(
                i) + 'human'))
    print("asdfasdfdsa")
    print(len(
            os.listdir('C:/Users/mmlab/PycharmProjects/UI_pyqt/cluster_people')))
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
            #print(folder_in_file[i])

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
    project_root_folder = os.path.join(os.path.abspath(__file__), 'C:/Users/mmlab/PycharmProjects/UI_pyqt/')
    classifier_path = project_root_folder + 'trained_classifier/video_new_name_test4.pkl'
    print(classifier_path)
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


            video_path = project_root_folder
            video_name = filename # 영상
            full_original_video_path_name = filename
            print(filename)
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
            #저장파일 이름
            video_recording = cv2.VideoWriter(project_root_folder + 'final_video_mosaic.avi', fourcc,15,(int(width), int(height)))
            output_video_name = project_root_folder + 'final_video_mosaic.avi'

            total_frames_passed = 0

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
                    frame_track=[]
                    faces_found = bounding_boxes.shape[0]
                    #known_name=under_folder
                    known_name=['4human']
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

                            scaled_reshape = scaled.reshape(-1, input_image_size, input_image_size, 3)
                            feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}
                            emb_array = sess.run(embeddings, feed_dict=feed_dict)
                            predictions = model.predict_proba(emb_array)
                            best_class_indices = np.argmax(predictions, axis=1)
                            best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                            best_name = class_names[best_class_indices[0]]

                            if best_class_probabilities > 0.09:

                                for y in range(0,len(known_name)):
                                    if class_names[best_class_indices[0]] == known_name[y]:
                                        frame[bb[i][1] + 5: bb[i][3] - 5, bb[i][0] + 2: bb[i][2] - 2] =cv2.blur(frame[bb[i][1] + 5: bb[i][3] - 5, bb[i][0] + 2: bb[i][2] - 2], (50,50))

                                    for j in range(100):
                                        c=+1

                                person_detected[best_name] += 1

                            else:
                                cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)

                        # total_frames_passed += 1
                        # if total_frames_passed == 2:

                        for person, count in person_detected.items():
                            if count > 4:
                                print("Person Detected: {}, Count: {}".format(person, count))
                                people_detected.add(person)

                        # person_detected.clear()
                        # total_frames_passed = 0

                    #cv2.putText(frame, "People detected so far:", (20, 20), cv2.FONT_HERSHEY_PLAIN,
                                #1, (255, 0, 0), thickness=1, lineType=2)
                    '''
                    currentYIndex = 40
                    for idx, name in enumerate(people_detected):
                        cv2.putText(frame, name, (20, currentYIndex + 20 * idx), cv2.FONT_HERSHEY_PLAIN,
                                    1, (0, 0, 255), thickness=1, lineType=2)
                    '''
                    cv2.imshow("Face Detection and Identification", frame)
                    video_recording.write(frame)
                    frame_track.append(frame)
                    #if cv2.waitKey(1) & 0xFF == ord('q'):
                    #    break
                else:
                    break
    print("mosaiced")
    video_recording.release()
    video_capture.release()
    cv2.destroyAllWindows()
    videoclip2 = VideoFileClip(output_video_name)
    videoclip2.audio = audioclip
    #저장파일 일므
    videoclip2.write_videofile("tmp_result1.mp4")
    print("done")
