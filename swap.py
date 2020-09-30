import tensorflow.compat.v1 as tf
import src.align.detect_face
from facenet.src import facenet
import cv2
import numpy as np
import glob
import pickle
import collections
import os
import dlib
#from pytube import YouTube
# import ffmpeg
from moviepy.editor import VideoFileClip,AudioFileClip
def extract_index_nparray(nparray):
    index = None
    for num in nparray[0]:
        index = num
        break
    return index


def main(filename,tmp_result, known_name):
    args = lambda: None
    args.video = True
    args.youtube_video_url = ''
    args.video_speedup = 2
    args.webcam = False

    minsize = 20
    threshold = [0.6, 0.7, 0.7]
    factor = 0.709
    image_size = 182
    input_image_size = 160

    img = cv2.imread("test7.jpg")
    img = cv2.resize(img, (int(img.shape[1] * 0.6), int(img.shape[0] * 0.6)))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(img_gray)
    indexes_triangles = []

    # dlib에 있는 정면 얼굴 검출기(detector)로 입력 사진에서 얼굴을 검출해 faces로 반환
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    # Face 1
    faces = detector(img_gray)
    for face in faces:
        # print(face)
        landmarks = predictor(img_gray, face)
        landmarks_points = []
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            landmarks_points.append((x, y))
    points = np.array(landmarks_points, np.int32)
    convexhull = cv2.convexHull(points)
    # cv2.polylines(img, [convexhull], True, (255, 0, 0), 3)
    cv2.fillConvexPoly(mask, convexhull, 255)

    face_image_1 = cv2.bitwise_and(img, img, mask=mask)

    # Delaunay triangulation
    rect = cv2.boundingRect(convexhull)
    subdiv = cv2.Subdiv2D(rect)
    subdiv.insert(landmarks_points)
    triangles = subdiv.getTriangleList()  # landmarks_point 값을 배열로 변환
    triangles = np.array(triangles, dtype=np.int32)
    for t in triangles:
        pt1 = (t[0], t[1])  # 삼각형의 좌표를 배열로 저장
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])
        # 삼각형의 좌표와 landmarks_point 가 만나는 곳의 점을 저장(0~68)
        index_pt1 = np.where((points == pt1).all(axis=1))
        index_pt1 = extract_index_nparray(index_pt1)

        index_pt2 = np.where((points == pt2).all(axis=1))
        index_pt2 = extract_index_nparray(index_pt2)

        index_pt3 = np.where((points == pt3).all(axis=1))
        index_pt3 = extract_index_nparray(index_pt3)

        if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
            triangle = [index_pt1, index_pt2, index_pt3]  # 삼각형마다의 landmarks_point
            indexes_triangles.append(triangle)



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
                video_name = "vlog"
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
            video_recording = cv2.VideoWriter(project_root_folder + 'final_video_swap.avi', fourcc, 13,(int(width), int(height)))
            output_video_name = project_root_folder + 'final_video_swap.avi'

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

                    bounding_boxes, _ = src.align.detect_face.detect_face(frame, minsize, pnet, rnet, onet,threshold, factor)
                    if bounding_boxes is not None:
                        print('maps:' + str(bounding_boxes))
                        faces_found = bounding_boxes.shape[0]
                        #number = len(under_folder)
                        #for n in range(number):
                            #known_name[n] = under_folder[n]
                        #known_name = ['2human']

                        if faces_found > 0:
                            det = bounding_boxes[:, 0:4]

                            bb = np.zeros((faces_found, 4), dtype=np.int32)
                            for i in range(faces_found):
                                bb[i][0] = det[i][0]
                                bb[i][1] = det[i][1]
                                bb[i][2] = det[i][2]
                                bb[i][3] = det[i][3]

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
                                print("Name: {}, Probability: {}".format(best_name, best_class_probabilities))
                                if best_class_probabilities > 0.09:
                                    #cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)    #boxing face
                                    text_x = bb[i][0]
                                    text_y = bb[i][3] + 20
                                    for j in range(len(known_name)):
                                        if class_names[best_class_indices[0]] == known_name[j]:
                                            img2=frame[bb[i][1]-10 : bb[i][3]+20, bb[i][0]-10: bb[i][2]+20]
                                            try:
                                                img2_gray=cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
                                                img2_new_face = np.zeros_like(img2)
                                                faces2 = detector(img2_gray)
                                                print(len(faces2))

                                                if len(faces2)>0:
                                                    face1=faces2[0]
                                                    landmarks = predictor(img2_gray, face1)
                                                    landmarks_points2 = []
                                                    for n in range(0, 68):
                                                        x = landmarks.part(n).x
                                                        y = landmarks.part(n).y
                                                        landmarks_points2.append((x, y))
                                                        # for face in faces2:
                                                        #     landmarks = predictor(img2_gray, face)
                                                        #     landmarks_points2 = []
                                                        #     for n in range(0, 68):
                                                        #         x = landmarks.part(n).x
                                                        #         y = landmarks.part(n).y
                                                        #         landmarks_points2.append((x, y))

                                                        # cv2.circle(img2, (x, y), 3, (0, 255, 0), -1)
                                                        points2 = np.array(landmarks_points2, np.int32)
                                                        convexhull2 = cv2.convexHull(points2)

                                                    lines_space_mask = np.zeros_like(img_gray)
                                                    lines_space_new_face = np.zeros_like(img2)

                                                    # Triangulation of both faces
                                                    for triangle_index in indexes_triangles:
                                                        # Triangulation of the first face
                                                        tr1_pt1 = landmarks_points[triangle_index[0]]
                                                        tr1_pt2 = landmarks_points[triangle_index[1]]
                                                        tr1_pt3 = landmarks_points[triangle_index[2]]
                                                        triangle1 = np.array([tr1_pt1, tr1_pt2, tr1_pt3], np.int32)

                                                        rect1 = cv2.boundingRect(triangle1)
                                                        (x, y, w, h) = rect1

                                                        cropped_triangle = img[y: y + h, x: x + w]
                                                        cropped_tr1_mask = np.zeros((h, w), np.uint8)

                                                        points = np.array([[tr1_pt1[0] - x, tr1_pt1[1] - y],[tr1_pt2[0] - x, tr1_pt2[1] - y],[tr1_pt3[0] - x, tr1_pt3[1] - y]], np.int32)

                                                        cv2.fillConvexPoly(cropped_tr1_mask, points, 255)

                                                        # Triangulation of second face
                                                        tr2_pt1 = landmarks_points2[triangle_index[0]]
                                                        tr2_pt2 = landmarks_points2[triangle_index[1]]
                                                        tr2_pt3 = landmarks_points2[triangle_index[2]]
                                                        triangle2 = np.array([tr2_pt1, tr2_pt2, tr2_pt3], np.int32)

                                                        rect2 = cv2.boundingRect(triangle2)
                                                        (x, y, w, h) = rect2

                                                        if x<0:
                                                            x=0
                                                            rect2=(x, y, w, h)
                                                            (x,y,w,h)=rect2
                                                        if y<0:
                                                            y=0
                                                            rect2 = (x, y, w, h)
                                                            (x, y, w, h) = rect2
                                                        if w<0:
                                                            w=0
                                                            rect2 = (x, y, w, h)
                                                            (x, y, w, h) = rect2
                                                        if h<0:
                                                            h=0
                                                            rect2 = (x, y, w, h)
                                                            (x, y, w, h) = rect2
                                                        print(rect2)
                                                        cropped_tr2_mask = np.zeros((h, w), np.uint8)

                                                        points2 = np.array([[tr2_pt1[0] - x, tr2_pt1[1] - y],[tr2_pt2[0] - x, tr2_pt2[1] - y],[tr2_pt3[0] - x, tr2_pt3[1] - y]], np.int32)

                                                        cv2.fillConvexPoly(cropped_tr2_mask, points2, 255)

                                                        # Warp triangles
                                                        points = np.float32(points)
                                                        points2 = np.float32(points2)
                                                        M = cv2.getAffineTransform(points, points2)
                                                        warped_triangle = cv2.warpAffine(cropped_triangle, M, (w, h))
                                                        warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle,mask=cropped_tr2_mask)

                                                        # Reconstructing destination face
                                                        img2_new_face_rect_area = img2_new_face[y: y + h, x: x + w]
                                                        img2_new_face_rect_area_gray = cv2.cvtColor(img2_new_face_rect_area,cv2.COLOR_BGR2GRAY)
                                                        _, mask_triangles_designed = cv2.threshold(img2_new_face_rect_area_gray, 1,255, cv2.THRESH_BINARY_INV)
                                                        _, mask_triangles_designed2 = cv2.threshold(warped_triangle,1, 255,cv2.THRESH_BINARY_INV)
                                                        print(len(warped_triangle))
                                                        print(len(mask_triangles_designed))

                                                        if len(warped_triangle) == len(mask_triangles_designed):
                                                            warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle,mask=mask_triangles_designed)
                                                        else:
                                                            warped_triangle = warped_triangle

                                                        img2_new_face_rect_area = cv2.add(img2_new_face_rect_area, warped_triangle)
                                                        img2_new_face[y: y + h, x: x + w] = img2_new_face_rect_area

                                                    img2_face_mask = np.zeros_like(img2_gray)
                                                    img2_head_mask = cv2.fillConvexPoly(img2_face_mask, convexhull2, 255)
                                                    img2_face_mask = cv2.bitwise_not(img2_head_mask)

                                                    img2_head_noface = cv2.bitwise_and(img2, img2, mask=img2_face_mask)
                                                    result = cv2.add(img2_head_noface, img2_new_face)

                                                    (x, y, w, h) = cv2.boundingRect(convexhull2)
                                                    center_face2 = (int((x + x + w) / 2), int((y + y + h) / 2))

                                                    seamlessclone = cv2.seamlessClone(result, img2, img2_head_mask, center_face2, cv2.MIXED_CLONE)
                                                    frame[bb[i][1]-10 : bb[i][3]+20, bb[i][0]-10: bb[i][2]+20]=seamlessclone
                                                    cv2.imshow("result", result)
                                            except Exception as e:
                                                print(e)
                                                pass




                                    # cv2.putText(frame, class_names[best_class_indices[0]], (text_x, text_y),cv2.FONT_HERSHEY_COMPLEX_SMALL,1, (0, 0, 255), thickness=1, lineType=2)
                                    # person_detected[best_name] += 1

                                # total_frames_passed += 1
                                # if total_frames_passed == 2:
                            for person, count in person_detected.items():
                                if count > 4:
                                    print("Person Detected: {}, Count: {}".format(person, count))
                                    people_detected.add(person)
                                # person_detected.clear()
                                total_frames_passed = 0

                            # cv2.putText(frame, "People detected so far:", (20, 20), cv2.FONT_HERSHEY_PLAIN,
                            #             1, (255, 0, 0), thickness=1, lineType=2)
                            currentYIndex = 40
                            for idx, name in enumerate(people_detected):
                                cv2.putText(frame, name, (20, currentYIndex + 20 * idx), cv2.FONT_HERSHEY_PLAIN,
                                            1, (0, 0, 255), thickness=1, lineType=2)
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

