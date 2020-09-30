from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import mmcv, cv2
import os
from PIL import Image, ImageDraw
import glob
from facenet_pytorch import MTCNN, training
import torch

import tensorflow as tf
import numpy as np
import os
import facenet
import facenet.src.align.detect_face
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import pairwise_distances
import facenet.src.facenet
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))
current_path = os.getcwd()
mtcnn = MTCNN(keep_all=True, device=device)
def test_cluster_distances_with_distance_threshold():
    rng = np.random.RandomState(0)
    n_samples = 100
    X = rng.randint(-10, 10, size=(n_samples, 3))
    # check the distances within the clusters and with other clusters
    distance_threshold = 2200
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        linkage="single").fit(X)
    labels = clustering.labels_
    D = pairwise_distances(X, metric="minkowski", p=2)
    # to avoid taking the 0 diagonal in min()
    np.fill_diagonal(D, np.inf)
    for label in np.unique(labels):
        in_cluster_mask = labels == label
        max_in_cluster_distance = (D[in_cluster_mask][:, in_cluster_mask]
                                   .min(axis=0).max())
        min_out_cluster_distance = (D[in_cluster_mask][:, ~in_cluster_mask]
                                    .min(axis=0).min())
        # single data point clusters only have that inf diagonal here
        if in_cluster_mask.sum() > 1:
            assert max_in_cluster_distance < distance_threshold
            return max_in_cluster_distance

        else:
            assert min_out_cluster_distance >= distance_threshold
            return min_out_cluster_distance


def main(mtcnn):
    a = 0
    b = 0

    with tf.Graph().as_default():
        with tf.compat.v1.Session() as sess:
            #facenet.src.facenet.load_model(args.MTCNN)

            image_list = load_images_from_folder('C:/Users/mmlab/PycharmProjects/UI_pyqt/models/out_dir/')
            images = align_data(image_list, mtcnn.image_size, mtcnn.margin)


            #images_placeholder = sess.graph.get_tensor_by_name("input:0")
            #embeddings = sess.graph.get_tensor_by_name("embeddings:0")
            #phase_train_placeholder = sess.graph.get_tensor_by_name("phase_train:0")
            #feed_dict = {images_placeholder: images, phase_train_placeholder: False}
            #emb = sess.run(embeddings, feed_dict=feed_dict)

            nrof_images = len(images)
            sum_1=0

            matrix = np.zeros((nrof_images, nrof_images))

            print('')
            # Print distance matrix
            print('Distance matrix')
            print('    ', end='')
            for i in range(nrof_images):
                print('    %1d     ' % i, end='')
            print('')
            for i in range(nrof_images):
                print('%1d  ' % i, end='')
                for j in range(nrof_images):
                    dist = np.sqrt(np.sum(np.square(np.subtract(images[i, :], images[j, :]))))
                    matrix[i][j] = dist
                    sum_1 +=dist


                    print('  %1.4f  ' % dist, end='')
                print('')
            #sum_2=sum_1/np.square(len(matrix),len(matrix))
            #print(sum_2)
            print('')
            #threshold=test_cluster_distances_with_distance_threshold()
            # DBSCAN is the only algorithm that doesn't require the number of clusters to be defined.
            db = DBSCAN(metric='precomputed',min_samples=5,eps=2500)
            db.fit(matrix)
            labels = db.labels_

            # get number of clusters
            no_clusters = len(set(labels)) - (1 if -1 in labels else 0)

            print('No of clusters:', no_clusters)

            if no_clusters > 0:
                if max(labels)>100000:
                    largest_cluster = 0
                    for i in range(no_clusters):
                        print('Cluster {}: {}'.format(i, np.nonzero(labels == i)[0]))
                        if len(np.nonzero(labels == i)[0]) > len(np.nonzero(labels == largest_cluster)[0]):
                            largest_cluster = i
                    print('Saving largest cluster (Cluster: {})'.format(largest_cluster))
                    cnt = 1
                    for i in np.nonzero(labels == largest_cluster)[0]:
                        cv2.imwrite(os.path.join('C:/Users/mmlab/PycharmProjects/UI_pyqt/models/clusteringfolder', str(cnt) + '.png'), images[i])
                        cnt += 1
                else:
                    print('Saving all clusters')
                    for i in range(no_clusters):
                        cnt = 1
                        print('Cluster {}: {}'.format(i, np.nonzero(labels == i)[0]))
                        path = os.path.join('C:/Users/mmlab/PycharmProjects/UI_pyqt/models/clusteringfolder', str(i))
                        if not os.path.exists(path):
                            os.makedirs(path)
                            for j in np.nonzero(labels == i)[0]:

                                cv2.imwrite(os.path.join(path, str(cnt) + '.png'), images[j])
                                cnt += 1
                        else:
                            for j in np.nonzero(labels == i)[0]:
                                cv2.imwrite(os.path.join(path, str(cnt) + '.png'), images[j])
                                cnt += 1

    for i in range(0, len(os.listdir(
            'C:/Users/mmlab/PycharmProjects/UI_pyqt/models/clusteringfolder/'))):
        # print(len(os.listdir('C:/Users/hroro/PycharmProjects/facenet-pytorch-master/facenet-pytorch-master/models/clusteringfolder/{}'.format(i))))
        a += len(os.listdir('C:/Users/mmlab/PycharmProjects/UI_pyqt/models/clusteringfolder/{}'.format(i)))
    folder = []
    folder_name = []
    folder_in_file = []
    for i in range(0, len(os.listdir('C:/Users/mmlab/PycharmProjects/UI_pyqt/models/clusteringfolder/'))):
        b = len(os.listdir('C:/Users/mmlab/PycharmProjects/UI_pyqt/models/clusteringfolder/{}'.format(i)))
        d = int(b / a * 100)
        print(d)
        folder.append('C:/Users/mmlab/PycharmProjects/UI_pyqt/models/clusteringfolder/{}'.format(i))
        folder_name.append(i)
    for i in range(len(folder)):
        print(folder[i])
        print(folder_name[i])
        if int(folder_name[i]) > 0:
            file = os.path.join('C:/Users/mmlab/PycharmProjects/UI_pyqt/models/clusteringfolder/{}'.format(int(folder_name[i])), '1.png')
            folder_in_file.append(file)
            # print(folder_in_file[i])
    for i in range(len(folder_in_file)):
        cv = cv2.imread(folder_in_file[i], cv2.IMREAD_COLOR)
        cv2.imwrite('C:/Users/mmlab/PycharmProjects/UI_pyqt/models/model_file/model{}.png'.format(i), cv)

def align_data(image_list, image_size, margin):
    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    image = []
    images=glob.glob('C:/Users/mmlab/PycharmProjects/UI_pyqt/models/test_file/*.jpg')
    print(len(images))
    for i in range(0, len(images)):
        image.append(Image.open(images[i]))
    if len(image) > 0:
            # print((img_list[10]).size)
            # for i in range(len(img_list)):
            # img_list[i]
        image_1 = np.stack(image)
        return image_1
    else:
        return None



def create_network_face_detection(gpu_memory_fraction):
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = facenet.src.align.detect_face.create_mtcnn(sess, None)
    return pnet, rnet, onet


def load_images_from_folder(folder):
    images = []
    image = glob.glob('C:/Users/mmlab/PycharmProjects/UI_pyqt/models/out_dir/*.jpg')
    images = []

    for i in range(1, len(image)):
        images.append(Image.open(image[i]))

    print(len(images))
    return images



if __name__ == '__main__':
    main(mtcnn)
