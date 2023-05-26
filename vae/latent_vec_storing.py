import numpy as np
import keras
import csv
import cv2
import os
from numpy import linspace

def latent_interpolate(vec1, vec2, inter_steps=5):
    inter_vectors = []
    ratios = linspace(0 , 1, num=inter_steps)
    for ratio in ratios:
        vec = (1.0 - ratio) * vec1 + ratio * vec2
        inter_vectors.append(vec)
    return np.asarray(inter_vectors)

def latent_clustering(attribute_num):
    latent_clusters = []
    young_count = 0
    middleAge_count = 0
    senior_count = 0
    young_latent = np.zeros((1,256))
    middleAge_latent = np.zeros((1,256))
    senior_latent = np.zeros((1,256))

    latent_vec_path = "vae\latent.csv"
    data_latent = np.genfromtxt(latent_vec_path, dtype=None, delimiter=',', names=True, encoding='UTF-8')

    # getting the  vectors of age (young, middle-aged, and senior)
    for i in range(data_latent.shape[0]):
        if (data_latent[i][2] == 1):
            young_latent += data_latent[i][48]
            young_count += 1

        elif (data_latent[i][3] == 1):
            middleAge_latent += data_latent[i][48]
            middleAge_count += 1

        elif (data_latent[i][4] == 1):
            senior_latent += data_latent[i][48]
            senior_count += 1

    young_latent /= young_count
    latent_clusters.append(young_latent)
    middleAge_latent /= middleAge_count
    latent_clusters.append(middleAge_latent)
    senior_latent /= senior_count
    latent_clusters.append(senior_latent)

    return 
    # getting the vectors of biological sex (male/ female)
    for i in range(data_latent.shape[0]):
        if (data_latent[i][1])
    # getting the vectors of eyewear (no eyewear, eyeglasses)
    # getting the vector of facial hair (no beard, mustache, goatee, beard)
    # getting the vector of face shape (oval, round, square)
    # getting the vector of lipstick (yes, no)
    return

IMAGE_SIZE = 256
cols_names =[]
out_row = []
dataset_folder = "vae\data"
Metadata_file_path ="vae\Metadata.csv"

encoder = keras.models.load_model("vae\encoder.hd5")

Metadata = np.genfromtxt(Metadata_file_path, dtype=None, delimiter=',', names=True, encoding='UTF-8', max_rows=10)
for j in range(49):
    if j != 1:
        cols_names.append(Metadata.dtype.names[j])
cols_names.append('latent_vector')
with open("vae\latent.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(cols_names)
    for i in range(Metadata.shape[0]):
        out_row = [Metadata[i][0]]
        for j in range(2,49):
            out_row.append(Metadata[i][j])
        img = cv2.resize(cv2.imread(os.path.join(dataset_folder, Metadata[i][0])), (IMAGE_SIZE, IMAGE_SIZE))
        img = img.astype(np.float32) / 128 - 1
        mean, _ = encoder(img.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 3))
        out_row.append(np.asarray(mean))
        writer.writerow(out_row)
f.close()


        

            





