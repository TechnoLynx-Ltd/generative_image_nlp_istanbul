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


IMAGE_SIZE = 256
cols_names =[]
out_row = []
# male_count = 0
# male_latent = np.zeros((1,256))
young_count = 0
young_latent = np.zeros((1,256))
middleAge_count = 0
middleAge_latent = np.zeros((1,256))
senior_count = 0
senior_latent = np.zeros((1,256))

dataset_folder = "vae\data"
Metadata_file_path ="vae\Metadata.csv"

encoder = keras.models.load_model("vae\encoder.hd5")

Metadata = np.genfromtxt(Metadata_file_path, dtype=None, delimiter=',', names=True, encoding='UTF-8')
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
        # calculating the avg latent for features
        # if (Metadata[i][2] == 1):
        #     male_latent += np.asarray(mean)
        #     male_count += 1
        # if (Metadata[i][2] == -1):
        #     female_latent += np.asarray(mean)
        #     female_count += 1
        if (Metadata[i][3] == 1):
            young_latent += np.asarray(mean)
            young_count += 1
        if (Metadata[i][4] == 1):
            middleAge_latent += np.asarray(mean)
            middleAge_count += 1
        if (Metadata[i][5] == 1):
            senior_latent += np.asarray(mean)
            senior_count += 1
f.close()
young_latent /= young_count
middleAge_latent /= middleAge_count
senior_latent /= senior_count
with open("vae\latent_avg.csv", "w", newline="") as l:
    writer_l = csv.writer(l)
    writer_l.writerow(young_latent)
    writer_l.writerow(middleAge_latent)
    writer_l.writerow(senior_latent)
        

            





