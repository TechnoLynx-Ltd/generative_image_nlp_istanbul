import numpy as np
import keras
import csv
import cv2
import os

IMAGE_SIZE = 256

cols_names =[]
out_row = []

dataset_folder = "data"
Metadata_file_path ="Metadata.csv"

encoder = keras.models.load_model("encoder.hd5")

Metadata = np.genfromtxt(Metadata_file_path, dtype=None, delimiter=',', names=True, encoding='UTF-8')
for j in range(49):
    if j != 1:
        cols_names.append(Metadata.dtype.names[j])
cols_names.append('latent_vector')
with open("latent.csv", "w", newline="") as f:
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