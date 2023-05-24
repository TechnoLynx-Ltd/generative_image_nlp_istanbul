import numpy as np
import cv2
import os


class DataLoader:
    def __init__(self, data_dir, image_size, batch_size):
        self.data_dir = data_dir
        self.data_files = np.array([os.path.join(data_dir, file) for file in os.listdir(data_dir)])
        self.batched_data_files = []
        self.image_size = image_size
        self.batch_size = batch_size
        self.index = 0
        self.num_batches = -1
        self.data_end = True

    def batch_data_files(self):
        self.batched_data_files = []
        for i in range(self.data_files.shape[0] // self.batch_size):
            self.batched_data_files.append(self.data_files[i * self.batch_size:(i + 1) * self.batch_size])
        if self.data_files.shape[0] % self.batch_size != 0:
            self.batched_data_files.append(self.data_files[-(self.data_files.shape[0] % self.batch_size):])

    def init_dataloader(self):
        np.random.shuffle(self.data_files)
        self.batch_data_files()
        self.num_batches = len(self.batched_data_files)
        self.index = 0
        self.data_end = False

    def load_next_batch(self):
        out = []
        for file in self.batched_data_files[self.index]:
            img = cv2.imread(file)
            img = cv2.resize(img, (self.image_size, self.image_size))
            img = img.astype(np.float32) / 128 - 1
            # TODO: augmentation
            out.append(img)

        self.index += 1
        if self.index == self.num_batches:
            self.index = -1
            self.data_end = True

        return np.array(out)

    def load_random_batches(self, n):
        batch_inds = np.random.choice(np.arange(self.num_batches, dtype=int), size=n, replace=False)
        out = []
        for i in range(n):
            img_batch = []
            for file in self.batched_data_files[batch_inds[i]]:
                img = cv2.imread(file)
                img = cv2.resize(img, (self.image_size, self.image_size))
                img = img.astype(np.float32) / 128 - 1
                img_batch.append(img)
            out.append(np.array(img_batch))
        return out

    def end_of_data(self):
        return self.data_end
