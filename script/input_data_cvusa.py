import cv2
import random
import numpy as np

class InputData:

    img_root = '../Data/CVUSA/'


    def __init__(self, polar):
        self.polar = polar

        self.train_list = self.img_root + 'splits/train-19zl.csv'
        self.test_list = self.img_root + 'splits/val-19zl.csv'

        print('InputData::__init__: load %s' % self.train_list)
        self.__cur_id = 0  # for training
        self.id_list = []
        self.id_idx_list = []
        with open(self.train_list, 'r') as file:
            idx = 0
            for line in file:
                data = line.split(',')
                pano_id = (data[0].split('/')[-1]).split('.')[0]
                # satellite filename, streetview filename, pano_id
                if self.polar:
                    item1 = data[0].replace('bing', 'polar').replace('jpg', 'png')
                else:
                    item1 = data[0]

                item2 = data[1]

                self.id_list.append([item1, item2, pano_id])
                self.id_idx_list.append(idx)
                idx += 1
        self.data_size = len(self.id_list)
        print('InputData::__init__: load', self.train_list, ' data_size =', self.data_size)

        print('InputData::__init__: load %s' % self.test_list)
        self.__cur_test_id = 0  # for training
        self.id_test_list = []
        self.id_test_idx_list = []
        with open(self.test_list, 'r') as file:
            idx = 0
            for line in file:
                data = line.split(',')
                pano_id = (data[0].split('/')[-1]).split('.')[0]
                # satellite filename, streetview filename, pano_id
                if self.polar:
                    item1 = data[0].replace('bing', 'polar').replace('jpg', 'png')
                else:
                    item1 = data[0]

                item2 = data[1]

                self.id_test_list.append([item1, item2, pano_id])
                self.id_test_idx_list.append(idx)
                idx += 1
        self.test_data_size = len(self.id_test_list)
        print('InputData::__init__: load', self.test_list, ' data_size =', self.test_data_size)




    def next_batch_scan(self, batch_size):
        if self.__cur_test_id >= self.test_data_size:
            self.__cur_test_id = 0
            return None, None
        elif self.__cur_test_id + batch_size >= self.test_data_size:
            batch_size = self.test_data_size - self.__cur_test_id

        if self.polar:
            batch_sat = np.zeros([batch_size, 112, 616, 3], dtype=np.float32)
        else:
            batch_sat = np.zeros([batch_size, 256, 256, 3], dtype=np.float32)

        batch_grd = np.zeros([batch_size, 112, 616, 3], dtype=np.float32)

        for i in range(batch_size):
            img_idx = self.__cur_test_id + i
            
            if batch_size == 1:
                print(self.id_test_list[img_idx][0])

            # satellite
            img = cv2.imread(self.img_root + self.id_test_list[img_idx][0])
            if not self.polar:
                img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)

            img_dup = img.copy()

            img_dup = img_dup.astype(np.float32)
            # img -= 100.0
            img_dup[:, :, 0] -= 103.939  # Blue
            img_dup[:, :, 1] -= 116.779  # Green
            img_dup[:, :, 2] -= 123.6  # Red
            batch_sat[i, :, :, :] = img_dup

            # ground
            img = cv2.imread(self.img_root + self.id_test_list[img_idx][1])
            img = cv2.resize(img, (616, 112), interpolation=cv2.INTER_AREA)

            img_dup = img.copy()

            img_dup = img_dup.astype(np.float32)
            # img -= 100.0
            img_dup[:, :, 0] -= 103.939  # Blue
            img_dup[:, :, 1] -= 116.779  # Green
            img_dup[:, :, 2] -= 123.6  # Red

            batch_grd[i, :, :, :] = img_dup

        self.__cur_test_id += batch_size

        return batch_sat, batch_grd


    def next_pair_batch(self, batch_size):
        if self.__cur_id == 0:
            for i in range(20):
                random.shuffle(self.id_idx_list)

        if self.__cur_id + batch_size + 2 >= self.data_size:
            self.__cur_id = 0
            return None, None

        if self.polar:
            batch_sat = np.zeros([batch_size, 112, 616, 3], dtype=np.float32)
        else:
            batch_sat = np.zeros([batch_size, 256, 256, 3], dtype=np.float32)

        batch_grd = np.zeros([batch_size, 112, 616, 3], dtype=np.float32)

        i = 0
        batch_idx = 0
        while True:
            if batch_idx >= batch_size or self.__cur_id + i >= self.data_size:
                break

            img_idx = self.id_idx_list[self.__cur_id + i]
            i += 1

            # satellite
            img = cv2.imread(self.img_root + self.id_list[img_idx][0])

            if not self.polar:
                if img is None:
                    print('InputData::next_pair_batch: read fail: %s, %d, ' % (self.img_root + self.id_list[img_idx][0], i), img.shape)
                    continue

                img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)

            else:
                if img is None:
                    print('InputData::next_pair_batch: read fail: %s, %d, ' % (self.img_root + self.id_list[img_idx][0], i), img.shape)
                    continue

            img = img.astype(np.float32)
            # img -= 100.0
            img[:, :, 0] -= 103.939  # Blue
            img[:, :, 1] -= 116.779  # Green
            img[:, :, 2] -= 123.6    # Red
            batch_sat[batch_idx, :, :, :] = img


            # ground
            img = cv2.imread(self.img_root + self.id_list[img_idx][1])

            if img is None or img.shape[0] != 224 or img.shape[1] != 1232:
                print('InputData::next_pair_batch: read fail: %s, %d, ' % (self.img_root + self.id_list[img_idx][1], i), img.shape)
                continue
            img = cv2.resize(img, (616, 112), interpolation=cv2.INTER_AREA)

            img = img.astype(np.float32)

            img[:, :, 0] -= 103.939  # Blue
            img[:, :, 1] -= 116.779  # Green
            img[:, :, 2] -= 123.6  # Red
            batch_grd[batch_idx, :, :, :] = img

            batch_idx += 1

        self.__cur_id += i

        return batch_sat, batch_grd


    def get_dataset_size(self):
        return self.data_size

    def get_test_dataset_size(self):
        return self.test_data_size

    def reset_scan(self):
        self.__cur_test_idd = 0
        
    def get_test_list(self):
        return self.id_test_list
    
    def get_train_list(self):
        return self.id_list
    
    def get_test_id(self):
        return self.__cur_test_id

