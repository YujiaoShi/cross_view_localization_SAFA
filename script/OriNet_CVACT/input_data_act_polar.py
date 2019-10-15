import cv2
import random
import numpy as np
import scipy.io as sio

class InputData:

    img_root = '../Data/CVACT/'

    posDistThr = 25
    posDistSqThr = posDistThr*posDistThr

    panoRows = 112

    panoCols = 616

    satSize = 256

    def __init__(self, polar):
        self.polar = polar

        self.allDataList = './OriNet_CVACT/CVACT_orientations/ACT_data.mat'
        print('InputData::__init__: load %s' % self.allDataList)

        self.__cur_allid = 0  # for training
        self.id_alllist = []
        self.id_idx_alllist = []

        # load the mat

        anuData = sio.loadmat(self.allDataList)

        idx = 0
        for i in range(0,len(anuData['panoIds'])):
            grd_id_ori = self.img_root + '_' + anuData['panoIds'][i] + '/' + anuData['panoIds'][i] + '_zoom_2.jpg'
            grd_id_align = self.img_root + 'streetview/' + anuData['panoIds'][i] + '_grdView.png'
            grd_id_ori_sem = self.img_root + '_' + anuData['panoIds'][i] + '/' + anuData['panoIds'][i] + '_zoom_2_sem.jpg'
            grd_id_align_sem = self.img_root + '_' + anuData['panoIds'][i] + '/' + anuData['panoIds'][i] + '_zoom_2_aligned_sem.jpg'
            if self.polar:
                sat_id_ori = self.img_root + 'polarmap/' + anuData['panoIds'][i] + '_satView_polish.png'
            else:
                sat_id_ori = self.img_root + 'satview_polish/' + anuData['panoIds'][i] + '_satView_polish.png'
            sat_id_sem = self.img_root + '_' + anuData['panoIds'][i] + '/' + anuData['panoIds'][i] + '_satView_sem.jpg'
            self.id_alllist.append([grd_id_ori, grd_id_align, grd_id_ori_sem, grd_id_align_sem, sat_id_ori, sat_id_sem, anuData['utm'][i][0], anuData['utm'][i][1]])
            self.id_idx_alllist.append(idx)
            idx += 1
        self.all_data_size = len(self.id_alllist)
        print('InputData::__init__: load', self.allDataList, ' data_size =', self.all_data_size)


        # partion the images into cells

        self.utms_all = np.zeros([2, self.all_data_size], dtype = np.float32)
        for i in range(0, self.all_data_size):
            self.utms_all[0, i] = self.id_alllist[i][6]
            self.utms_all[1, i] = self.id_alllist[i][7]

        self.training_inds = anuData['trainSet']['trainInd'][0][0] - 1


        self.trainNum = len(self.training_inds)

        self.trainList = []
        self.trainIdList = []
        self.trainUTM = np.zeros([2, self.trainNum], dtype = np.float32)
        for k in range(self.trainNum):
            self.trainList.append(self.id_alllist[self.training_inds[k][0]])
            self.trainUTM[:,k] = self.utms_all[:,self.training_inds[k][0]]
            self.trainIdList.append(k)

        self.__cur_id = 0  # for training

        self.val_inds = anuData['valSet']['valInd'][0][0] - 1
        self.valNum = len(self.val_inds)

        self.valList = []
        self.valUTM = np.zeros([2, self.valNum], dtype=np.float32)
        for k in range(self.valNum):
            self.valList.append(self.id_alllist[self.val_inds[k][0]])
            self.valUTM[:, k] = self.utms_all[:, self.val_inds[k][0]]
        # cur validation index
        self.__cur_test_id = 0


    def next_batch_scan(self, batch_size):
        if self.__cur_test_id >= self.valNum:
            self.__cur_test_id = 0
            return None, None, None
        elif self.__cur_test_id + batch_size >= self.valNum:
            batch_size = self.valNum - self.__cur_test_id

        if self.polar:
            batch_sat = np.zeros([batch_size, 112, 616, 3], dtype=np.float32)
        else:
            batch_sat = np.zeros([batch_size, 256, 256, 3], dtype=np.float32)

        batch_grd = np.zeros([batch_size, 112, 616, 3], dtype=np.float32)

        # the utm coordinates are used to define the positive sample and negative samples
        batch_utm = np.zeros([batch_size, 2], dtype=np.float32)
        batch_dis_utm = np.zeros([batch_size, batch_size,1], dtype=np.float32)
        for i in range(batch_size):
            img_idx = self.__cur_test_id + i

            # satellite
            img = cv2.imread(self.valList[img_idx][4])
            if self.polar:
                if img is None or img.shape[0] != self.panoRows or img.shape[1] != self.panoCols:
                    print('InputData::next_pair_batch: read fail: %s, %d, ' % (self.valList[img_idx][4], i))
                    continue
            else:
                if img is None or img.shape[0] != img.shape[1]:
                    print('InputData::next_pair_batch: read fail: %s, %d, ' % (self.valList[img_idx][4], i))
                    continue

            img = img.astype(np.float32)

            img[:, :, 0] -= 103.939  # Blue
            img[:, :, 1] -= 116.779  # Green
            img[:, :, 2] -= 123.6  # Red
            batch_sat[i, :, :, :] = img

            # ground
            img = cv2.imread(self.valList[img_idx][1])

            if img is None:
                print('InputData::next_pair_batch: read fail: %s, %d, ' % (self.valList[img_idx][2], i))
                continue
            img = cv2.resize(img, (616, 112), interpolation=cv2.INTER_AREA)

            img = img.astype(np.float32)

            img[:, :, 0] -= 103.939  # Blue
            img[:, :, 1] -= 116.779  # Green
            img[:, :, 2] -= 123.6  # Red
            batch_grd[i, :, :, :] = img

            batch_utm[i,0] = self.valUTM[0, img_idx]
            batch_utm[i, 1] = self.valUTM[1, img_idx]

        self.__cur_test_id += batch_size

        # compute the batch gps distance
        for ih in range(batch_size):
            for jh in range(batch_size):
                batch_dis_utm[ih,jh,0] = (batch_utm[ih,0] - batch_utm[jh,0])*(batch_utm[ih,0] - batch_utm[jh,0]) + (batch_utm[ih, 1] - batch_utm[jh, 1]) * (batch_utm[ih, 1] - batch_utm[jh, 1])

        return batch_sat, batch_grd, batch_dis_utm


    def next_pair_batch(self, batch_size):
        if self.__cur_id == 0:
            for i in range(20):
                random.shuffle(self.trainIdList)

        if self.__cur_id + batch_size + 2 >= self.trainNum:
            self.__cur_id = 0
            return None, None, None

        if self.polar:
            batch_sat = np.zeros([batch_size, 112, 616, 3], dtype=np.float32)
        else:
            batch_sat = np.zeros([batch_size, 256, 256, 3], dtype=np.float32)

        batch_grd = np.zeros([batch_size, 112, 616, 3], dtype=np.float32)

        # the utm coordinates are used to define the positive sample and negative samples
        batch_utm = np.zeros([batch_size, 2], dtype=np.float32)
        batch_dis_utm = np.zeros([batch_size, batch_size, 1], dtype=np.float32)

        i = 0
        batch_idx = 0
        while True:
            if batch_idx >= batch_size or self.__cur_id + i >= self.trainNum:
                break

            img_idx = self.trainIdList[self.__cur_id + i]
            i += 1

            # satellite
            img = cv2.imread(self.trainList[img_idx][4])

            if img is None:
                print('InputData::next_pair_batch: read fail: %s, %d, ' % (self.trainList[img_idx][4], i))
                continue

            img = img.astype(np.float32)

            # normalize it to -1 --- 1
            img[:, :, 0] -= 103.939  # Blue
            img[:, :, 1] -= 116.779  # Green
            img[:, :, 2] -= 123.6  # Red

            batch_sat[batch_idx, :, :, :] = img

            # ground
            img = cv2.imread(self.trainList[img_idx][1])

            if img is None:
                print('InputData::next_pair_batch: read fail: %s, %d, ' % (self.trainList[img_idx][1], i))
                continue
            img = cv2.resize(img, (616, 112), interpolation=cv2.INTER_AREA)

            img = img.astype(np.float32)

            img[:, :, 0] -= 103.939  # Blue
            img[:, :, 1] -= 116.779  # Green
            img[:, :, 2] -= 123.6  # Red
            batch_grd[batch_idx, :, :, :] = img

            batch_utm[batch_idx, 0] = self.trainUTM[0, img_idx]
            batch_utm[batch_idx, 1] = self.trainUTM[1, img_idx]

            batch_idx += 1

        # compute the batch gps distance
        for ih in range(batch_size):
            for jh in range(batch_size):
                batch_dis_utm[ih,jh,0] = (batch_utm[ih,0] - batch_utm[jh,0])*(batch_utm[ih,0] - batch_utm[jh,0]) + (batch_utm[ih, 1] - batch_utm[jh, 1]) * (batch_utm[ih, 1] - batch_utm[jh, 1])



        self.__cur_id += i

        return batch_sat, batch_grd, batch_dis_utm

    def get_dataset_size(self):
        return self.trainNum

    def get_test_dataset_size(self):
        return self.valNum

    def reset_scan(self):
        self.__cur_test_id = 0


if __name__ == '__main__':
    input_data = InputData()
    batch_sat, batch_grd, batch_utm = input_data.next_batch_scan(12)