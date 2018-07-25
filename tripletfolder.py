from torchvision import datasets
import os
import numpy as np
import random
import torch

class TripletFolder(datasets.ImageFolder):

    def __init__(self, root, transform):
        super(TripletFolder, self).__init__(root, transform)
        targets = np.asarray([s[1] for s in self.samples])
        self.targets = targets
        cams = []
        for s in self.samples:
            cams.append( self._get_cam_id(s[0]) )
        self.cams = np.asarray(cams)

    def _get_cam_id(self, path):
        camera_id = []
        filename = os.path.basename(path)
        camera_id = filename.split('c')[1][0]
        #camera_id = filename.split('_')[2][0:2]
        return int(camera_id)-1

    def _get_pos_sample(self, target,index):
        pos_index = np.argwhere(self.targets == target)
        pos_index = pos_index.flatten()
        rand = random.randint(0,len(pos_index)-1)
        if pos_index[rand] == index:
            rand = random.randint(0,len(pos_index)-1)
        return self.samples[pos_index[rand]]

    def _get_neg_sample(self, target):
        neg_index = np.argwhere(self.targets != target)
        neg_index = neg_index.flatten()
        rand = random.randint(0,len(neg_index)-1)
        return self.samples[neg_index[rand]]

    def __getitem__(self, index):
        path, target = self.samples[index]
        # cam = self.cams[index]
        # pos_path, neg_path
        pos_path, pos_target = self._get_pos_sample(target, index)
        neg_path, neg_target = self._get_neg_sample(target)
        neg_path2, neg_target2 = self._get_neg_sample(target)

        sample = self.loader(path)
        pos = self.loader(pos_path)
        neg = self.loader(neg_path)
        neg2 = self.loader(neg_path2)

        if self.transform is not None:
            sample = self.transform(sample)
            pos = self.transform(pos)
            neg = self.transform(neg)
            neg2 = self.transform(neg2)

        if self.target_transform is not None:
            target = self.target_transform(target)
            pos_target = self.target_transform(pos_target)
            neg_target = self.target_transform(neg_target)
            neg_target2 = self.target_transform(neg_target2)

        return sample, target, pos, pos_target, neg, neg_target, neg2, neg_target2
