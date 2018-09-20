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

    def _get_pos_sample(self, target, index):
        pos_index = np.argwhere(self.targets == target)
        pos_index = pos_index.flatten()
        pos_index = np.setdiff1d(pos_index, index)
        rand = np.random.permutation(len(pos_index))
        result_path = []
        for i in range(4):
           t = i%len(rand)
           tmp_index = pos_index[rand[t]]
           result_path.append(self.samples[tmp_index][0])
        return result_path

    def _get_neg_sample(self, target):
        neg_index = np.argwhere(self.targets != target)
        neg_index = neg_index.flatten()
        rand = random.randint(0,len(neg_index)-1)
        return self.samples[neg_index[rand]]

    def __getitem__(self, index):
        path, target = self.samples[index]
        cam = self.cams[index]
        # pos_path, neg_path
        pos_path = self._get_pos_sample(target, index)

        sample = self.loader(path)
        pos0 = self.loader(pos_path[0])
        pos1 = self.loader(pos_path[1])
        pos2 = self.loader(pos_path[2])
        pos3 = self.loader(pos_path[3])

        if self.transform is not None:
            sample = self.transform(sample)
            pos0 = self.transform(pos0)
            pos1 = self.transform(pos1)
            pos2 = self.transform(pos2)
            pos3 = self.transform(pos3)

        if self.target_transform is not None:
            target = self.target_transform(target)

        c,h,w = pos0.shape
        pos = torch.cat((pos0.view(1,c,h,w), pos1.view(1,c,h,w), pos2.view(1,c,h,w), pos3.view(1,c,h,w)), 0)
        pos_target = target
        return sample, target, pos, pos_target
