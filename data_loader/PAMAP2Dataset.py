import torch.utils.data
import pandas as pd
import time
import numpy as np
import itertools
import sys

sys.path.append('..')
import conf

opt = conf.PAMAP2Opt
WIN_LEN = opt['seq_len']

class PAMAP2Dataset(torch.utils.data.Dataset):

    def __init__(self, file, transform=None, user=None, position=None, activity=None, complementary=False):
        """
        Args:
            file_path (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            users: condition on user
            positions: condition on position of sensor
            activities: condition on action

            file shape: ['Activity', 'xacc', 'yacc', 'zacc', 'xgyro', 'ygyro', 'zgyro', 'xmag', 'ymag', 'zmag', 'User', 'Position', 'Activity']
        """
        st = time.time()
        self.user = user
        self.position = position
        self.activity = activity
        self.complementary = complementary

        self.df = pd.read_csv(file)

        if complementary:
            if user:
                self.df = self.df[self.df['User'] != user]
            if position:
                self.df = self.df[self.df['Position'] != position]
            if activity:
                self.df = self.df[self.df['Activity'] != activity]
        else:
            if user:
                self.df = self.df[self.df['User'] == user]
            if position:
                self.df = self.df[self.df['Position'] == position]
            if activity:
                self.df = self.df[self.df['Activity'] == activity]

        print(len(self.df))

        self.transform = transform
        ppt = time.time()
        self.preprocessing()
        print('Loading data done with rows:{:d}\tPreprocessing:{:f}\tTotal Time:{:f}'.format(len(self.df.index),
                                                                                             time.time() - ppt,
                                                                                             time.time() - st))

    def preprocessing(self):
        self.num_domains = 0
        self.features = []
        self.class_labels = []
        self.domain_labels = []

        self.datasets = [] # list of dataset per each domain

        if self.complementary:
            positions = set(conf.PAMAP2Opt['positions']) - set(self.position)
        else:
            positions = set([self.position])

        domain_superset = list(positions)
        valid_domains = []

        for idx in range(max(len(self.df)//WIN_LEN, 0)):
            user = self.df.iloc[idx * WIN_LEN, 9]
            position = self.df.iloc[idx * WIN_LEN, 10]
            class_label = self.df.iloc[idx * WIN_LEN, 11]
            domain_label = -1

            for i in range(len(domain_superset)):
                if domain_superset[i] == position and domain_superset[i] not in valid_domains:
                    valid_domains.append(domain_superset[i])
                    break
            if position in valid_domains:
                domain_label = valid_domains.index(position)
            else:
                continue

            feature = self.df.iloc[idx * WIN_LEN:(idx + 1) * WIN_LEN, 0:9].values
            feature = feature.T

            self.features.append(feature)
            self.class_labels.append(self.class_to_number(class_label))
            self.domain_labels.append(domain_label)

        self.num_domains = len(valid_domains)
        self.features = np.array(self.features, dtype=np.float)
        self.class_labels = np.array(self.class_labels)
        self.domain_labels = np.array(self.domain_labels)


        # append dataset for each domain
        for domain_idx in range(self.num_domains):
            indices = np.where(self.domain_labels == domain_idx)[0]
            self.datasets.append(torch.utils.data.TensorDataset(torch.from_numpy(self.features[indices]).float(),
                                                                torch.from_numpy(self.class_labels[indices]),
                                                                torch.from_numpy(self.domain_labels[indices])))

        self.dataset = torch.utils.data.ConcatDataset(self.datasets)
        print('Valid domains: ' + str(valid_domains))


    def __len__(self):
        return len(self.dataset)

    def get_num_domains(self):
        return self.num_domains

    def get_datasets_per_domain(self):
        # todo: not implemented currently => separate domains is not used in the code
        return self.datasets

    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.item()

        return self.dataset[idx]

    def class_to_number(self, label):
        # only 8 classes that will be used in our settings
        dic = {'lying': 0,
               'sitting': 1,
               'standing': 2,
               'walking': 3,
               'ascending_stairs': 4,
               'descending_stairs': 5,
               'vacuum_cleaning': 6,
               'ironing': 7,
               }

        return dic[label]

if __name__ == '__main__':
    pass

