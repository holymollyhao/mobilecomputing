import torch.utils.data
import io, os
import pandas as pd
import time
import numpy as np
import sys
import pickle
from itertools import islice

import multiprocessing as mp

manager = mp.Manager()

sys.path.append('..')
import conf

from sklearn.preprocessing import MinMaxScaler
import itertools

opt = conf.WESADOpt
WIN_LEN = opt['seq_len']

filePath = '../dataset/wesad/'


class KSHOTTensorDataset(torch.utils.data.Dataset):
    # class for MAML
    def __init__(self, num_classes, features, classes, domains):
        assert (features.shape[0] == classes.shape[0] == domains.shape[0])

        self.num_classes = num_classes
        self.features_per_class = []
        self.classes_per_class = []
        self.domains_per_class = []

        for class_idx in range(self.num_classes):
            indices = np.where(classes == class_idx)
            self.features_per_class.append(np.random.permutation(features[indices]))
            self.classes_per_class.append(np.random.permutation(classes[indices]))
            self.domains_per_class.append(np.random.permutation(domains[indices]))

        self.data_num = min(
            [len(feature_per_class) for feature_per_class in self.features_per_class])  # get min number of classes

        for i in range(self.num_classes):
            self.features_per_class[i] = torch.from_numpy(self.features_per_class[i][:self.data_num]).float()
            self.classes_per_class[i] = torch.from_numpy(self.classes_per_class[i][:self.data_num])
            self.domains_per_class[i] = torch.from_numpy(self.domains_per_class[i][:self.data_num])

    def __getitem__(self, index):

        features = torch.FloatTensor(self.num_classes, *(
            self.features_per_class[0][0].shape))  # make FloatTensor with shape num_classes x F-dim1 x F-dim2...
        classes = torch.LongTensor(self.num_classes)
        domains = torch.LongTensor(self.num_classes)

        rand_indices = [i for i in range(self.num_classes)]
        np.random.shuffle(rand_indices)

        for i in range(self.num_classes):
            features[i] = self.features_per_class[rand_indices[i]][index]
            classes[i] = self.classes_per_class[rand_indices[i]][index]
            domains[i] = self.domains_per_class[rand_indices[i]][index]

        # print('features, classes, domains : ')
        # print(features)
        # print(classes)
        # print(domains)
        return (features, classes, domains)

    def __len__(self):
        return self.data_num


class WESADDataset(torch.utils.data.Dataset):
    # load static files

    def __init__(self, file='./dataset/wesad/both_all.csv', domains=None, labels=None, get_calculated_features=False,
                 max_source=100):
        st = time.time()
        self.domains = domains
        self.activity = labels
        self.max_source = max_source

        self.df = pd.read_csv(file)

        if domains is not None:
            cond_list = []
            for d in domains:
                cond_list.append('domain == "{:s}"'.format(d))
            cond_str = ' | '.join(cond_list)
            self.df = self.df.query(cond_str)

        if labels is not None:
            cond_list = []
            for d in labels:
                cond_list.append('label == "{:s}"'.format(d))
            cond_str = ' | '.join(cond_list)
            self.df = self.df.query(cond_str)

        ppt = time.time()

        self.preprocessing()
        print('Loading data done with rows:{:d}\tPreprocessing:{:f}\tTotal Time:{:f}'.format(len(self.df.index),
                                                                                             time.time() - ppt,
                                                                                             time.time() - st))

    def preprocessing(self):
        self.features = []
        self.class_labels = []
        self.domain_labels = []

        self.datasets = []  # list of dataset per each domain
        self.kshot_datasets = []  # list of dataset per each domain

        for idx in range(max(len(self.df) // WIN_LEN, 0)):
            domain = self.df.iloc[idx * WIN_LEN, 11]
            class_label = self.df.iloc[idx * WIN_LEN, 10]
            domain_label = self.domains.index(domain)

            feature = self.df.iloc[idx * WIN_LEN:(idx + 1) * WIN_LEN, 0:10].values
            feature = feature.T

            self.features.append(feature)
            self.class_labels.append(self.class_to_number(class_label))
            self.domain_labels.append(domain_label)

        self.features = np.array(self.features, dtype=np.float)
        self.class_labels = np.array(self.class_labels)
        self.domain_labels = np.array(self.domain_labels)

        # append dataset for each domain
        for domain_idx in range(self.get_num_domains()):
            indices = np.where(self.domain_labels == domain_idx)[0]
            self.datasets.append(torch.utils.data.TensorDataset(torch.from_numpy(self.features[indices]).float(),
                                                                torch.from_numpy(self.class_labels[indices]),
                                                                torch.from_numpy(self.domain_labels[indices])))
            kshot_dataset = KSHOTTensorDataset(len(np.unique(self.class_labels)),
                                               self.features[indices],
                                               self.class_labels[indices],
                                               self.domain_labels[indices])
            self.kshot_datasets.append(kshot_dataset)

            # print("i:{:d}, len:{:d}".format(domain_idx, len(kshot_dataset))) # print number of available shots per domain
        # concated dataset
        self.dataset = torch.utils.data.ConcatDataset(self.datasets)

    def __len__(self):
        # return max(len(self.df) // OVERLAPPING_WIN_LEN - 1, 0)
        return len(self.dataset)

    def get_num_domains(self):
        return len(self.domains)

    def get_datasets_per_domain(self):
        return self.kshot_datasets

    def class_to_number(self, label):
        # 'baseline','stress','amusement','mediation'
        dic = {'baseline': 0,
               'stress': 1,
               'amusement': 2,
               'mediation': 3,
               }
        return dic[label]

    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.item()

        return self.dataset[idx]


def getColumnNameDict(df):
    columnNameDict = {}

    for i in range(len(list(df))):
        columnNameDict[i] = list(df)[i]
    return columnNameDict


def minmaxScaling(df):
    scaler = MinMaxScaler()
    df[df.columns] = scaler.fit_transform(df[df.columns])
    '''
    columnName = getColumnNameDict(df)
    resultDf = df.copy().reset_index(drop=True)
    for i in range(len(df.columns)):
        data = df.iloc[:, i].values.reshape(-1, 1)

        if len(data) <=1:
            continue
        print('------------------')
        print('i는 ')
        print(i)
        print(data)
        scaler.fit(data)
        data = scaler.transform(data)

        newDf = pd.DataFrame({columnName[i]: data[:,0]})

        resultDf = resultDf.drop(columnName[i], axis=1)
        resultDf.insert(loc=i, column=columnName[i], value=newDf)
    '''
    # print(df)
    return df


def sampling(df):
    columnName = getColumnNameDict(df)
    sampledDf = pd.DataFrame(columns=columnName.values())
    i = 0
    while 1:
        tmpLabel = df["label"][i]
        for j in range(1, WIN_LEN):
            if i + j >= len(df):
                break
            if tmpLabel != df["label"][i + j]:
                i += j - 1  # break -> i+=1
                break
            if j is WIN_LEN - 1:
                newDf = df.iloc[i:i + WIN_LEN, :]
                newDf = newDf.reset_index(drop=True)
                sampledDf = sampledDf.append(newDf, ignore_index=True)
                # sampledDf.insert(loc= sampleNum*16, column = columnName, value=newDf)
                i += WIN_LEN - 1  # -> i+=1
        i += 1
        if i >= len(df): break

    return sampledDf


def initial_preprocessing_extract_required_data(file_path):
    domains = opt['domains']
    # filePath = "../dataset/wesad/"
    filePath = '../dataset/wesad/'
    currentExtractedData = pd.read_csv(file_path + 'both_all.csv')
    '''
    # LabelList = ['LYING_DOWN', 'SITTING', 'FIX_WALKING',]
    df = pd.DataFrame(columns=deformedColumns)
    extractedData = pd.DataFrame(columns=colDomLabelList)
    for domain in domains:
        currentExtractedData = pd.read_csv(file_path + domain + '.features_labels.csv', usecols= colLabelList)
        # Insert domain
        currentDoamin = [domain for i in range(len(currentExtractedData))]
        currentExtractedData.insert(loc=len(colList), column='domain', value=currentDoamin)
        extractedData= extractedData.append(currentExtractedData, ignore_index=True)
    print(len(extractedData))
    # Extract only required columns
    nanIndex = []
    str = 'domain'
    colListWithDomain = colList + [str]

    # Remove samples which include any NaN sensor value
    truncatedData = extractedData.dropna(subset=colListWithDomain)
    truncatedData = truncatedData.reset_index(drop=True)

    # Find out label
    labelDict = {}
    labelList = []
    isLabeled = []
    for key, value in truncatedData.iteritems():
        if key in colList:
            continue
        if key is 'domain':
            continue
        for i in range(value.values.size):
            if value.values[i] == 1:
                labelDict[i] = key
                isLabeled.append(i)
    labeledData = truncatedData.loc[isLabeled, colListWithDomain]
    labeledData = labeledData.sort_index()
    labeledData = labeledData.reset_index(drop=True)
    #isLabeled = isLabeled.sort()
    sortedLabelDict = sorted(labelDict.items())
    for tupleKey, tupleValue in sortedLabelDict:
        labelList.append(tupleValue)

    labeledData.insert(loc=len(colListWithDomain), column='label', value=labelList)


    # print(labeledData)

    minmaxScaledData = minmaxScaling(labeledData)
    sampledData = sampling(minmaxScaledData)
    #print(sampledData)
    sampledData = sampledData.sort_values(['domain','label'], ascending=[True, True])

    fourUserSampledData = sampledData.loc[df['domain'].isin(['136562B6-95B2-483D-88DC-065F28409FD2', '1538C99F-BA1E-4EFB-A949-6C7C47701B20', '5152A2DF-FAF3-4BA8-9CA9-E66B32671A53', '7CE37510-56D0-4120-A1CF-0E23351428D2', '83CF687B-7CEC-434B-9FE8-00C3D5799BE6', '9DC38D04-E82E-4F29-AB52-B476535226F2', 'A76A5AF5-5A93-4CF2-A16E-62353BB70E8A'])]

    #sampledData.to_csv('../dataset/ExtraSensory/extrasensory_minmax_scaling_8win_all3.csv', index=False)
    fourUserSampledData.to_csv('../dataset/extrasensory/extrasensory_minmax_scaling_8win_all3.csv', index=False)
    '''


def numShot():
    df = pd.read_csv(filePath + 'both_all.csv')
    df = df.drop(['Unnamed: 0'], axis=1)

    domain = opt['domains']
    activities = opt['classes']

    # Initialize
    num_shot_actset = {key: 0 for key in activities}
    num_shot = {key: num_shot_actset.copy() for key in domain}
    i = 0
    while 1:
        j = 0
        while 1:
            if i + j >= len(df):
                break
            if j != 0:
                if df['domain'][i + j] != df['domain'][i + j - 1]:  # Different Domain
                    i = i + j
                    j = 0
                    break
                if df['label'][i + j] != df['label'][i + j - 1]:  # Different Activity
                    i = i + j
                    j = 0
                    break
            if j == WIN_LEN - 1:  # consistent domain and activity with 8 rows
                num_shot[df['domain'][i + j]][df['label'][i + j]] += 1
                i = i + j
                j = 0
                break
            j += 1

        if i >= len(df):
            break
        i += 1

    print("num_shot is =========")
    print(num_shot)
    df_num_shot = pd.DataFrame.from_dict(num_shot, orient="index", columns=activities)
    df_num_shot.to_csv(filePath + 'wesad_num_shot.csv', index=True)


def pkl_to_csv():
    colDomainList = opt['domains']
    # filePath = '../dataset/wesad/'
    filePath = '../dataset/wesad/'

    def readPKL():
        for domain in colDomainList:
            df = pd.read_pickle('../dataset/wesad/' + domain + '/' + domain + '.pkl')
            # print(df)
            signalChest = df['signal']['chest'].copy()
            signalWrist = df['signal']['wrist'].copy()
            label = df['label']  # array
            # (1) ACC processing
            signalChest['chestAcc'] = []
            signalWrist['wristAcc'] = []
            for i in range(len(signalChest['ACC'])):
                signalChest['chestAcc'].append(np.sqrt(
                    np.power(signalChest['ACC'][i][0], 2) + np.power(signalChest['ACC'][i][1], 2) + np.power(
                        signalChest['ACC'][i][2], 2)))
            for i in range(len(signalWrist['ACC'])):
                signalWrist['wristAcc'].append(np.sqrt(
                    np.power(signalWrist['ACC'][i][0], 2) + np.power(signalWrist['ACC'][i][1], 2) + np.power(
                        signalChest['ACC'][i][2], 2)))
            # Remove [ACC] from signalChest and signalWrist
            signalChest.pop('ACC')
            signalWrist.pop('ACC')

            print("--------Acc processing completed--------")
            # (2) Flat columns
            signalChest['chestECG'] = np.array(signalChest['ECG'][:, 0])
            signalChest['chestEMG'] = np.array(signalChest['EMG'][:, 0])
            signalChest['chestEDA'] = np.array(signalChest['EDA'][:, 0])
            signalChest['chestTemp'] = np.array(signalChest['Temp'][:, 0])
            signalChest['chestResp'] = np.array(signalChest['Resp'][:, 0])
            signalChest.pop('ECG')
            signalChest.pop('EMG')
            signalChest.pop('EDA')
            signalChest.pop('Temp')
            signalChest.pop('Resp')
            signalWrist['wristBVP'] = np.array(signalWrist['BVP'])
            signalWrist['wristEDA'] = np.array(signalWrist['EDA'])
            signalWrist['wristTemp'] = np.array(signalWrist['TEMP'])
            signalWrist.pop('BVP')
            signalWrist.pop('EDA')
            signalWrist.pop('TEMP')

            print("--------First flatting completed--------")

            # (3) Downsampling
            print("-------Before downsampling-----------")
            '''
            print(len(signalWrist['wristAcc'])) # 32Hz : 8row
            print(len(signalWrist['wristTemp']))   # 4Hz : 1row
            print(len(signalWrist['wristBVP']))  # 64Hz : 16row
            print(len(signalWrist['wristEDA'])) # 4Hz : 1row
            '''

            def downsample_to_proportion(rows, proportion=1):
                return list(islice(rows, 0, len(rows), int(1 / proportion)))

            signalWrist['wristAcc'] = downsample_to_proportion(rows=signalWrist['wristAcc'], proportion=0.125)
            signalWrist['wristBVP'] = downsample_to_proportion(rows=signalWrist['wristBVP'], proportion=0.0625)
            signalChest['chestAcc'] = downsample_to_proportion(rows=signalChest['chestAcc'], proportion=0.005714285714)
            signalChest['chestECG'] = downsample_to_proportion(rows=signalChest['chestECG'], proportion=0.005714285714)
            signalChest['chestEMG'] = downsample_to_proportion(rows=signalChest['chestEMG'], proportion=0.005714285714)
            signalChest['chestEDA'] = downsample_to_proportion(rows=signalChest['chestEDA'], proportion=0.005714285714)
            signalChest['chestTemp'] = downsample_to_proportion(rows=signalChest['chestTemp'],
                                                                proportion=0.005714285714)
            signalChest['chestResp'] = downsample_to_proportion(rows=signalChest['chestResp'],
                                                                proportion=0.005714285714)

            print("-------After downsampling-----------")
            '''
            print(len(signalWrist['wristAcc'])) # 32Hz : 8row
            print(len(signalWrist['wristTemp']))   # 4Hz : 1row
            print(len(signalWrist['wristBVP']))  # 64Hz : 16row
            print(len(signalWrist['wristEDA'])) # 4Hz : 1row
            print(len(signalWrist['label']))
            print(len(signalWrist['domain']))
            '''
            # Flat dimension
            l = []
            m = []
            n = []
            for sublist in signalWrist['wristTemp']:
                for item in sublist:
                    l.append(item)
            for sublist in signalWrist['wristBVP']:
                for item in sublist:
                    m.append(item)
            for sublist in signalWrist['wristEDA']:
                for item in sublist:
                    n.append(item)
            signalWrist['wristTemp'] = l
            signalWrist['wristBVP'] = m
            signalWrist['wristEDA'] = n

            # (4) Dict to dataframe
            dfSignalChest = pd.DataFrame.from_dict(signalChest)
            dfSignalWrist = pd.DataFrame.from_dict(signalWrist)

            # minmax Scaling
            minmaxScaledSignalChest = minmaxScaling(dfSignalChest)
            minmaxScaledSignalWrist = minmaxScaling(dfSignalWrist)

            # label and domain fields
            minmaxScaledSignalChest['label'] = downsample_to_proportion(rows=label, proportion=0.005714285714)
            minmaxScaledSignalWrist['label'] = downsample_to_proportion(rows=label, proportion=0.005714285714)
            l = []
            m = []
            for _ in range(len(signalWrist['wristAcc'])):
                l.append(domain)
            for _ in range(len(signalChest['chestAcc'])):
                m.append(domain)
            minmaxScaledSignalChest['domain'] = m
            minmaxScaledSignalWrist['domain'] = l
            minmaxScaledSignalChest = minmaxScaledSignalChest.loc[dfSignalChest['label'].isin([1, 2, 3, 4])]
            minmaxScaledSignalWrist = minmaxScaledSignalWrist.loc[dfSignalWrist['label'].isin([1, 2, 3, 4])]

            # sampling
            minmaxScaledSignalChest = minmaxScaledSignalChest.reset_index(drop=True)
            minmaxScaledSignalWrist = minmaxScaledSignalWrist.reset_index(drop=True)
            minmaxScaledSignalChest = sampling(minmaxScaledSignalChest)
            minmaxScaledSignalWrist = sampling(minmaxScaledSignalWrist)

            minmaxScaledSignalBoth = pd.concat([minmaxScaledSignalChest, minmaxScaledSignalWrist], axis=1)
            minmaxScaledSignalBoth = minmaxScaledSignalBoth.loc[:, ~minmaxScaledSignalBoth.columns.duplicated()]
            minmaxScaledSignalBoth = minmaxScaledSignalBoth[
                ['chestAcc', 'chestECG', 'chestEMG', 'chestEDA', 'chestTemp', 'chestResp', 'wristAcc', 'wristBVP',
                 'wristEDA', 'wristTemp', 'label', 'domain']]

            minmaxScaledSignalChest = minmaxScaledSignalChest.reset_index(drop=True)
            minmaxScaledSignalWrist = minmaxScaledSignalWrist.reset_index(drop=True)
            minmaxScaledSignalBoth = minmaxScaledSignalBoth.reset_index(drop=True)

            minmaxScaledSignalChest['label'] = minmaxScaledSignalChest['label'].replace(
                {1: "baseline", 2: "stress", 3: "amusement", 4: "mediation"})
            minmaxScaledSignalWrist['label'] = minmaxScaledSignalWrist['label'].replace(
                {1: "baseline", 2: "stress", 3: "amusement", 4: "mediation"})
            minmaxScaledSignalBoth['label'] = minmaxScaledSignalBoth['label'].replace(
                {1: "baseline", 2: "stress", 3: "amusement", 4: "mediation"})

            minmaxScaledSignalChest.to_csv('../dataset/wesad/' + domain + '/' + domain + '_chest.csv')
            minmaxScaledSignalWrist.to_csv('../dataset/wesad/' + domain + '/' + domain + '_wrist.csv')
            minmaxScaledSignalBoth.to_csv('../dataset/wesad/' + domain + '/' + domain + '_both.csv')

            # (5) 200samples per domain/label
            # chestLabel1 = minmaxScaledSignalChest_wClass.loc[minmaxScaledSignalChest_wClass['label'].isin([1])][:280000]
            # chestLabel2 = minmaxScaledSignalChest_wClass.loc[minmaxScaledSignalChest_wClass['label'].isin([2])][:280000]
            # chestLabel3 = minmaxScaledSignalChest_wClass.loc[minmaxScaledSignalChest_wClass['label'].isin([3])][:280000]
            # chestLabel4 = minmaxScaledSignalChest_wClass.loc[minmaxScaledSignalChest_wClass['label'].isin([4])][:280000]
            # wristLabel1 = minmaxScaledSignalWrist_wClass.loc[minmaxScaledSignalWrist_wClass['label'].isin([1])]

            print("---------" + domain + " saved------------")

    # readPKL()

    def combineCSV():
        # colChestLabel = ['chestAcc','chestECG','chestEMG','chestEDA','chestTemp','chestResp', 'label','domain']
        # colWristLabel = ['wristAcc','wristBVP','wristEDA','wristTemp','label','domain']
        colBothLabel = ['chestAcc', 'chestECG', 'chestEMG', 'chestEDA', 'chestTemp', 'chestResp', 'wristAcc',
                        'wristBVP', 'wristEDA', 'wristTemp', 'label', 'domain']

        # chestAll = pd.DataFrame(columns=colChestLabel)
        # wristAll = pd.DataFrame(columns=colWristLabel)
        bothAll = pd.DataFrame(columns=colBothLabel)

        # extractedData_chest = pd.DataFrame(columns=colChestLabel)
        # extractedData_wrist = pd.DataFrame(columns = colWristLabel)
        extractedData_both = extractedData_both.sort_values(['domain', 'label'], ascending=[True, True])
        extractedData_both = pd.DataFrame(columns=colBothLabel)

        for domain in domains:
            # currentExtractedData_chest = pd.read_csv(filePath + domain + '/'+domain+'_chest.csv', usecols=colChestLabel)
            # currentExtractedData_wrist = pd.read_csv(filePath + domain + '/'+domain+'_wrist.csv', usecols=colWristLabel)
            currentExtractedData_both = pd.read_csv(filePath + domain + '/' + domain + '_both.csv',
                                                    usecols=colBothLabel)

            # extractedData_chest = extractedData_chest.append(currentExtractedData_chest, ignore_index=True)
            # extractedData_wrist = extractedData_wrist.append(currentExtractedData_wrist, ignore_index=True)
            extractedData_both = extractedData_both.append(currentExtractedData_both, ignore_index=True)

            print("----------" + domain + " completed----------")

        # extractedData_chest.to_csv('../dataset/wesad/chest_all.csv', index=False)
        # extractedData_wrist.to_csv('../dataset/wesad/wrist_all.csv', index=False)
        extractedData_both.to_csv('../dataset/wesad/both_all.csv', index=False)

    combineCSV()


if __name__ == '__main__':
    is_minmax_scaling = True
    domains = opt['domains']
    activities = opt['classes']
    pkl_to_csv()
    # numShot()
    # initial_preprocessing_extract_required_data('../dataset/wesad/')

# initial_preprocessing_extract_required_data('../dataset/wesad/')
# numShot()
