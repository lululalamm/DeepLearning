import os, torch
from utils import protocol_decoder
import math
import pandas as pd


class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]

def make_weights_h5(df):
    
    img_paths = df[0]
    id_list = df[1]
    label_list = df[2]

    n = len(label_list)
    keys = list(range(n))
    weights = [0 for i in range(n)]

    for key in keys:
        target = label_list[key]
        if target=='real':
            weights[int(key)]=0.2
        else:
            weights[int(key)]=0.1

    assert len(weights) == n
    return n, weights


def get_single_dataset(data_dir, FaceDataset, data_name="", train=True, label=None, img_size=256, map_size=32, transform=None, debug_subset_size=None, UUID=-1,ca='crop',use_sampler=False):
    
    if train:
        if data_name=='wild_train':
            csv_names = ["wild_{}_list_train.csv".format(ca)]
        elif data_name=='celebA_train':
            csv_names = ["celebA_{}_list_train.csv".format(ca)]
        elif data_name=='casia_train':
            csv_names = ["casia_{}_list_train.csv".format(ca)]  
        elif data_name=='casia_all':
            csv_names = ['casia_{}_list_train.csv'.format(ca),'casia_{}_list_test.csv'.format(ca)]
        elif data_name=='celebA_all':
            csv_names = ['celebA_{}_list_train.csv'.format(ca),'celebA_{}_list_test.csv'.format(ca)]
        elif data_name=='wild_all':
            csv_names = ['wild_{}_list_train.csv'.format(ca),'wild_{}_list_test.csv'.format(ca)]

        csv_df = pd.DataFrame()
        for name in csv_names:
            df = pd.read_csv(os.path.join(data_dir,name),header=None)
            csv_df = pd.concat((csv_df,df))
        csv_df = csv_df.sample(frac=1).reset_index(drop=True)

        if use_sampler:
            num_instances, weights = make_weights_h5(csv_df)

        data_set = FaceDataset(data_name, csv_df, split='train', label=label,
                                      transform=transform, UUID=UUID)


        if debug_subset_size is not None:
            data_set = torch.utils.data.Subset(data_set, range(0, debug_subset_size))
    else:
        if data_name=='wild_val':
            csv_names = ['wild_{}_list_test.csv'.format(ca)]
        elif data_name=='wild_celebA_test':
            csv_names = ['wild_{}_list_test.csv'.format(ca),'celebA_{}_list_test.csv'.format(ca)]
        elif data_name=='all_test':
            csv_names = ['wild_{}_list_test.csv'.format(ca),'celebA_{}_list_test.csv'.format(ca),'casia_{}_list_test.csv'.format(ca)]
        elif data_name=='casia_all':
            csv_names = ['casia_{}_list_train.csv'.format(ca),'casia_{}_list_test.csv'.format(ca)]
        elif data_name=='celebA_all':
            csv_names = ['celebA_{}_list_train.csv'.format(ca),'celebA_{}_list_test.csv'.format(ca)]
        elif data_name=='wild_all':
            csv_names = ['wild_{}_list_train.csv'.format(ca),'wild_{}_list_test.csv'.format(ca)]

        csv_df = pd.DataFrame()
        for name in csv_names:
            df = pd.read_csv(os.path.join(data_dir,name),header=None)
            csv_df = pd.concat((csv_df,df))
        csv_df = csv_df.sample(frac=1).reset_index(drop=True)

        data_set = FaceDataset(data_name, csv_df, split='test', label=label,
                                      transform=transform, map_size=map_size, UUID=UUID)

        if debug_subset_size is not None:
            data_set = torch.utils.data.Subset(data_set, range(0, debug_subset_size))
    # print("Loading {}, number: {}".format(data_name, len(data_set)))
    if use_sampler:
        return data_set, num_instances, weights
    else:
        return data_set

def get_datasets(data_dir, FaceDataset, train=True, protocol="1", img_size=256, map_size=32, transform=None, debug_subset_size=None,ca='crop',use_sampler=False):

    data_name_list_train, data_name_list_test = protocol_decoder(protocol)

    if train:
        print("train transform")
        print(transform)
    else:
        print("test transform")
        print(transform)

    sum_n = 0
    if train:
        if use_sampler:
            data_set_sum, num_instances, weights = get_single_dataset(data_dir, FaceDataset, data_name=data_name_list_train[0], train=True, img_size=img_size, map_size=map_size, transform=transform, debug_subset_size=debug_subset_size, UUID=0,ca=ca,use_sampler=use_sampler)
        else:
            data_set_sum = get_single_dataset(data_dir, FaceDataset, data_name=data_name_list_train[0], train=True, img_size=img_size, map_size=map_size, transform=transform, debug_subset_size=debug_subset_size, UUID=0,ca=ca,use_sampler=use_sampler)
        sum_n = len(data_set_sum)
        for i in range(1, len(data_name_list_train)):
            data_tmp = get_single_dataset(data_dir, FaceDataset, data_name=data_name_list_train[i], train=True, img_size=img_size, map_size=map_size, transform=transform, debug_subset_size=debug_subset_size, UUID=i,ca=ca,use_sampler=use_sampler)
            data_set_sum += data_tmp
            sum_n += len(data_tmp)
    else:
        data_set_sum = {}
        for i in range(len(data_name_list_test)):
            data_tmp = get_single_dataset(data_dir, FaceDataset, data_name=data_name_list_test[i], train=False, img_size=img_size, map_size=map_size, transform=transform, debug_subset_size=debug_subset_size, UUID=i,ca=ca,use_sampler=use_sampler)
            data_set_sum[data_name_list_test[i]] = data_tmp
            sum_n += len(data_tmp)
    print("Total number: {}".format(sum_n))

    if use_sampler:
        return data_set_sum,num_instances, weights
    else:
        return data_set_sum
