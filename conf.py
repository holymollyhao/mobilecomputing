import numpy as np
import itertools

args = None
DogwalkAll_WIN5_Opt = {
    'domain_loc': 8,
    'class_loc': 7,
    'name': 'dogwalk',
    'batch_size': 16,
    'seq_len': 5,
    'input_dim': 6,

    'learning_rate': 0.001,
    'weight_decay': 0.0005,

    'momentum': 0.9,
    'file_path': './dataset/dog_walk_winsize_5_train_valid_split_std_all.csv',

    'classes': ['ambiguous', 'not walking', 'stationary', 'walking', 'running'],
    'num_class': 5,

    # 22 users
    'users': [
        'user1_train',
        'user1_test',
        'user2_train',
        'user2_test',
        'user3_train',
        'user3_test',
        'user4_train',
        'user4_test',
        'user5_train',
        'user5_test'
    ],

    'src_domains': [
        'user1_train',
        'user1_test',
        'user2_train',
        'user2_test',
        'user3_train',
        'user3_test',
        'user4_train',
        'user4_test',
        'user5_train',
        'user5_test'
    ],
    'tgt_domains': [
        'user1_train',
        'user1_test',
        'user2_train',
        'user2_test',
        'usesr3_train',
        'user3_test'
        'user4_train',
        'user4_test',
        'user5_train',
        'user5_test'
    ],
}
DogwalkAll_WIN100_Opt = {
    'domain_loc': 8,
    'class_loc': 7,
    'name': 'dogwalk',
    'batch_size': 16,
    'seq_len': 100,
    'input_dim': 6,

    'learning_rate': 0.001,
    'weight_decay': 0.0005,

    'momentum': 0.9,
    'file_path': './dataset/dog_walk_winsize_100_train_valid_split_std_all.csv',

    'classes': ['ambiguous', 'not walking', 'stationary', 'walking', 'running'],
    'num_class': 5,

    # 22 users
    'users': [
        'user1_train',
        'user1_test',
        'user2_train',
        'user2_test',
        'user3_train',
        'user3_test',
        'user4_train',
        'user4_test',
        'user5_train',
        'user5_test'
    ],
    'src_domains': [
        'user1_train',
        'user1_test',
        'user2_train',
        'user2_test',
        'user3_train',
        'user3_test',
        'user4_train',
        'user4_test',
        'user5_train',
        'user5_test'
    ],
    'tgt_domains': [
        'user1_train',
        'user1_test',
        'user2_train',
        'user2_test',
        'usesr3_train',
        'user3_test'
        'user4_train',
        'user4_test',
        'user5_train',
        'user5_test'
    ],
}
DogwalkAllOpt = {
    'domain_loc': 8,
    'class_loc': 7,
    'name': 'dogwalk',
    'batch_size': 16,
    'seq_len': 50,
    'input_dim': 6,

    'learning_rate': 0.001,
    'weight_decay': 0.0005,

    'momentum': 0.9,
    'file_path': './dataset/dog_walk_winsize_50_train_valid_split_std_all.csv',

    'classes': ['ambiguous', 'not walking', 'stationary', 'walking', 'running'],
    'num_class': 5,

    # 22 users
    'users': [
        'user1_train',
        'user1_test',
        'user2_train',
        'user2_test',
        'user3_train',
        'user3_test',
        'user4_train',
        'user4_test',
        'user5_train',
        'user5_test'
    ],
    'src_domains': [
        'user1_train',
        'user1_test',
        'user2_train',
        'user2_test',
        'user3_train',
        'user3_test',
        'user4_train',
        'user4_test',
        'user5_train',
        'user5_test'
    ],
    'tgt_domains': [
        'user1_train',
        'user1_test',
        'user2_train',
        'user2_test',
        'usesr3_train',
        'user3_test'
        'user4_train',
        'user4_test',
        'user5_train',
        'user5_test'
    ],
}
Dogwalk_WIN100_Opt = {
    'domain_loc': 5,
    'class_loc': 4,
    'name': 'dogwalk',
    'batch_size': 16,
    'seq_len': 100,
    'input_dim': 3,

    'learning_rate': 0.001,
    'weight_decay': 0.0005,

    'momentum': 0.9,
    'file_path': './dataset/dog_walk_winsize_100_train_valid_split_std_acc.csv',
    # 'file_path': './dataset/dog_walk_winsize_50_train_valid_split_minmax_acc.csv',

    'classes': ['ambiguous', 'not walking', 'stationary', 'walking', 'running'],
    'num_class': 5,

    # 22 users
    'users': [
        'user1_train',
        'user1_test',
        'user2_train',
        'user2_test',
        'user3_train',
        'user3_test',
        'user4_train',
        'user4_test',
        'user5_train',
        'user5_test'
    ],
    'src_domains': [
        'user1_train',
        'user1_test',
        'user2_train',
        'user2_test',
        'user3_train',
        'user3_test',
        'user4_train',
        'user4_test',
        'user5_train',
        'user5_test'
    ],
    'tgt_domains': [
        'user1_train',
        'user1_test',
        'user2_train',
        'user2_test',
        'usesr3_train',
        'user3_test'
        'user4_train',
        'user4_test',
        'user5_train',
        'user5_test'
    ],
}

DogwalkOpt = {
    'domain_loc': 5,
    'class_loc': 4,
    'name': 'dogwalk',
    'batch_size': 16,
    'seq_len': 50,
    'input_dim': 3,

    'learning_rate': 0.001,
    'weight_decay': 0.0005,

    'momentum': 0.9,
    'file_path': './dataset/dog_walk_winsize_50_train_valid_split_std_acc.csv',

    'classes': ['ambiguous', 'not walking', 'stationary', 'walking', 'running'],
    'num_class': 5,

    # 22 users
    'users': [
        'user1_train',
        'user1_test',
        'user2_train',
        'user2_test',
        'user3_train',
        'user3_test',
        'user4_train',
        'user4_test',
        'user5_train',
        'user5_test'
    ],
    'src_domains': [
        'user1_train',
        'user1_test',
        'user2_train',
        'user2_test',
        'user3_train',
        'user3_test',
        'user4_train',
        'user4_test',
        'user5_train',
        'user5_test'
    ],
    'tgt_domains': [
        'user1_train',
        'user1_test',
        'user2_train',
        'user2_test',
        'usesr3_train',
        'user3_test'
        'user4_train',
        'user4_test',
        'user5_train',
        'user5_test'
    ],
}

def init_domains():
    seed = 0
    import random
    import numpy as np
    np.random.seed(seed)
    random.seed(seed)
    import math



if __name__ == "__main__":
    init_domains()
