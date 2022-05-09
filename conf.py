import numpy as np
import itertools

args = None

DogwalkOpt = {
    'name': 'dogwalk',
    'batch_size': 64,
    'seq_len': 50,
    'input_dim': 3,

    'learning_rate': 0.001,
    'weight_decay': 0.0005,

    'momentum': 0.9,
    'file_path': './dataset/dog_walk_winsize_50_stdscale_labelasstring.csv',

    'classes': ['not walking', 'stationary', 'walking', 'running'],
    'num_class': 4,

    # 22 users
    'users': ['user0'],

    'src_domains': ['user0'],
    'tgt_domains': [
        'user0',
    ],
}

CIFAR10Opt = {
    'name': 'cifar10',
    'batch_size': 128,

    'learning_rate': 0.1,  # initial learning rate
    'weight_decay': 0.0005,
    'momentum': 0.9,
    'img_size': 3072,

    'file_path': './dataset/CIFAR-10-C',
    'classes': ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
    'num_class': 10,
    'severity': 5,
    # 'corruptions': ["shot_noise", "motion_blur", "snow", "pixelate", "gaussian_noise", "defocus_blur",
    #                 "brightness", "fog", "zoom_blur", "frost", "glass_blur", "impulse_noise", "contrast",
    #                 "jpeg_compression", "elastic_transform"],
    'domains': ["original",

                "test",

                "gaussian_noise-1", "gaussian_noise-2", "gaussian_noise-3", "gaussian_noise-4", "gaussian_noise-5",
                "gaussian_noise-all",

                "shot_noise-1", "shot_noise-2", "shot_noise-3", "shot_noise-4", "shot_noise-5", "shot_noise-all",

                "impulse_noise-1", "impulse_noise-2", "impulse_noise-3", "impulse_noise-4", "impulse_noise-5",
                "impulse_noise-all",

                "defocus_blur-1", "defocus_blur-2", "defocus_blur-3", "defocus_blur-4", "defocus_blur-5",
                "defocus_blur-all",

                "glass_blur-1", "glass_blur-2", "glass_blur-3", "glass_blur-4", "glass_blur-5", "glass_blur-all",

                "motion_blur-1", "motion_blur-2", "motion_blur-3", "motion_blur-4", "motion_blur-5", "motion_blur-all",

                "zoom_blur-1", "zoom_blur-2", "zoom_blur-3", "zoom_blur-4", "zoom_blur-5", "zoom_blur-all",

                "snow-1", "snow-2", "snow-3", "snow-4", "snow-5", "snow-all",

                "frost-1", "frost-2", "frost-3", "frost-4", "frost-5", "frost-all",

                "fog-1", "fog-2", "fog-3", "fog-4", "fog-5", "fog-all",

                "brightness-1", "brightness-2", "brightness-3", "brightness-4", "brightness-5", "brightness-all",

                "contrast-1", "contrast-2", "contrast-3", "contrast-4", "contrast-5", "contrast-all",

                "elastic_transform-1", "elastic_transform-2", "elastic_transform-3", "elastic_transform-4",
                "elastic_transform-5", "elastic_transform-all",

                "pixelate-1", "pixelate-2", "pixelate-3", "pixelate-4", "pixelate-5", "pixelate-all",

                "jpeg_compression-1", "jpeg_compression-2", "jpeg_compression-3", "jpeg_compression-4",
                "jpeg_compression-5", "jpeg_compression-all",
                ],
    'src_domains': ["original"],
    'tgt_domains': [
        "gaussian_noise-5",
        "shot_noise-5",
        "impulse_noise-5",
        "defocus_blur-5",
        "glass_blur-5",
        "motion_blur-5",
        "zoom_blur-5",
        "snow-5",
        "frost-5",
        "fog-5",
        "brightness-5",
        "brightness-5",
        "elastic_transform-5",
        "pixelate-5",
        "jpeg_compression-5",

    ],
}

CIFAR100Opt = {
    'name': 'cifar100',
    'batch_size': 128,

    'learning_rate': 0.1, #initial learning rate
    'weight_decay': 0.0005,
    'momentum': 0.9,
    'img_size': 3072,

    'file_path': './dataset/CIFAR-100-C',
    'classes': ['beaver', 'dolphin', 'otter', 'seal', 'whale',
                'aquarium fish', 'flatfish', 'ray', 'shark', 'trout',
                'orchids', 'poppies', 'roses', 'sunflowers', 'tulips',
                'bottles', 'bowls', 'cans', 'cups', 'plates',
                'apples', 'mushrooms', 'oranges', 'pears', 'sweet peppers',
                'clock', 'computer keyboard', 'lamp', 'telephone', 'television',
                'bed', 'chair', 'couch', 'table', 'wardrobe',
                'bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach',
                'bear', 'leopard', 'lion', 'tiger', 'wolf',
                'bridge', 'castle', 'house', 'road', 'skyscraper',
                'cloud', 'forest', 'mountain', 'plain', 'sea',
                'camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo',
                'fox', 'porcupine', 'possum', 'raccoon', 'skunk',
                'crab', 'lobster', 'snail', 'spider', 'worm',
                'baby', 'boy', 'girl', 'man', 'woman',
                'crocodile', 'dinosaur', 'lizard', 'snake', 'turtle',
                'hamster', 'mouse', 'rabbit', 'shrew', 'squirrel',
                'maple', 'oak', 'palm', 'pine', 'willow',
                'bicycle', 'bus', 'motorcycle', 'pickup truck', 'train',
                'lawn-mower', 'rocket', 'streetcar', 'tank', 'tractor'],
    'num_class': 100,
    'severity': 5,
    # 'corruptions': ["shot_noise", "motion_blur", "snow", "pixelate", "gaussian_noise", "defocus_blur",
    #                 "brightness", "fog", "zoom_blur", "frost", "glass_blur", "impulse_noise", "contrast",
    #                 "jpeg_compression", "elastic_transform"],
    'domains': ["original",

                "test",

                "gaussian_noise-1", "gaussian_noise-2", "gaussian_noise-3", "gaussian_noise-4", "gaussian_noise-5",
                "gaussian_noise-all",

                "shot_noise-1", "shot_noise-2", "shot_noise-3", "shot_noise-4", "shot_noise-5", "shot_noise-all",

                "impulse_noise-1", "impulse_noise-2", "impulse_noise-3", "impulse_noise-4", "impulse_noise-5",
                "impulse_noise-all",

                "defocus_blur-1", "defocus_blur-2", "defocus_blur-3", "defocus_blur-4", "defocus_blur-5",
                "defocus_blur-all",

                "glass_blur-1", "glass_blur-2", "glass_blur-3", "glass_blur-4", "glass_blur-5", "glass_blur-all",

                "motion_blur-1", "motion_blur-2", "motion_blur-3", "motion_blur-4", "motion_blur-5", "motion_blur-all",

                "zoom_blur-1", "zoom_blur-2", "zoom_blur-3", "zoom_blur-4", "zoom_blur-5", "zoom_blur-all",

                "snow-1", "snow-2", "snow-3", "snow-4", "snow-5", "snow-all",

                "frost-1", "frost-2", "frost-3", "frost-4", "frost-5", "frost-all",

                "fog-1", "fog-2", "fog-3", "fog-4", "fog-5", "fog-all",

                "brightness-1", "brightness-2", "brightness-3", "brightness-4", "brightness-5", "brightness-all",

                "contrast-1", "contrast-2", "contrast-3", "contrast-4", "contrast-5", "contrast-all",

                "elastic_transform-1", "elastic_transform-2", "elastic_transform-3", "elastic_transform-4",
                "elastic_transform-5", "elastic_transform-all",

                "pixelate-1", "pixelate-2", "pixelate-3", "pixelate-4", "pixelate-5", "pixelate-all",

                "jpeg_compression-1", "jpeg_compression-2", "jpeg_compression-3", "jpeg_compression-4",
                "jpeg_compression-5", "jpeg_compression-all",
                ],
    'src_domains': ["original"],
    'tgt_domains': [
        "gaussian_noise-5",
                    "shot_noise-5",
                    "impulse_noise-5",
                    "defocus_blur-5",
                    "glass_blur-5",
                    "motion_blur-5",
                    "zoom_blur-5",
                    "snow-5",
                    "frost-5",
                    "fog-5",
                    "brightness-5",
                    "contrast-5",
                    "elastic_transform-5",
                    "pixelate-5",
                    "jpeg_compression-5",

    ],
}

VLCSOpt = {
    'name': 'vlcs',
    'batch_size': 32,

    'learning_rate': 0.00005,
    'weight_decay': 0,
    'momentum': 0.9,
    'img_size': 224,

    'file_path': './dataset/VLCS',

    'classes': ['bird', 'car', 'chair', 'dog', 'person'],
    # 'sub_classes': ['Car', 'Pedestrian', 'Cyclist'],
    'num_class': 5,  # 8 #TODO: need to change config path as well
    # 'config_path': 'config/yolov3-kitti.cfg',
    'domains': ['Caltech101', 'LabelMe', 'SUN09', 'VOC2007'],
    'src_domains': ['SUN09', 'LabelMe', 'VOC2007', 'Caltech101'],
    'tgt_domains': ['SUN09'],
    # 'val_domains': ['rain-200-val'],
}

PACSOpt = {
    'name': 'pacs',
    'batch_size': 32,

    'learning_rate': 0.00005,
    'weight_decay': 0,
    'momentum': 0.9,
    'img_size': 224,

    'file_path': './dataset/PACS',

    'classes': ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person'],
    # 'sub_classes': ['Car', ''Pedestrian'', 'Cyclist'],
    'num_class': 7,  # 8 #TODO: need to change config path as well
    # 'config_path': 'config/yolov3-kitti.cfg',
    'domains': ['art_painting', 'cartoon', 'photo', 'sketch'],
    'src_domains': ['art_painting', 'cartoon', 'photo', 'sketch'],
    'tgt_domains': ['art_painting', 'cartoon', 'photo', 'sketch'],
    # 'val_domains': ['rain-200-val'],
}

OfficeHomeOpt = {
    'name': 'officehome',
    'batch_size': 32,

    'learning_rate': 0.00005,
    'weight_decay': 0,
    'momentum': 0.9,
    'img_size': 224,

    'file_path': './dataset/OfficeHomeDataset',

    'classes': ['Alarm_Clock', 'Bottle', 'Chair', 'Desk_Lamp', 'File_Cabinet', 'Glasses', 'Knives', 'Mop', 'Pan',
                'Printer', 'Scissors', 'Soda', 'Telephone',
                'Backpack', 'Bucket', 'Clipboards', 'Drill', 'Flipflops', 'Hammer', 'Lamp_Shade', 'Mouse', 'Paper_Clip',
                'Push_Pin', 'Screwdriver', 'Speaker', 'ToothBrush',
                'Batteries', 'Calculator', 'Computer', 'Eraser', 'Flowers', 'Helmet', 'Laptop', 'Mug', 'Pen', 'Radio',
                'Shelf', 'Spoon', 'Toys',
                'Bed', 'Calendar', 'Couch', 'Exit_Sign', 'Folder', 'Kettle', 'Marker', 'Notebook', 'Pencil',
                'Refrigerator', 'Sink', 'TV', 'Trash_Can',
                'Bike', 'Candles', 'Curtains', 'Fan', 'Fork', 'Keyboard', 'Monitor', 'Oven', 'Postit_Notes', 'Ruler',
                'Sneakers', 'Table', 'Webcam'],
    # 'sub_classes': ['Car', ''Pedestrian'', 'Cyclist'],
    'num_class': 65,  # 8 #TODO: need to change config path as well
    # 'config_path': 'config/yolov3-kitti.cfg',
    'domains': ['Art', 'Clipart', 'RealWorld'],
    'src_domains': ['Art', 'Clipart', 'RealWorld'],
    'tgt_domains': ['Art', 'Clipart', 'RealWorld'],
    # 'val_domains': ['rain-200-val'],
}

KITTI_SOT_Opt = {
    'name': 'kitti_sot',
    'batch_size': 64,

    'learning_rate': 0.001,
    'weight_decay': 0,
    'momentum': 0.9,

    'file_path': './dataset/kitti_sot',

    'classes': ['Car', 'Van', 'Truck', 'Pedestrian', 'Person', 'Cyclist', 'Tram', 'Misc'],
    'sub_classes': ['Car', 'Van', 'Truck', 'Pedestrian', 'Person', 'Cyclist', 'Tram', 'Misc'],
    # 'sub_classes': ['Car', 'Pedestrian', 'Cyclist'],
    'num_class': 8,
    'domains': ['2d_detection', 'original', 'rain-100mm', 'rain-100mm'],
    # 'domains': ['half1', 'half2'],
    # 'src_domains': ['half1'],
    # 'tgt_domains': ['half2'],

    'src_domains': ['2d_detection'],
    # 'src_domains': ['original'],
    # 'src_domains': ['original-val'],

    # 'tgt_domains': ['rain-200-tgt'],
    'tgt_domains': ['rain-200'],
    'val_domains': ['rain-200-val'],
    # 'src_domains': ['rain'],
    # 'tgt_domains': ['original'],
}

HARTHOpt = {
    'name': 'hhar',
    'batch_size': 64,
    'seq_len': 50,  # 128, 32, 5
    'input_dim': 3,  # 161, #6
    # 'learning_rate': 0.0001,
    # 'weight_decay': 0,

    'learning_rate': 0.1,  # initial learning rate
    'weight_decay': 0.0005,

    'momentum': 0.9,
    # 'file_path': './dataset/harth_std_scaling_all_win32.csv', # 32, 64, 128
    # 'file_path': './dataset/harth_minmax_all_win250.csv',
    # 'file_path': './dataset/harth_minmax_all_win50.csv',
    'file_path': './dataset/harth_minmax_scaling_all_split_win50.csv',

    'classes': ['walking', 'running', 'shuffling', 'stairs ascending', 'stairs descending', 'standing', 'sitting',
                'lying', 'cycling sit', 'cycling stand', 'transport sit', 'transport stand'],
    'num_class': 12,

    # 22 users
    'users': ['S006', 'S008', 'S009', 'S010', 'S012', 'S013', 'S014', 'S015', 'S016', 'S017', 'S018', 'S019', 'S020',
              'S021', 'S022', 'S023', 'S024', 'S025', 'S026', 'S027', 'S028', 'S029'],

    'src_domains': ['S006_back', 'S009_back', 'S010_back', 'S012_back', 'S013_back', 'S014_back', 'S015_back', 'S016_back', 'S017_back', 'S020_back', 'S023_back', 'S024_back',
                    'S025_back', 'S026_back', 'S027_back'],
    'tgt_domains': [
        'S008_thigh',
        'S018_thigh',
        'S019_thigh',
        'S021_thigh',
        'S022_thigh',
        'S028_thigh',
        'S029_thigh'
    ],
}

ExtraSensoryOpt = {
    'name': 'extrasensory',
    'batch_size': 64,
    'seq_len': 5,
    'input_dim': 31,
    # 'learning_rate': 0.001,
    # 'weight_decay': 0,

    'learning_rate': 0.1,  # initial learning rate
    'weight_decay': 0.0005,
    'momentum': 0.9,
    'file_path': './dataset/extrasensory_selectedfeat_woutloc_std_scaling_all_win5.csv',  # 5, 10

    'classes': [
        'label:LYING_DOWN',
        'label:SITTING',
        'label:FIX_walking',
        'label:FIX_running',
        'label:OR_standing'],

    'num_class': 5,

    # 23
    'users': [
        '098A72A5-E3E5-4F54-A152-BBDA0DF7B694',
        '0A986513-7828-4D53-AA1F-E02D6DF9561B',
        '1155FF54-63D3-4AB2-9863-8385D0BD0A13',
        '1DBB0F6F-1F81-4A50-9DF4-CD62ACFA4842',
        '4FC32141-E888-4BFF-8804-12559A491D8C',
        '5119D0F8-FCA8-4184-A4EB-19421A40DE0D',
        '59818CD2-24D7-4D32-B133-24C2FE3801E5',
        '61976C24-1C50-4355-9C49-AAE44A7D09F6',
        '665514DE-49DC-421F-8DCB-145D0B2609AD',
        '74B86067-5D4B-43CF-82CF-341B76BEA0F4',
        '797D145F-3858-4A7F-A7C2-A4EB721E133C',
        '7CE37510-56D0-4120-A1CF-0E23351428D2',
        '806289BC-AD52-4CC1-806C-0CDB14D65EB6',
        '9DC38D04-E82E-4F29-AB52-B476535226F2',
        'A5A30F76-581E-4757-97A2-957553A2C6AA',
        'A5CDF89D-02A2-4EC1-89F8-F534FDABDD96',
        'A76A5AF5-5A93-4CF2-A16E-62353BB70E8A',
        'B09E373F-8A54-44C8-895B-0039390B859F',
        'B7F9D634-263E-4A97-87F9-6FFB4DDCB36C',
        'B9724848-C7E2-45F4-9B3F-A1F38D864495',
        'C48CE857-A0DD-4DDB-BEA5-3A25449B2153',
        'CF722AA9-2533-4E51-9FEB-9EAC84EE9AAC',
        'D7D20E2E-FC78-405D-B346-DBD3FD8FC92B',
    ],

    'src_domains': ['098A72A5-E3E5-4F54-A152-BBDA0DF7B694',
                    '0A986513-7828-4D53-AA1F-E02D6DF9561B',
                    '1155FF54-63D3-4AB2-9863-8385D0BD0A13',
                    '1DBB0F6F-1F81-4A50-9DF4-CD62ACFA4842',
                    '5119D0F8-FCA8-4184-A4EB-19421A40DE0D',
                    '665514DE-49DC-421F-8DCB-145D0B2609AD',
                    '74B86067-5D4B-43CF-82CF-341B76BEA0F4',
                    '7CE37510-56D0-4120-A1CF-0E23351428D2',
                    '806289BC-AD52-4CC1-806C-0CDB14D65EB6',
                    '9DC38D04-E82E-4F29-AB52-B476535226F2',
                    'A5A30F76-581E-4757-97A2-957553A2C6AA',
                    'A76A5AF5-5A93-4CF2-A16E-62353BB70E8A',
                    'B09E373F-8A54-44C8-895B-0039390B859F',
                    'B7F9D634-263E-4A97-87F9-6FFB4DDCB36C',
                    'B9724848-C7E2-45F4-9B3F-A1F38D864495',
                    'CF722AA9-2533-4E51-9FEB-9EAC84EE9AAC'],

    'tgt_domains': ['4FC32141-E888-4BFF-8804-12559A491D8C',
                    '59818CD2-24D7-4D32-B133-24C2FE3801E5',
                    '61976C24-1C50-4355-9C49-AAE44A7D09F6',
                    '797D145F-3858-4A7F-A7C2-A4EB721E133C',
                    'A5CDF89D-02A2-4EC1-89F8-F534FDABDD96',
                    'C48CE857-A0DD-4DDB-BEA5-3A25449B2153',
                    'D7D20E2E-FC78-405D-B346-DBD3FD8FC92B'],

}

RealLifeHAROpt = {
    'name': 'hhar',
    'batch_size': 64,

    'seq_len': 400,  # 150, 50, 5, 400
    # 'seq_len': 5,  # 150, 50, 5, 400

    'input_dim': 3,
    # 'input_dim': 54, #6, 54, 72, 90
    # 'input_dim': 72, #6, 54, 72, 90
    # 'input_dim': 90, #6, 54, 72, 90,
    # 'learning_rate': 0.0001,
    # 'weight_decay': 0,

    'learning_rate': 0.1,  # initial learning rate
    'weight_decay': 0.0005,
    'momentum': 0.9,

    # 'file_path': './dataset/reallifehar_acc_std_scaling_all_win50.csv', #50, 100, 150

    # 'file_path': './dataset/reallifehar_acc_gps_std_scaling_win20s_0.0_0.csv',  # 54 feats
    # 'file_path': './dataset/reallifehar_acc_magn_gps_std_scaling_win20s_0.0_1.csv', #72feats
    # 'file_path': './dataset/reallifehar_acc_gyro_magn_gps_std_scaling_win20s_0.0_2.csv', #90 feats

    # 'file_path': './dataset/reallifehar_acc_gps_std_scaling_win20s_19.0_0.csv',  # overlapping 19s, 54 feats
    # 'file_path': './dataset/reallifehar_acc_magn_gps_std_scaling_win20s_19.0_1.csv', #overlapping 19s, 72feats
    # 'file_path': './dataset/reallifehar_acc_gyro_magn_gps_std_scaling_win20s_19.0_2.csv', #overlapping 19s, 90 feats

    # 'file_path': './dataset/reallifehar_acc_minmax_scaling_all_win400_overlap380.csv',
    # 'file_path': './dataset/reallifehar_acc_std_scaling_all_win400_overlap380.csv',

    'file_path': './dataset/reallifehar_acc_minmax_scaling_all_win400_overlap0.csv',
    # 'file_path': './dataset/reallifehar_acc_std_scaling_all_win400_overlap0.csv',

    'classes': ['Inactive', 'Active', 'Walking', 'Driving'],
    'num_class': 4,

    # 19
    # 'users': ['p0', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10', 'p11', 'p12', 'p13', 'p14', 'p15', 'p16', 'p17', 'p18'], # original
    # 'users': [ 'p0', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10', 'p11', 'p12', 'p13', 'p14', 'p17', 'p18'], #./dataset/reallifehar_acc_gps_win20s_0.0_0.csv
    # 'users': [ 'p0', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10', 'p11', 'p12', 'p13', 'p17'], #./dataset/reallifehar_acc_magn_gps_win20s_0.0_1.csv
    # 'users': [ 'p0', 'p1', 'p2', 'p3', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10', 'p11', 'p12', 'p13',], #./dataset/reallifehar_acc_gyro_magn_gps_win20s_0.0_2.csv

    # 'users': ['p0', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10', 'p11', 'p12', 'p13', 'p14', 'p17', 'p18'] # original - p15 (p15 has only 1 sample..) - p16 (p16 has only 8 samples..)

    'users': ['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10', 'p11', 'p12', 'p13', 'p14', 'p17', 'p18'],
    # original -p0, - p15 (p15 has only 1 sample..) - p16 (p16 has only 8 samples..)

    'src_domains': ['p1', 'p10', 'p11', 'p13', 'p14', 'p17', 'p18', 'p3', 'p4', 'p5', 'p8'],
    'tgt_domains': ['p12', 'p2', 'p6', 'p7', 'p9'],

}
ImageNetOpt = {
    'name': 'imagenet',
    'batch_size': 64,

    'learning_rate': 0.001,
    'weight_decay': 0,
    'momentum': 0.9,
    'img_size': 3072,

    'file_path': './dataset/ImageNet-C',
    # 'classes': ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
    'num_class': 1000,
    'severity': 5,
    # 'corruptions': ["shot_noise", "motion_blur", "snow", "pixelate", "gaussian_noise", "defocus_blur",
    #                 "brightness", "fog", "zoom_blur", "frost", "glass_blur", "impulse_noise", "contrast",
    #                 "jpeg_compression", "elastic_transform"],
    'domains': ["gaussian_noise-1", "gaussian_noise-2", "gaussian_noise-3", "gaussian_noise-4", "gaussian_noise-5",
                "gaussian_noise-all",

                "shot_noise-1", "shot_noise-2", "shot_noise-3", "shot_noise-4", "shot_noise-5", "shot_noise-all",

                "impulse_noise-1", "impulse_noise-2", "impulse_noise-3", "impulse_noise-4", "impulse_noise-5",
                "impulse_noise-all",

                "defocus_blur-1", "defocus_blur-2", "defocus_blur-3", "defocus_blur-4", "defocus_blur-5",
                "defocus_blur-all",

                "glass_blur-1", "glass_blur-2", "glass_blur-3", "glass_blur-4", "glass_blur-5", "glass_blur-all",

                "motion_blur-1", "motion_blur-2", "motion_blur-3", "motion_blur-4", "motion_blur-5", "motion_blur-all",

                "zoom_blur-1", "zoom_blur-2", "zoom_blur-3", "zoom_blur-4", "zoom_blur-5", "zoom_blur-all",

                "snow-1", "snow-2", "snow-3", "snow-4", "snow-5", "snow-all",

                "frost-1", "frost-2", "frost-3", "frost-4", "frost-5", "frost-all",

                "fog-1", "fog-2", "fog-3", "fog-4", "fog-5", "fog-all",

                "brightness-1", "brightness-2", "brightness-3", "brightness-4", "brightness-5", "brightness-all",

                "contrast-1", "contrast-2", "contrast-3", "contrast-4", "contrast-5", "contrast-all",

                "elastic_transform-1", "elastic_transform-2", "elastic_transform-3", "elastic_transform-4",
                "elastic_transform-5", "elastic_transform-all",

                "pixelate-1", "pixelate-2", "pixelate-3", "pixelate-4", "pixelate-5", "pixelate-all",

                "jpeg_compression-1", "jpeg_compression-2", "jpeg_compression-3", "jpeg_compression-4",
                "jpeg_compression-5", "jpeg_compression-all",
                ],
    'src_domains': ["pixelate-1"], # dummy data
    'tgt_domains': ["gaussian_noise-5",
                    "shot_noise-5",
                    "impulse_noise-5",
                    "defocus_blur-5",
                    "glass_blur-5",
                    "motion_blur-5",
                    "zoom_blur-5",
                    "snow-5",
                    "frost-5",
                    "fog-5",
                    "brightness-5",
                    "contrast-5",
                    "elastic_transform-5",
                    "pixelate-5",
                    "jpeg_compression-5",

    ],
}


KITTI_MOT_Opt = {

    'name': 'kitti_mot',
    'batch_size': 20,
    'conf_thres': 0.1,  # object confidence threshold
    'nms_thres': 0.4,  # iou thresshold for non-maximum suppression
    'iou_thres': 0.5,  # IoU metric threshold for evaluation

    # 10, 3, 416, 416..

    'learning_rate': 0.0001,
    'weight_decay': 0,
    'momentum': 0.9,
    'img_size': 416,

    'file_path': './dataset/kitti_mot',

    'classes': ['Car', 'Van', 'Truck', 'Pedestrian', 'Person', 'Cyclist', 'Tram', 'Misc'],
    'sub_classes': ['Car', 'Van', 'Truck', 'Pedestrian', 'Person', 'Cyclist', 'Tram', 'Misc'],
    # 'sub_classes': ['Car', 'Pedestrian', 'Cyclist'],
    'num_class': 8,  # 8 #TODO: need to change config path as well
    'config_path': 'config/yolov3-kitti.cfg',
    # 'config_path': 'config/yolov3-kitti-3class.cfg',
    'domains': ['2d_detection', 'original', 'rain-100mm', 'rain-10mm'],
    # 'domains': ['half1', 'half2'],

    # 'src_domains': ['half1'],
    # 'tgt_domains': ['half2'],

    # 'src_domains': ['2d_detection'],
    # 'src_domains': ['original'],
    'src_domains': ['original-val'],

    'tgt_domains': ['rain-200-tgt'],
    'val_domains': ['rain-200-val'],

    # 'src_domains': ['rain'],
    # 'tgt_domains': ['original'],
}

HHAROpt = {
    'name': 'hhar',
    'batch_size': 64,
    'seq_len': 256,
    'input_dim': 6,
    'learning_rate': 0.001,
    'weight_decay': 0,
    'momentum': 0.9,
    # 'file_path': './dataset/hhar_minmax_scaling_all.csv',
    'file_path': './dataset/hhar_std_scaling_all.csv',

    ###---- 24 available domains for evaluation
    # 'users': ['a', 'b', 'c', 'd', 'e', 'f'],
    # 'models': ['nexus4', 's3', 's3mini', 'lgwatch'],
    # 'devices': ['lgwatch_1', 'lgwatch_2', 'gear_1', 'gear_2', 'nexus4_1', 'nexus4_2',
    #             's3_1', 's3_2', 's3mini_1', 's3mini_2'],

    'classes': ['bike', 'sit', 'stand', 'walk', 'stairsup', 'stairsdown'],
    'num_class': 6,

    #############---- For full option ----#############
    # 'users': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'],
    'users': ['a', 'b', 'c', 'd', 'e', 'f', 'g'],  # h, i has no gear data (h has no labels for some activities)
    # 'models': ['phone','lgwatch', 'gear'],
    # 'models': ['nexus4', 's3', 's3mini', 'watch'],
    # 'models': ['phone','watch'],
    'models': ['nexus4', 's3', 's3mini', 'lgwatch', 'gear'],
    'devices': ['lgwatch_1', 'lgwatch_2', 'gear_1', 'gear_2', 'nexus4_1', 'nexus4_2',
                's3_1', 's3_2', 's3mini_1', 's3mini_2'],

    ##------------------------------------------------##
    # collected each activity for 5 minutes with maximum sampling rates below:
    # nexus4: 200Hz
    # s3: 150Hz
    # s3mini: 100Hz
    # lgwatch: 200Hz
    # gear: 100Hz
}

WESADOpt = {
    'name': 'wesad',
    'input_dim': 10,
    'seq_len': 8,
    'file_path': './dataset/wesad_both_minmax_scaling_all.csv',
    'learning_rate': 0.001,
    'weight_decay': 0,
    'momentum': 0.9,
    'batch_size': 32,
    'domains': ['S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11',
                'S13', 'S14', 'S15', 'S16', 'S17'],  # 15 users
    'columns': ['chestAcc', 'chestECG', 'chestEMG', 'chestEDA', 'chestTemp', 'chestResp',
                'wristAcc', 'wristBVP', 'wristEDA', 'wristTemp'],  # features
    'classes': ['baseline', 'stress', 'amusement', 'mediation'],
    'num_class': 4,
}

ICHAROpt = {
    'name': 'ichar',
    'batch_size': 32,
    'seq_len': 256,
    'ecdf_bin_size': 50,
    'file_path': './dataset/ichar_std_scaling_all.csv',
    # 'file_path': './dataset/extrasensory_selectedfeat_woutloc_std_scaling_all_5.csv',
    'input_dim': 6,
    'learning_rate': 0.001,
    'weight_decay': 0,
    'momentum': 0.9,

    ###---- 10 available domains for evaluation (full)
    'domains': ['PH0007-jskim', 'PH0012-thanh', 'PH0014-wjlee', 'PH0034-ykha', 'PH0038-iygoo', 'PH0041-hmkim',
                'PH0045-sjlee', 'WA0002-bkkim', 'WA0003-hskim', 'WA4697-jhryu'],
    'domain_information': {
        'PH0007-jskim': {
            'sampling_rate': 100
        },
        'PH0012-thanh': {
            'sampling_rate': 200
        },
        'PH0014-wjlee': {
            'sampling_rate': 400
        },
        'PH0034-ykha': {
            'sampling_rate': 400
        },
        'PH0038-iygoo': {
            'sampling_rate': 500
        },
        'PH0041-hmkim': {
            'sampling_rate': 500
        },
        'PH0045-sjlee': {
            'sampling_rate': 200
        },
        'WA0002-bkkim': {
            'sampling_rate': 200
        },
        'WA0003-hskim': {
            'sampling_rate': 100
        },
        'WA4697-jhryu': {
            'sampling_rate': 100
        }
    },

    'classes': ['walking', 'running', 'sitting', 'standing', 'lying',
                'stairup', 'stairdown', 'jumping', 'stretching'],
    'num_class': 9,
}

ICSROpt = {
    'name': 'icsr',
    'batch_size': 32,
    'seq_len': 32000,
    # 'file_path': './dataset/icsr_minmax_scaling_all.csv',
    'file_path': './dataset/icsr_std_scaling_all.csv',
    'input_dim': 1,
    'learning_rate': 0.001,
    'weight_decay': 0,
    'momentum': 0.9,

    ###---- 10 available domains for evaluation (full)
    'domains': ['PH0007-jskim', 'PH0012-thanh', 'PH0014-wjlee', 'PH0034-ykha', 'PH0038-iygoo', 'PH0041-hmkim',
                'PH0045-sjlee', 'WA0002-bkkim', 'WA0003-hskim', 'WA4697-jhryu'],
    'domain_information': {
        'PH0007-jskim': {
            'sampling_rate': 100
        },
        'PH0012-thanh': {
            'sampling_rate': 200
        },
        'PH0014-wjlee': {
            'sampling_rate': 400
        },
        'PH0034-ykha': {
            'sampling_rate': 400
        },
        'PH0038-iygoo': {
            'sampling_rate': 500
        },
        'PH0041-hmkim': {
            'sampling_rate': 500
        },
        'PH0045-sjlee': {
            'sampling_rate': 200
        },
        'WA0002-bkkim': {
            'sampling_rate': 200
        },
        'WA0003-hskim': {
            'sampling_rate': 100
        },
        'WA4697-jhryu': {
            'sampling_rate': 100
        }
    },
    ###--- End

    'classes': ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off',
                'stop', 'go', 'forward', 'backward', 'follow', 'learn', ],
    'num_class': 14,
}

Office31_Opt = {
    'name': 'office31',
    'batch_size': 36,
    'learning_rate': 0.001,
    'prep': {"test_10crop": True, "params": {"resize_size": 256, "crop_size": 224, "alexnet": False}},
    'loss': {"trade_off": 1.0},
    'network': {"name": "ResNetFc", "params": {"use_bottleneck": True, "bottleneck_dim": 256, "new_cls": True}},
    'seq_len': 256,
    'weight_decay': 0,
    'momentum': 0.9,
    'file_path': {
        'amazon': './image_dataset/data/office/amazon_list.txt',
        'webcam': './image_dataset/data/office/webcam_list.txt',
        'dslr': './image_dataset/data/office/dslr_list.txt'
    },
    'classes': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18',
                '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30'],
    'num_class': 31
}

DSAOpt = {
    'name': 'dsa',
    'batch_size': 32,
    'seq_len': 125,
    'input_dim': 9,
    'learning_rate': 0.0001,
    'weight_decay': 0,
    'momentum': 0.9,
    # 'file_path': './dataset/dsa_minmax_scaling_all.csv',
    'file_path': './dataset/dsa_std_scaling_all.csv',

    ###---- 40 available domains for evaluation (full)
    'users': ['p' + str(i) for i in range(1, 9)],
    'positions': ['T', 'RA', 'LA', 'RL', 'LL'],

    'activities': ['a0' + str(i) if (i < 10) else 'a' + str(i) for i in range(1, 20)],
    'classes': ['sitting', 'standing', 'lying_on_back', 'lying_on_right', 'ascending_stairs',
                'descending_stairs', 'standing_in_elevator', 'moving_in_elevator', 'walking_in_parking_lot',
                'walking_on_treadmill_4_flat', 'walking_on_treadmill_4_inclined', 'running_on_treadmill_8',
                'exercising_on_stepper', 'exercising_on_cross_trainer', 'cycling_on_bike_horizontal',
                'cycling_on_bike_vertical', 'rowing', 'jumping', 'playing_basketball'],
    'num_class': 19,

}
# include conf for PAMAP2 dataset
PAMAP2Opt = {
    'name': 'pamap2',
    'batch_size': 32,
    'seq_len': 256,
    'input_dim': 9,
    'learning_rate': 0.0001,
    'weight_decay': 0,
    # 'file_path': './dataset/pamap2_minmax_scaling_all.csv',
    'file_path': './dataset/pamap2_std_scaling_all.csv',

    ###---- 27 available domains for SenSys'19 evaluation (full)
    'users': ['subject10' + str(i) for i in range(1, 9)],  # make sure to include all 8!!
    # 'users': ['subject101', 'subject102', 'subject105', 'subject107', 'subject108'],
    'positions': ['Arm', 'Chest', 'Ankle'],

    'classes': ['lying', 'sitting', 'standing', 'walking', 'ascending_stairs', 'descending_stairs', 'vacuum_cleaning',
                'ironing'],
    'num_class': 8,
    # 'classes': ['lying', 'sitting', 'ascending_stairs', 'descending_stairs', 'vacuum_cleaning', 'ironing'],
    # # 'classes': ['lying', 'sitting'],
    # 'num_class': 6
}

GaitOpt = {
    'name': 'gait',
    'batch_size': 64,
    'seq_len': 192,
    'file_path': './dataset/gait_std_scaling_all.csv',
    'input_dim': 3,
    'learning_rate': 0.0001,
    'weight_decay': 0,
    'fc_parameters': {
        "mean": None,
        "standard_deviation": None,
        "maximum": None,
        "minimum": None,
        "abs_energy": None,  # energy
        "fft_aggregated": [{"aggtype": "centroid"}, {"aggtype": "variance"}, {"aggtype": "skew"},
                           {"aggtype": "kurtosis"}],
        "fft_coefficient": []
    },

    ###---- 24 available domains for SenSys'19 evaluation
    'users': ['p01', 'p02', 'p03', 'p05', 'p06', 'p08', 'p09'],
    'positions': ['Ankle', 'UpperLeg', 'Trunk'],
    ###--- End

    ###---- 9 domains for lr tune
    #     'users': ['p01', 'p02', 'p03'],
    #     'positions': ['Ankle', 'UpperLeg', 'Trunk'],
    ###---- End

    ###--- All domains (30) for data generation
    # 'users': ['p0' + str(i) if (i < 10) else 'p'+str(i) for i in range(1, 11)],
    # 'positions': ['Ankle', 'UpperLeg', 'Trunk'],
    ###--- End

    'classes': ['no_freeze', 'freeze'],
    'num_class': 2,
}

# sampling rate is 30Hz, go for => seq_len = 60
OpportunityOpt = {
    'batch_size': 64,
    'seq_len': 60,
    'input_dim': 6,
    'learning_rate': 0.0001,
    'weight_decay': 0,
    'file_path': './dataset/opportunity_std_all.csv',
    # 4 x 7 = 28 available domains
    'users': ['s' + str(i) for i in range(1, 5)],  # 4 available users
    'positions': ['RUA', 'LUA', 'RLA', 'LLA', 'Back', 'L_Shoe', 'R_Shoe'],  # complete IMU sensor data only
    'classes': ['stand', 'walk', 'sit', 'lie'],  # remove null class
    'num_class': 4,
}


# processed_domains = list(np.random.permutation(args.opt['domains'], size=args.opt['num_src']))

def init_domains():
    seed = 0
    import random
    import numpy as np
    np.random.seed(seed)
    random.seed(seed)
    import math

    test_size = math.ceil(len(HARTHOpt['users']) * 0.3)
    HARTHOpt['tgt_domains'] = sorted(list(np.random.permutation(HARTHOpt['users'])[:test_size]))
    HARTHOpt['src_domains'] = sorted(list(set(HARTHOpt['users']) - set(HARTHOpt['tgt_domains'])))

    test_size = math.ceil(len(ExtraSensoryOpt['users']) * 0.3)
    ExtraSensoryOpt['tgt_domains'] = sorted(list(np.random.permutation(ExtraSensoryOpt['users'])[:test_size]))
    ExtraSensoryOpt['src_domains'] = sorted(list(set(ExtraSensoryOpt['users']) - set(ExtraSensoryOpt['tgt_domains'])))

    test_size = math.ceil(len(RealLifeHAROpt['users']) * 0.3)
    RealLifeHAROpt['tgt_domains'] = sorted(list(np.random.permutation(RealLifeHAROpt['users'])[:test_size]))
    RealLifeHAROpt['src_domains'] = sorted(list(set(RealLifeHAROpt['users']) - set(RealLifeHAROpt['tgt_domains'])))

    # 15 ['S006', 'S009', 'S010', 'S012', 'S013', 'S014', 'S015', 'S016', 'S017', 'S020', 'S023', 'S024', 'S025', 'S026', 'S027']
    # 7 ['S008', 'S018', 'S019', 'S021', 'S022', 'S028', 'S029']

    # 16 ['098A72A5-E3E5-4F54-A152-BBDA0DF7B694', '0A986513-7828-4D53-AA1F-E02D6DF9561B', '1155FF54-63D3-4AB2-9863-8385D0BD0A13', '1DBB0F6F-1F81-4A50-9DF4-CD62ACFA4842', '5119D0F8-FCA8-4184-A4EB-19421A40DE0D', '665514DE-49DC-421F-8DCB-145D0B2609AD', '74B86067-5D4B-43CF-82CF-341B76BEA0F4', '7CE37510-56D0-4120-A1CF-0E23351428D2', '806289BC-AD52-4CC1-806C-0CDB14D65EB6', '9DC38D04-E82E-4F29-AB52-B476535226F2', 'A5A30F76-581E-4757-97A2-957553A2C6AA', 'A76A5AF5-5A93-4CF2-A16E-62353BB70E8A', 'B09E373F-8A54-44C8-895B-0039390B859F', 'B7F9D634-263E-4A97-87F9-6FFB4DDCB36C', 'B9724848-C7E2-45F4-9B3F-A1F38D864495', 'CF722AA9-2533-4E51-9FEB-9EAC84EE9AAC']
    # 7 ['4FC32141-E888-4BFF-8804-12559A491D8C', '59818CD2-24D7-4D32-B133-24C2FE3801E5', '61976C24-1C50-4355-9C49-AAE44A7D09F6', '797D145F-3858-4A7F-A7C2-A4EB721E133C', 'A5CDF89D-02A2-4EC1-89F8-F534FDABDD96', 'C48CE857-A0DD-4DDB-BEA5-3A25449B2153', 'D7D20E2E-FC78-405D-B346-DBD3FD8FC92B']

    # 11 ['p1', 'p10', 'p11', 'p13', 'p14', 'p17', 'p18', 'p3', 'p4', 'p5', 'p8']
    # 5 ['p12', 'p2', 'p6', 'p7', 'p9']

    '''
    # single src_domain
    # HARTHOpt['src_domains'] = sorted(list(np.random.permutation(HARTHOpt['users'])[:int(len(HARTHOpt['users']) * 0.7)]))
    # HARTHOpt['tgt_domains'] = sorted(list(set(HARTHOpt['users']) - set(HARTHOpt['src_domains'])))
    #
    # RealLifeHAROpt['src_domains'] = sorted(list(np.random.permutation(RealLifeHAROpt['users'])[:int(len(RealLifeHAROpt['users']) * 0.7)]))
    # RealLifeHAROpt['tgt_domains'] = sorted(list(set(RealLifeHAROpt['users']) - set(RealLifeHAROpt['src_domains'])))
    #
    # ExtraSensoryOpt['src_domains'] = sorted(list(np.random.permutation(ExtraSensoryOpt['users'])[:int(len(ExtraSensoryOpt['users']) * 0.7)]))
    # ExtraSensoryOpt['tgt_domains'] = sorted(list(set(ExtraSensoryOpt['users']) - set(ExtraSensoryOpt['src_domains'])))

    ICHAROpt['src_domains'] = sorted(list(np.random.permutation(ICHAROpt['domains'])[:7]))
    ICHAROpt['tgt_domains'] = sorted(
        list(set(ICHAROpt['domains']) - set(ICHAROpt['src_domains'])))
    # 7,34,41

    # MetaSense_ActivityOpt['src_domains'] = ['PH0012-thanh', 'PH0014-wjlee', 'PH0038-iygoo', 'PH0045-sjlee', 'WA0002-bkkim', 'WA0003-hskim',
    #  'WA4697-jhryu']
    # MetaSense_ActivityOpt['tgt_domains'] = sorted(list(set(MetaSense_ActivityOpt['domains']) - set(MetaSense_ActivityOpt['src_domains'])))

    ICSROpt['src_domains'] = ICHAROpt['src_domains']
    ICSROpt['tgt_domains'] = ICHAROpt['tgt_domains']

    src_users = list(np.random.permutation(HHAROpt['users'])[:4])
    src_models = list(np.random.permutation(HHAROpt['models'])[:3])
    tgt_users = list(set(HHAROpt['users']) - set(src_users))
    tgt_models = list(set(HHAROpt['models']) - set(src_models))
    devices = HHAROpt['devices']
    HHAROpt['src_domains'] = [('.').join(x) for x in sorted(list(itertools.product(src_users, src_models)))]
    HHAROpt['tgt_domains'] = [('.').join(x) for x in sorted(list(itertools.product(tgt_users, tgt_models)))]

    src_users = list(np.random.permutation(DSAOpt['users'])[:5])
    src_poss = list(np.random.permutation(DSAOpt['positions'])[:3])
    tgt_users = list(set(DSAOpt['users']) - set(src_users))
    tgt_poss = list(set(DSAOpt['positions']) - set(src_poss))
    DSAOpt['src_domains'] = sorted(list(itertools.product(src_users, src_poss)))
    DSAOpt['tgt_domains'] = sorted(list(itertools.product(tgt_users, tgt_poss)))

    src_users = list(np.random.permutation(PAMAP2Opt['users'])[:4])
    src_poss = list(np.random.permutation(PAMAP2Opt['positions'])[:2])
    tgt_users = list(set(PAMAP2Opt['users']) - set(src_users))
    tgt_poss = list(set(PAMAP2Opt['positions']) - set(src_poss))
    PAMAP2Opt['src_domains'] = sorted(list(itertools.product(src_users, src_poss)))
    PAMAP2Opt['tgt_domains'] = sorted(list(itertools.product(tgt_users, tgt_poss)))

    WESADOpt['src_domains'] = sorted(list(np.random.permutation(WESADOpt['domains'])[:10]))
    WESADOpt['tgt_domains'] = sorted(list(set(WESADOpt['domains']) - set(WESADOpt['src_domains'])))

    src_users = list(np.random.permutation(OpportunityOpt['users'])[:2])
    src_poss = list(np.random.permutation(OpportunityOpt['positions'])[:5])
    tgt_users = list(set(OpportunityOpt['users']) - set(src_users))
    tgt_poss = list(set(OpportunityOpt['positions']) - set(src_poss))
    OpportunityOpt['src_domains'] = sorted(list(itertools.product(src_users, src_poss)))
    OpportunityOpt['tgt_domains'] = sorted(list(itertools.product(tgt_users, tgt_poss)))

    src_users = list(np.random.permutation(GaitOpt['users'])[:4])
    src_poss = list(np.random.permutation(GaitOpt['positions'])[:2])
    tgt_users = list(set(GaitOpt['users']) - set(src_users))
    tgt_poss = list(set(GaitOpt['positions']) - set(src_poss))
    GaitOpt['src_domains'] = sorted(list(itertools.product(src_users, src_poss)))
    GaitOpt['tgt_domains'] = sorted(list(itertools.product(tgt_users, tgt_poss)))
    '''
    print(len(HARTHOpt['src_domains']), HARTHOpt['src_domains'])
    print(len(HARTHOpt['tgt_domains']), HARTHOpt['tgt_domains'])

    print(len(ExtraSensoryOpt['src_domains']), ExtraSensoryOpt['src_domains'])
    print(len(ExtraSensoryOpt['tgt_domains']), ExtraSensoryOpt['tgt_domains'])

    print(len(RealLifeHAROpt['src_domains']), RealLifeHAROpt['src_domains'])
    print(len(RealLifeHAROpt['tgt_domains']), RealLifeHAROpt['tgt_domains'])

    '''
    print(len(ICHAROpt['src_domains']), ICHAROpt['src_domains'])
    print(len(ICHAROpt['tgt_domains']), ICHAROpt['tgt_domains'])

    print(len(ICSROpt['src_domains']), ICSROpt['src_domains'])
    print(len(ICSROpt['tgt_domains']), ICSROpt['tgt_domains'])

    print(len(HHAROpt['src_domains']), HHAROpt['src_domains'])
    print(len(HHAROpt['tgt_domains']), HHAROpt['tgt_domains'])

    print(len(WESADOpt['src_domains']), WESADOpt['src_domains'])
    print(len(WESADOpt['tgt_domains']), WESADOpt['tgt_domains'])

    #
    # print(len(DSAOpt['src_domains']), DSAOpt['src_domains'])
    # print(len(DSAOpt['tgt_domains']), DSAOpt['tgt_domains'])
    #
    # print(len(PAMAP2Opt['src_domains']), PAMAP2Opt['src_domains'])
    # print(len(PAMAP2Opt['tgt_domains']), PAMAP2Opt['tgt_domains'])
    #
    # print(len(OpportunityOpt['src_domains']), OpportunityOpt['src_domains'])
    # print(len(OpportunityOpt['tgt_domains']), OpportunityOpt['tgt_domains'])
    #
    # print(len(GaitOpt['src_domains']),GaitOpt['src_domains'] )
    # print(len(GaitOpt['tgt_domains']),GaitOpt['tgt_domains'] )
    '''


if __name__ == "__main__":
    init_domains()
