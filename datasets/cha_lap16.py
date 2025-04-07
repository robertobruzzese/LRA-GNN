import sys
from collections import defaultdict
import csv
import numpy as np
import pickle
import os
from tqdm import tqdm
from cv2 import cv2

# Assuming vgg2_dataset.py and dataset_tools.py are similar to those in chalearn_lap.py
from vgg2_dataset import PARTITION_TRAIN, PARTITION_VAL, PARTITION_TEST
sys.path.append("../training")
from dataset_tools import DataGenerator, enclosing_square, add_margin

# ChaLearn LAP 2016 dataset specific settings
NUM_CLASSES = 1  # For regression purposes
CROPPED_SUFFIX = "_face.jpg"  # For non-aligned dataset
EXT_ROOT = os.path.dirname(os.path.abspath(__file__))
lap16_ages = defaultdict(dict)

def get_roi_lap16(d):

    """Extracts the region of interest (ROI) from the metadata."""
    return (int(d[0]), int(d[1]), int(d[2]), int(d[3]))

# Loading structured data
def _get_structured_lap16_meta(metacsv):
    """Parses the structured metadata from a CSV file."""
    data = dict()
    csv_array = _readcsv(metacsv)
    for line in csv_array[1:]:  # Exclude header
        data[line[0]] = {
            "cropped": line[0],
            "apparent_age": line[1],  # Adjust indices based on ChaLearn LAP 2016 CSV format
            "real_age": line[2],
            "roi": get_roi_lap16(line[3:7]),
        }
    return data

def structured_lap16_data_wrapper(partition):

    """Wraps the metadata loading function for different partitions."""
    metacsv = 'chalearn_lap16/gt_avg_<part>.csv'
    metapart = get_metapartition_label(get_partition_label(partition))
    metacsv = os.path.join(EXT_ROOT, metacsv.replace("<part>", metapart))
    return _get_structured_lap16_meta(metacsv)

def _load_ages(metacsv, partition):

    """Loads age data for a specific partition."""
    global lap16_ages
    if lap16_ages is None or partition not in lap16_ages or lap16_ages[partition] is None:
        lap16_ages[partition] = _get_structured_lap16_meta(metacsv)

def get_age_label(floating_string, precision=3):

    """Converts a floating-point age string to a rounded float."""
    return float(floating_string) if precision is None else np.round(float(floating_string), precision)

def _readcsv(csvpath, debug_max_num_samples=None):

    """Reads a CSV file and returns its contents as a NumPy array."""
    data = list()
    with open(csvpath, newline='', encoding="utf8") as csvfile:
        reader = csv.reader(csvfile, skipinitialspace=True, delimiter=',', quotechar='|')
        i = 0
        for row in reader:
            if debug_max_num_samples is not None and i >= debug_max_num_samples:
                break
            i += 1
            data.append(row)
    return np.array(data)

def get_partition_label(partition):

    """Maps partition names to their corresponding labels."""
    if partition == 'train':
        return PARTITION_TRAIN
    elif partition == 'val':
        return PARTITION_VAL
    elif partition == 'test':
        return PARTITION_TEST
    else:
        raise Exception("Unknown partition")

def get_metapartition_label(partition_label):

    """Maps partition labels to their directory names."""
    if partition_label == PARTITION_TRAIN:
        return 'train'
    elif partition_label == PARTITION_VAL:
        return 'valid'
    elif partition_label == PARTITION_TEST:
        return 'test'
    else:
        raise Exception("Unknown meta partition")

def get_age_from_lap16(path, metacsv, partition):

    """Retrieves age and ROI data for a given image path."""
    _load_ages(metacsv, partition)
    try:
        cropped = lap16_ages[partition][path]['cropped']
        age = lap16_ages[partition][path]['apparent_age']
        roi = lap16_ages[partition][path]['roi']
        return cropped, age, roi
    except KeyError:
        return None, None, None

def get_prebuilt_roi(filepath):

    """Loads pre-built ROI data from a .mat file."""
    if filepath.endswith(CROPPED_SUFFIX):
        filepath = filepath[:-len(CROPPED_SUFFIX)]
    filepath = filepath + ".mat"
    startx, starty, endx, endy, _, _ = loadmat(filepath)['fileinfo']['face_location'][0][0][0]
    roi = (startx, starty, endx-startx, endy-starty)
    roi = enclosing_square(roi)
    roi = add_margin(roi, 0.2)
    return roi

# Load dataset
def _load_dataset(meta, csvmeta, imagesdir, partition, cropped=True):

    """Loads the dataset, including image reading, ROI cropping, and data preprocessing."""
    data = []
    n_discarded = 0
    for item in tqdm(meta[1:]):  # Exclude header
        image_path = item[0]
        cropped_image_path, apparent_age, roi = get_age_from_lap16(image_path, csvmeta, partition)
        complete_image_path = os.path.join(imagesdir, cropped_image_path if cropped else image_path)
        partition_label = get_partition_label(partition)

        img = cv2.imread(complete_image_path)

        if img is None:
            print("Unable to read the image:", complete_image_path)
            n_discarded += 1
            continue

        if np.max(img) == np.min(img):
            print('Blank image, sample discarded:', complete_image_path)
            n_discarded += 1
            continue

        if roi == (0, 0, 0, 0) or roi is None:
            print("No face detected, entire sample added:", complete_image_path)
            roi = (0, 0, img.shape[1], img.shape[0])

        example = {
            'img': complete_image_path,
            'label': get_age_label(apparent_age),
            'roi': roi,
            'part': partition_label
        }

        data.append(example)
    print("Data loaded. {} samples ({} discarded)".format(len(data), n_discarded))
    return data

def _load_lap16(csvmeta, imagesdir, partition, debug_max_num_samples=None):

    """Loads the ChaLearn LAP 2016 dataset for a specific partition."""
    metapartition = get_metapartition_label(get_partition_label(partition))
    print("Directory partition:", metapartition)
    lap16_partition_dir = imagesdir.replace('<part>', metapartition)
    lap16_partition_csv = csvmeta.replace('<part>', metapartition)
    lap16_partition_meta = _readcsv(lap16_partition_csv, debug_max_num_samples)
    print("CSV {} read complete: {} samples".format(lap16_partition_csv, len(lap16_partition_meta)))
    return _load_dataset(lap16_partition_meta, lap16_partition_csv, lap16_partition_dir, partition)

class LAP16Age:

    """Class for handling ChaLearn LAP 2016 dataset loading and preprocessing."""
    def __init__(self,
                 partition='train',
                 imagesdir='chalearn_lap16/<part>',
                 csvmeta='chalearn_lap16/gt_avg_<part>.csv',
                 target_shape=(224, 224, 3),
                 augment=True,
                 custom_augmentation=None,
                 preprocessing='full_normalization',
                 method='apparent',
                 debug_max_num_samples=None):
        
        self.target_shape = target_shape
        self.custom_augmentation = custom_augmentation
        self.augment = augment
        self.gen = None
        self.preprocessing = preprocessing
        print('Loading %s data...' % partition)

        num_samples = "_" + str(debug_max_num_samples) if debug_max_num_samples is not None else ''
        cache_file_name = 'lap16_age_{method}_{partition}{num_samples}.cache'.format(method=method, partition=partition, num_samples=num_samples)
        cache_file_name = os.path.join("dataset_cache", cache_file_name)
        cache_file_name = os.path.join(EXT_ROOT, cache_file_name)
        print("Cache file name %s" % cache_file_name)

        try:
            with open(cache_file_name, 'rb') as f:
                self.data = pickle.load(f)
                self.data = self.data[:debug_max_num_samples]
                print("Data loaded. %d samples, from cache" % (len(self.data)))
        except FileNotFoundError:

            if partition == "all":
                self.data = list()
                for partition in ["train", "val", "test"]:
                   