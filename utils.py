from pathlib import Path
import numpy as np
import imageio
from collections.abc import MutableMapping
import os
import shutil
import zipfile


def load_and_crop(image_folder, annotation_folder, save=False, savepath=None):
    """
    Load images in a folder and crop them according to yolo bounding boxes files
    :param image_folder
    :param annotation_folder
    :param save: save the cropped file to disk
    :param savepath: specifies a folder where to put the cropped files. If None, the cropped files will be put in a
    "cropped" sub-folder.
    :return: a list of image crops as numpy arrays
    """
    image_folder = Path(image_folder)
    annotation_folder = Path(annotation_folder)
    if savepath:
        savepath = Path(savepath)
        if not savepath.exists():
            savepath.mkdir()
    else:
        savepath = image_folder / 'cropped'
        if not savepath.exists():
            savepath.mkdir()

    image_list = sorted([x for x in Path(image_folder).iterdir() if x.is_file()])
    label_list = sorted([x for x in Path(annotation_folder).iterdir() if x.is_file()])
    if len(image_list) != len(label_list):
        raise ValueError(
            f'The number of images found in {image_folder}\n is different from the number of annotations files in {annotation_folder}')

    cropped_list = list()
    for f_image, f_label in zip(image_list, label_list):
        bboxes = np.loadtxt(annotation_folder / f_label.name)
        # if the annotation contains only one bounding box, extend the array dimension
        if len(bboxes.shape) < 2:
            bboxes = np.expand_dims(bboxes, axis=0)
        bboxes = bboxes[:, 1:]  # drop 'class label' column
        image = imageio.imread(image_folder / f_image.name)
        h, w = image.shape[0:2]
        # for each bounding box in the image
        for i, bbox in enumerate(bboxes):
            bbox = [bbox[0] * w, bbox[1] * h, bbox[2] * w, bbox[3] * h]
            a = int(bbox[1] - bbox[3] / 2)
            b = int(bbox[1] + bbox[3] / 2)
            c = int(bbox[0] - bbox[2] / 2)
            d = int(bbox[0] + bbox[2] / 2)
            cropped_image = image[a:b, c:d]
            if save:
                stem, ext = f_image.name.split('.')
                imageio.imwrite(savepath / (stem + f'_{i}.' + ext), cropped_image)
            cropped_list.append(cropped_image)
    return cropped_list


# Turns a dictionary into a class
class Dict2Class(object):

    def __init__(self, my_dict):
        for key in my_dict:
            setattr(self, key, my_dict[key])


def compute_hist_feature_len(cross, n_bin_x, n_bin_y):
    out = None
    if cross != 'none':
        if cross == 'thin':
            out = [n_bin_y / 4, n_bin_y - n_bin_y / 4,
                   n_bin_x / 4, n_bin_x / 2
                   ]
        elif cross == 'fat':
            out = [n_bin_y / 4, n_bin_y - n_bin_y / 4,
                   n_bin_x / 4, n_bin_x - n_bin_x / 4
                   ]
    nf = int((out[1] - out[0]) * (out[3] - out[2]))
    return out, nf


def flatten_dict(d: MutableMapping, parent_key: str = '', sep: str = '.') -> MutableMapping:
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def flat_unzip(out_dir, zipfile_path):
    """
    Extracts all files from a zip archive without keeping the directory structure.
    :return:
    """
    with zipfile.ZipFile(zipfile_path) as zip_file:
        for member in zip_file.namelist():
            filename = os.path.basename(member)
            # skip directories
            if not filename:
                continue

            # copy file (taken from zipfile's extract)
            source = zip_file.open(member)
            target = open(os.path.join(out_dir, filename), "wb")
            with source, target:
                shutil.copyfileobj(source, target)