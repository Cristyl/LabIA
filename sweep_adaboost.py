import torch
import random
import numpy as np
import wandb
from grapefeatureextractor import GrapeFeatureExtractor
import yaml
from joblib import dump
from torch.utils.data import DataLoader
from brixcolordataset import BrixColorDataset
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from PIL import Image
from pathlib import Path
import torchvision.transforms as T
from sklearn.metrics import accuracy_score

SEED = 0
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

BRIXCOLOR_SPLIT = "MixedSeason_"

def apply_params_and_extract_features(config, images_tr, images_te):
    cross = None
    if config.cross != 'none':
        if config.cross == 'thin':
            cross = [config.n_bin_y / 4, config.n_bin_y - config.n_bin_y / 4,
                     config.n_bin_x / 4, config.n_bin_x / 2
                     ]
        elif config.cross == 'fat':
            cross = [config.n_bin_y / 4, config.n_bin_y - config.n_bin_y / 4,
                     config.n_bin_x / 4, config.n_bin_x - config.n_bin_x / 4
                     ]

    # extract features
    extractor = GrapeFeatureExtractor()
    kwargs = dict()
    if config.pixels_per_cell != 'None':
        kwargs['pixels_per_cell'] = config.pixels_per_cell
    else:
        kwargs['pixels_per_cell'] = None
    X_tr = extractor.extract(images_tr, config.n_bin_y, config.n_bin_x, cross=cross, hsv=config.hsv,
                             hog_type=config.hog_type, only_hog=config.only_hog, **kwargs)
    X_te = extractor.extract(images_te, config.n_bin_y, config.n_bin_x, cross=cross, hsv=config.hsv,
                             hog_type=config.hog_type, only_hog=config.only_hog, **kwargs)
    return X_tr, X_te, cross

def colorDataset():
    #crea il dataset training
    dataset_tr = BrixColorDataset(Path("./data"), std_split=BRIXCOLOR_SPLIT + "train")

    #dividi in training e validation
    sampler_tr, sampler_val, _, _ = train_test_split(
        list(range(len(dataset_tr))), dataset_tr.labels, test_size=0.2,
        random_state=42,  stratify=dataset_tr.labels['color'])

    #carica i dati del training e le label
    dataloader_tr = DataLoader(dataset_tr, batch_size=1, shuffle=False, sampler=sampler_tr)
    images_tr = list()
    labels_tr = list()
    for data, target in dataloader_tr:
        images_tr.append(Image.fromarray(data.numpy().squeeze().transpose(1, 2, 0)))
        labels_tr.append(target['color'])
    im_w, im_h = images_tr[0].size[0:2]

    #trasformazioni sulle immagini concatenate in seguito
    horizontalflip_transfomer = T.RandomHorizontalFlip(p=1.0)
    new_images = [horizontalflip_transfomer(im) for im in images_tr]
    images_tr = images_tr + new_images
    labels_tr += labels_tr
    ds = 0.2
    perspective_transformer = T.RandomPerspective(distortion_scale=ds, p=1.0, fill=255)
    centercrop_transformer = T.CenterCrop(size=(im_h-ds*im_h, im_w-ds*im_w))
    resize_transfomrmer = T.Resize(size=(im_h, im_w))
    compose_transformer = T.Compose([perspective_transformer, centercrop_transformer, resize_transfomrmer])
    new_images = [compose_transformer(im) for im in images_tr]
    images_tr = images_tr + new_images
    labels_tr += labels_tr

    #carica i dati del test
    dataloader_val = DataLoader(dataset_tr, batch_size=1, shuffle=False, sampler=sampler_val)
    images_val = list()
    labels_val = list()
    for data, target in dataloader_val:
        images_val.append(Image.fromarray(data.numpy().squeeze().transpose(1, 2, 0)))
        labels_val.append(target['color'])

    #set wandb
    with open(Path("./config_adaboost.yaml")) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    wandb.init(config=config)
    config = wandb.config

    #estrai feature
    X_tr, X_val, _ = apply_params_and_extract_features(config, images_tr, images_val)
    y_tr = labels_tr
    y_val = labels_val

    return X_tr, X_val, y_tr, y_val, config

def adaboost_sweep():
    #set dati necessari
    X_tr, X_val, y_tr, y_val, config = colorDataset()

    base_model = DecisionTreeClassifier(max_depth=config["max_depth"], min_samples_split=config["min_samples_split"], min_samples_leaf=config["min_samples_leaf"])
    model = AdaBoostClassifier(base_estimator=base_model, n_estimators=config["n_estimators"], learning_rate=config["learning_rate"], random_state=42)

    model.fit(X_tr, y_tr)

    #predict del modello
    y_val_hat = model.predict(X_val)

    #parametri da valutare
    accuracy = accuracy_score(y_val_hat, y_val)
    wandb.log({
        "accuracy": accuracy,
    })
    #salva il modello
    dump(model, Path(".") / "saved_models_adaboost.joblib")