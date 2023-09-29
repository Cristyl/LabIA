import torch
import random
import numpy as np
import wandb
from brixcolordataset import BrixColorDataset
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader
from PIL import Image
import torchvision.transforms as T
from grapefeatureextractor import GrapeFeatureExtractor
import yaml
from joblib import dump

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
    print(config.n_bin_y)
    print(config["n_bin_y"])
    X_tr = extractor.extract(images_tr, config.n_bin_y, config.n_bin_x, cross=cross, hsv=config.hsv,
                             hog_type=config.hog_type, only_hog=config.only_hog, **kwargs)
    X_te = extractor.extract(images_te, config.n_bin_y, config.n_bin_x, cross=cross, hsv=config.hsv,
                             hog_type=config.hog_type, only_hog=config.only_hog, **kwargs)
    return X_tr, X_te, cross


def brixDataset():
    #crea il dataset training
    dataset_tr = BrixColorDataset(Path("./data"), std_split=BRIXCOLOR_SPLIT + "train")

    #dividi in training e validation
    sampler_tr, sampler_val, _, _ = train_test_split(
        list(range(len(dataset_tr))), dataset_tr.labels, test_size=0.2,
        random_state=42)

    #carica i dati del training e le label
    dataloader_tr = DataLoader(dataset_tr, batch_size=1, shuffle=False, sampler=sampler_tr)
    images_tr = list()
    labels_tr = list()
    for data, target in dataloader_tr:
        images_tr.append(Image.fromarray(data.numpy().squeeze().transpose(1, 2, 0)))
        labels_tr.append(target['brix'])
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
        labels_val.append(target['brix'])

    #set wandb
    with open(Path("./config_svr.yaml")) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    wandb.login(key="86a812c5e52ce2215d34c29e91dcd07b2862456d")
    wandb.init(config=config)
    config = wandb.config

    #estrai feature
    X_tr, X_val, _ = apply_params_and_extract_features(config, images_tr, images_val)
    y_tr = labels_tr
    y_val = labels_val

    return X_tr, X_val, y_tr, y_val, config

def svr_sweep():
    #set dati necessari
    X_tr, X_val, y_tr, y_val, config = brixDataset()

    #crea il modello
    model = SVR(kernel= config.kernel, C= config.C, epsilon= config.epsilon)

    #fit del modello
    model.fit(X_tr, y_tr)

    #predict del modello
    y_val_hat = model.predict(X_val)

    #parametri da valutare
    mae_val = mean_absolute_error(y_val, y_val_hat)
    mse_val = mean_squared_error(y_val, y_val_hat)
    wandb.log({
        "mae_val": mae_val,
        "mse_val": mse_val,
    })
    #salva il modello
    dump(model, Path(".") / "saved_models_svr.joblib")

svr_sweep()