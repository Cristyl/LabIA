import joblib
from brixcolordataset import BrixColorDataset
from pathlib import Path
from torch.utils.data import DataLoader
from PIL import Image
from grapefeatureextractor import GrapeFeatureExtractor
import yaml
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR

BRIXCOLOR_SPLIT = "MixedSeason_"
#modificare in generale il file per farlo più simile a brixextimator del prof
class Brix_SVR():
    def __init__(self):
        self.X_te, self.y_te = self.brixDataset_test()
        self.config = None
    
    def apply_params_and_extract_features(self, config, images_te):
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
        X_te = extractor.extract(images_te, config.n_bin_y, config.n_bin_x, cross=cross, hsv=config.hsv,
                                hog_type=config.hog_type, only_hog=config.only_hog, **kwargs)
        return X_te, cross

    def brixDataset_test(self):
        #cambia il file config con quello della migliore run di wandb, in seguito riaddestra e evaluate
        with open(Path("./config_svr.yaml")) as file:
            self.config = yaml.load(file, Loader=yaml.FullLoader)
        
        dataset_te = BrixColorDataset(Path("./data"), std_split=BRIXCOLOR_SPLIT + "test")

        dataloader_te = DataLoader(dataset_te, batch_size=1, shuffle=False)
        images_te = list()
        labels_te = list()
        for data, target in dataloader_te:
            images_te.append(Image.fromarray(data.numpy().squeeze().transpose(1, 2, 0)))
            labels_te.append(target['brix'])
        
        X_te, _ = self.apply_params_and_extract_features(self.config, images_te)
        y_te = labels_te

        return X_te, y_te

    def fit(self):
        #ricrea training set per poter ritreinare sui migliori iperparametri
        model = SVR(kernel=self.config["kernel"], C=self.config["C"], epsilon=self.config["espilon"])
        model.fit(self.X_tr, self.y_tr)
        joblib.dump(model, Path("./saved_models_svr.joblib"))

    def predict(self):
        model = joblib.load("./saved_models_svr.joblib")
        y_predicted = model.predict(self.X_te)

        return y_predicted
    
    def evaluate(self):
        print("MSE: ",mean_squared_error(self.y_te, self.predict()))



svr = Brix_SVR()
svr.evaluate()