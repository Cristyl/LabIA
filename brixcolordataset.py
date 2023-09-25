import datetime
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as T
import csv
import pandas as pd
from pathlib import Path
from datetime import datetime
from PIL import Image
import h5py
import shutil
from zipfile import ZipFile
from zipfile import Path as ZipPath
from utils import flat_unzip


class BrixColorDataset(Dataset):

    def __init__(self, dataset_root, std_split='2021PhoneJPG', grape_type=None, device=None, fmt=None, date=None,
                 transforms=None, compressed=None):
        """
        This class implements the dataset for brix and color estimation. According to the given parameters
        it will load different images from different devices. Additionally, the images are
        divided by year of collection and grape type. The following is a general description,
        for more details on the dataset organization check this document: TODO add url

        The years currently available are '2021' and '2022'. Each year is divided in field trips
        when the data was actually collected.
        The devices that were used to collect the data are: 'phone', 'reflex' and
        'd435i'. Not all devices were used each year.
        There are different file formats in the folders, but the main target is jpg, the various raw formats are
        there for reference. Where it was possible to generate them, the jpg_wb folder contains jpg that were
        computed from raw images with white balance (using the grey card). Not all images had a corresponding raw
        file, or were taken with grey cars. Again, check the reference file for details.

        Given that, the class uses some standard splits to load data:
        ________________________________________________________________________________________________________________
        '2021PhoneJPG': Contains all the jpg images collected with smartphone in 2021, but without white balance.
            From A, C and D splits.
        '2021PhoneJPG_WB': Contains all the jpg images collected with smartphone in 2021, from B, C, D splits
        '2021PhoneJPG_reduced': Contains all the jpg images from 2021 that have a corresponding white balanced image
            both taken with the smartphone. Corresponds to C and D splits.
        '2021PhoneJPG_WB_reduced': Contains all the white balanced images from 2021 corresponding to '2021PhoneJPG_reduced',
            that is C and D splits.
        '2021ReflexJPG_WB': Contains all the jpg images collected with smartphone in 2021 from B, C and D splits
        '2022PhoneJPG_WB': Contains all the jpg images collected with smartphone in 2022 from F, H and I splits
        '2022d435iAutoWB': Contains all the auto white balanced jpg images collected with d435i in 2022, from F, H and
            I splits

        Mixed Splits:
        These splits present a wider range of devices and years, and some of them have predefined train and test
        sets. The ones with predefined train and test sets are the ones that have a suffix '_<split>' in the name,
        that should be substituted with the actual split name.
        ________________________________________________________________________________________________________________
        'Season21ToSeason22Phone_<split>': Contains all the jpg images collected with smartphone in 2021 and 2022,
            from A, C, D, F, H and I splits. A, C and D are used for training, F, H and I for testing. The train set is
            an alias for '2021PhoneJPG', the test set is an alias for '2022PhoneJPG'.
        'Season21ToSeason22Phone_WB_<split>': Contains all the white balanced jpg images collected with smartphone in
            2021 and 2022, from B, C, D, F, H and I splits. B, C and D are used for training, F, H and I for
            testing. The train set is an alias for '2021PhoneJPG_WB', the test set is an alias for '2022PhoneJPG_WB'.
        'Season21ToSeason22All_WB_<split>': Contains all the white balanced jpg images collected with smartphone and
            reflex cameras in 2021 and 2022, from B, C, D, F, H and I splits. B, C and D are used for training,
            F, H and I for testing. The test set is an alias for '2022PhoneJPG_WB', since there are no reflex images.
        'All_Std': Contains all the phone images and all the d435i images from 2021 and 2022, from A, C, D, F,
            H and I. No predefined train and test sets.
            #FIXME there are multiple images taken with different devices of the same grape bunches that can bleed from
                training to test.
        'All_WB': Contains all the white balanced phone and reflex images and all the white balanced d435i images from
            2021 and 2022, from B, C, D, F, H and I. No predefined train and test sets.
            #FIXME there are multiple images taken with different devices of the same grape bunches that can bleed from
                training to test.
        'MixedSeason_<split>': train set contains the phone images from A, C, D, H, and I splits, test set contains
            the phone and d435i images from F split. This is in consideration of the fact that the phone images from F
            split has more brix variability than the H and I splits, and we want to avoid covariate shift.
        'MixedSeason_WB_<split>': train set contains the white balanced phone and reflex images, and d435i images from
            B, C, D, H, and I splits, test set contains the white balanced phone and d435i images from F split. This
            is in consideration of the fact that the phone images from F split has more brix variability than the H and
            I splits, and we want to avoid covariate shift.


        Folder structure:

            dataset_root
            |--Pizzutello Nero
            |  |--2021.mm.dd
            |  |  |--<device1>
            |  |  |   |--[jpg]
            |  |  |   |  |--[full size]
            |  |  |   |  |  |--[annotations]
            |  |  |   |  |  |--<image0001.jpg>
            |  |  |   |  |  |--<image0002.jpg>
            |  |  |   |  |  ...
            !  |  |   |  |--[cropped] # computed from the full size images using the annotations
            |  |  |   |  |  |--[<image0001_0.jpg>]
            |  |  |   |  |  |--[<image0001_1.jpg>]
            |  |  |   |  |  |--[<image0002_0.jpg>]
            |  |  |   |  |  |--[<image0002_1.jpg>]  # if there are more than one bunch in the image
            |  |  |   |  |  ...
            |  |  |   |--[jpg_wb]
            |  |  |   |  |--[full size]
            |  |  |   |     |--[annotations]
            |  |  |   |     |--[illuminant annotations] # if present
            |  |  |   |     |--<image0001.jpg>
            |  |  |   |     ...
            |  |  |   |  |--[cropped]
            |  |  |   |--[raw]
            |  |  |--<device2>
            |  |  |   ...
            |  |  ...
            |  |--2021.mm.dd
            |  ...
            |--Red Globe
               |--2021.mm.dd
               |--2021.mm.dd
               ...

        The "full images" folders, if present, contain the images in the original size. The "cropped" folders contain
        the bunch instance cropped images.
        The "annotations" folders, if present, contain the bounding boxes annotations in yolo format for each grape
        bunch instance contained in the "full images". The "illuminant annotations" folders, if present, contain the
        white balance correction vector computed with the tool in color_correction/illuminant_extractor.py,
        and the bounding box of the area of the grey card used to compute the correction vector.

        :param dataset_root: the base folder for the brix datasets
        :param grape_type: acronym for the grape variety to be used. Currently, only 'PN'
        (pizzutello nero), 'RG' (red globe) and 'Pl' (Plastic) are allowed.
        :param device: string or list of strings. Currently 'phone', 'reflex' and 'd435i' are allowed.
        :param fmt: string or list of strings. Currently 'jpg' and 'jpg_wb' are allowed.
        :param date: string or list of strings with format yyy.mm.dd. Only the combination found in the dataset are
        allowed.
        :param transforms: torch transforms for data pre processing and augmentation. If 'None', the images will be
        resized to (256,128) to make the batch stacking possible. In addition, the only standard other transform that
        is applied is PILtoTensor.
        :param compressed: Can be None, 'zip', 'hdf5'. This dataset comes in different formats. The standard one is the
        uncompressed one, but there is a version where images in leaves folder are compressed in a single file. The
        HDF5 is similar to the compressed one, but the images are scaled to the same size to be stored in a single
        hdf5 dataset. In this version, the raw images for the hdf5 version are not available.
        """
        if std_split:
            if std_split == '2021PhoneJPG' or std_split == 'Season21ToSeason22Phone_train':
                self.grape_type = list(['PN'])
                self.device = list(['phone'])
                self.format = list(['jpg'])
                self.field_trip = ['2021.09.24', '2021.09.30', '2021.10.07']
            elif std_split == '2021PhoneJPG_WB' or std_split == 'Season21ToSeason22Phone_WB_train':
                self.grape_type = list(['PN'])
                self.device = list(['phone'])
                self.format = list(['jpg_wb'])
                self.field_trip = ['2021.09.30', '2021.10.07']
            elif std_split == '2021PhoneJPG_reduced':
                self.grape_type = list(['PN'])
                self.device = list(['phone'])
                self.format = list(['jpg'])
                self.field_trip = ['2021.09.30', '2021.10.07']
            elif std_split == '2021PhoneJPG_WB_reduced':
                self.grape_type = list(['PN'])
                self.device = list(['phone'])
                self.format = list(['jpg_wb_red'])
                self.field_trip = ['2021.09.30', '2021.10.07']
            elif std_split == '2021ReflexJPG_WB':
                self.grape_type = list(['PN'])
                self.device = list(['reflex'])
                self.format = list(['jpg_wb'])
                self.field_trip = ['2021.09.30', '2021.10.07']
            elif std_split == '2022PhoneJPG' or std_split == 'Season21ToSeason22Phone_test':
                self.grape_type = list(['PN'])
                self.device = list(['phone'])
                self.format = list(['jpg'])
                self.field_trip = ['2022.08.08', '2022.08.22', '2022.09.01']
            elif std_split == '2022PhoneJPG_WB' or std_split == 'Season21ToSeason22Phone_WB_test' \
                    or std_split == 'Season21ToSeason22All_WB_test':
                self.grape_type = list(['PN'])
                self.device = list(['phone'])
                self.format = list(['jpg_wb'])
                self.field_trip = ['2022.08.08', '2022.08.22', '2022.09.01']
            elif std_split == '2022d435iAutoWB':
                self.grape_type = list(['PN'])
                self.device = list(['d435i'])
                self.format = list(['jpg'])
                self.field_trip = ['2022.08.08', '2022.08.22', '2022.09.01']
            elif std_split == '2022d435iUPCLab':
                self.grape_type = list(['Pl'])
                self.device = list(['d435i'])
                self.format = list(['jpg'])
                self.field_trip = ['2022.11.29']
            elif std_split == 'Season21ToSeason22All_WB_train':
                self.grape_type = list(['PN'])
                self.device = list(['phone', 'reflex'])
                self.format = list(['jpg_wb'])
                self.field_trip = ['2021.09.30', '2021.10.07']
            elif std_split == 'All_Std':
                self.grape_type = list(['PN'])
                self.device = list(['phone', 'd435i'])
                self.format = list(['jpg'])
                self.field_trip = ['2021.09.24', '2021.09.30', '2021.10.07', '2022.08.08', '2022.08.22', '2022.09.01']
            elif std_split == 'All_Std_WB':
                self.grape_type = list(['PN'])
                self.device = list(['phone', 'reflex', 'd435i'])
                self.format = list(['jpg_wb'])
                self.field_trip = ['2021.09.30', '2021.10.07', '2022.08.08', '2022.08.22', '2022.09.01']
            elif std_split == 'MixedSeason_train':
                self.grape_type = list(['PN'])
                self.device = list(['phone', 'd435i'])
                self.format = list(['jpg'])
                self.field_trip = ['2021.09.24', '2021.09.30', '2021.10.07', '2022.08.22', '2022.09.01']
            elif std_split == 'MixedSeason_test':
                self.grape_type = list(['PN'])
                self.device = list(['phone', 'd435i'])
                self.format = list(['jpg'])
                self.field_trip = ['2022.08.08']
            elif std_split == 'MixedSeason_WB_train':
                self.grape_type = list(['PN'])
                self.device = list(['phone', 'reflex', 'd435i'])
                self.format = list(['jpg_wb'])
                self.field_trip = ['2021.09.30', '2021.10.07', '2022.08.22', '2022.09.01']
            elif std_split == 'MixedSeason_WB_test':
                self.grape_type = list(['PN'])
                self.device = list(['phone', 'd435i'])
                self.format = list(['jpg_wb'])
                self.field_trip = ['2022.08.08']
            else:
                raise ValueError(f'Unrecognized std_split parameter string: {std_split}')
        else:
            self.grape_type = list([grape_type])
            self.device = list([device])
            self.format = list([fmt])
            self.field_trip = list([date])

        self.dataset_root = Path(dataset_root)
        if not transforms:
            self.transforms = T.Compose([T.Resize((256, 128)), T.PILToTensor()])
        else:
            self.transforms = transforms

        self.compressed = compressed

        # Compile dataset folder paths according to parameters
        #   e.g. '<dataset_root>/Pizzutello Nero/2022.09.30/phone/jpg
        self.image_list = list()
        label_list = list()
        header = []
        for gt in self.grape_type:
            if gt == 'PN':
                gt = 'Pizzutello Nero'
            if gt == 'RG':
                gt = 'Red Globe'
            if gt == 'Pl':
                gt = 'Plastic'
            for f_trip in self.field_trip:
                f_trip = datetime.strptime(f_trip, '%Y.%m.%d')
                for device in self.device:
                    for fmt in self.format:
                        # check if the combination device/format exists and eventually list images
                        self.__add_to_image_list(gt, f_trip, device, fmt)
                        # image_folder = dataset_root / gt / f_trip.strftime('%Y.%m.%d') / device / fmt
                        # if image_folder.exists():
                        #     self.image_list = self.image_list + [x for x in image_folder.iterdir() if x.is_file()]
                # load csv file
                label_path = dataset_root / gt / f_trip.strftime('%Y.%m.%d') / \
                    f'labels_{f_trip.strftime("%Y.%m.%d")}.csv'
                current_header, out_list = BrixColorDataset.read_from_csv(label_path)
                # check header consistency with previously loaded sets
                if current_header != ['b1', 'b2', 'b3', 'phone', 'reflex', 'd435i', 'color']:
                    raise ValueError(f"One of the csv header seems to be wrong: current {current_header}, "
                                     f"previous {header}")
                header = current_header
                label_list += out_list

        self.labels = pd.DataFrame.from_records(label_list, columns=list(header))
        self.labels.replace('', np.nan, inplace=True)
        # keep only labels corresponding to images
        self.labels = self.labels[self.labels.isin([x.name for x in self.image_list]).any(axis=1)]
        self.labels = self.labels.astype(dtype={'b1': float, 'b2': float, 'b3': float, 'color': float})
        self.labels['avg_brix'] = self.labels[['b1', 'b2', 'b3']].mean(axis=1)
        # compute some statistics
        self.mean_brix = self.labels['avg_brix'].mean()
        self.std_brix = self.labels['avg_brix'].std()

    def __getitem__(self, item):
        filename = self.image_list[item]
        # Load image
        image = Image.open(filename)
        image = self.transforms(image)
        # get corresponding label
        target_line = self.labels[self.labels.isin([filename.name]).any(axis=1)]
        target = {'brix': target_line['avg_brix'].values[0],
                  'color': -1 if np.isnan(target_line['color'].values[0]) else int(target_line['color'].values[0]-1)}

        return image, target

    def __len__(self):
        return len(self.image_list)

    def __add_to_image_list(self, gt, f_trip, device, fmt):
        image_folder = self.dataset_root / gt / f_trip.strftime('%Y.%m.%d') / device / fmt
        if image_folder.exists():
            if self.compressed is None:
                image_folder /= "cropped"
                self.image_list = self.image_list + [x for x in image_folder.iterdir() if x.is_file() and x.suffix in ['.jpg', '.png', '.jpeg']]
            # FIXME: not sure if to keep this option in future releases. It is working for legacy version where the
            #  folder structure was different
            elif self.compressed == 'zip':
                zip_file = self.dataset_root / gt / f_trip.strftime('%Y.%m.%d') / device / fmt / 'images.zip'
                if zip_file.exists():
                    # check if the images have been already extracted, if so, just add them to the list
                    exist = False
                    for f in image_folder.iterdir():
                        if f.is_file():
                            # check if is a image file
                            if f.suffix in ['.jpg', '.png', '.jpeg']:
                                exist = True
                                break # we suppose that all the files have been extracted
                    if not exist:
                        flat_unzip(image_folder, zip_file)
                    self.image_list = self.image_list + [x for x in image_folder.iterdir() if x.is_file() and x.suffix in ['.jpg', '.png', '.jpeg']]
                else:
                    raise FileNotFoundError(f'Zip file {zip_file} not found.')
            elif self.compressed == 'hdf5':
                raise NotImplementedError("The compressed format 'hdf5' will be implemented in a future release.")


    @staticmethod
    def read_from_csv(filename):
        """
        This method allow for loading the label data without using a pandas dataframe.
        :param filename:
        :return:
        """
        with open(filename) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            header = next(csv_reader)
            rows = list()
            for row in csv_reader:
                rows.append(row)
        return header, rows

def BrixColor_imgs2hdf5(in_path, out_path):
    """
    Scans the BrixColor dataset given and stores it into a number of hdf5 files.
    The function follows the folder structure of the dataset described in the BrixColorDataset class.
    When it reaches a folder containing a file format, it creates a new hdf5 file if the format is readable by
    Pillow. If the format is RAW (DNG or NEF), the folder is compressed in zip archive.
    The images are stored as numpy arrays with uint8 type. All the images are stored in the same dataset.
    The csv files with the brix and color labels are copied as they are. The datasets are named after the
    folder containing them.
    :param in_path: path to the BrixColor dataset
    :param out_path: path where to store the hdf5 files
    :return:
    """
    # check if the input path is valid
    in_path = Path(in_path)
    if not in_path.exists():
        raise ValueError(f'Input path {in_path} does not exist.')
    # check if the output path is valid
    out_path = Path(out_path)
    if not out_path.exists():
        out_path.mkdir(parents=True)
    # check if the output path is empty
    if len(list(out_path.iterdir())) > 0:
        raise ValueError(f'Output path {out_path} is not empty.')

    # scan the dataset
    for gt in in_path.iterdir():
        for f_trip in (in_path / gt).iterdir():
            if f_trip.is_dir():
                for device in f_trip.iterdir():
                    if device.is_dir():
                        for fmt in device.iterdir():
                            if fmt.is_dir():
                                # check if the format is readable by Pillow
                                if fmt.name in ['jpg', 'jpg_wb', 'png']:
                                    # create a new hdf5 file
                                    hdf5_file = h5py.File(out_path / f'{gt.name}_{f_trip.name}_{device.name}'
                                                                     f'_{fmt.name}.hdf5', 'w')
                                    # scan the folder and store the images, while archiving other folders
                                    images = list()
                                    for item in fmt.iterdir():
                                        # stack images in a single dataset
                                        if item.is_file():
                                            # load the image
                                            image = Image.open(item)
                                            # convert to numpy array
                                            images.append(np.array(image))
                                        elif item.is_dir():
                                            # compress folder in zip archive
                                            shutil.make_archive(out_path
                                                                /gt.name/f_trip.name/device.name/fmt.name/item.name,
                                                                'zip', item)
                                    images = np.stack(images, axis=0)
                                    # store the image
                                    hdf5_file.create_dataset('images', data=images)
                    # elif is a csv file, copy it
                    elif device.is_file() and device.suffix == '.csv':
                        shutil.copy(device, out_path / f'{gt.name}_{f_trip.name}_{device.name}')