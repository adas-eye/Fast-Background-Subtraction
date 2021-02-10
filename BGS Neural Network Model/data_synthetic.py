# file handling 
import os
from glob import glob
import zipfile

# matrix handling
import numpy as np

# image handling 
import cv2
from PIL import Image

# ----------------------
# pytorch modules import
# -----------------------
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import RandomCrop, ColorJitter
import torchvision.transforms.functional as TF
from tqdm import tqdm
from skimage import io
from glob import glob
from utils import *

def read_image(filename):
    # return np.float32 ( io.imread(filename, as_gray=  False) ) 
    img = cv2.imread( filename,cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    return np.float32(img)

class ZipSet:

    def __init__(self, root_path, cache_into_memory=False):
        # directory structure => first level : zipName/  next_level : image1.jpg  image2.jpg ...

        if cache_into_memory:
            f = open(root_path, 'rb')
            self.zip_content = f.read()
            f.close()
            self.zip_file = zipfile.ZipFile(io.BytesIO(self.zip_content), 'r')

        else:
            self.zip_file = zipfile.ZipFile(root_path, 'r')

        self.files = list(self.zip_file.NameToInfo.keys())  # get the keys and convert to list
        self.files.pop(0)  # first file is directory

    def __call__(self, fileName):
        buf = self.zip_file.read(name=os.path.join(fileName))
        img = cv2.imdecode(np.frombuffer(buf, dtype=np.uint8), cv2.IMREAD_COLOR)
        return img

    def __len__(self):
        return len(self.files)

    # def readImageFromZip(self, filename ):
    #     # get a reference to file 
    #     data = self.zip.open( filename )
    #     # read through PIL
    #     img = np.asarray ( Image.open(data) ) 
    #     return img 


class COCO:

    def __init__(self, dataFolder):
        self.folder = dataFolder
        self.files = os.listdir(self.folder)

    def __call__(self, filename):
        # img = cv2.imread(os.path.join(self.folder, filename))
        path = os.path.join( self.folder, filename )
        return read_image(path) 

    def __len__(self):
        return len(self.files)


class CDNETMasks:
    ''' a class that handles the CDNET masks '''

    def __init__(self, path_to_CDNET_folder):
        self.root_dir = path_to_CDNET_folder
        self.masks = self.getMasksList()

    def readTemporalFile(self, path):
        ''' reads the temporal file in each subdirectory'''
        with open(path) as f:
            start, end = f.readline().split()
            start = int(start) - 1;
            end = int(end) - 1  # indexing type 1 of temporalROI.txt
        return start, end

    def getMasksList(self):
        ''' returns a list of masks '''

        # check for directory validation 
        if not os.path.exists(self.root_dir) or len(os.listdir(self.root_dir)) == 0:
            raise (Exception(f"root_dir passed as {self.root_dir} either does not exists or h sub data folder"))

            # get list of validation images
        gt_images = []

        # loop over each subdirectory 
        print("loading data from disk...", end=" ")
        for i, sub_dir in enumerate(tqdm(os.listdir(self.root_dir))):
            path = os.path.join(self.root_dir, sub_dir)
            temp_gt = sorted(glob(os.path.join(path, "groundtruth") + "/*"))

            # read file for given valid range of images 
            temporalFile = os.path.join(path, 'temporalROI.txt')
            if not os.path.exists(path):
                start = 0;
                end = len(temp_gt) - 1
            else:
                start, end = self.readTemporalFile(temporalFile)

            gt_images += temp_gt[start:end]
            # print(f"{i+1}. {sub_dir} : useful frames - start : {start}, end : { end } ")
        print("done.")
        return gt_images

    def readMask(self, filename, channel=1):
        # return np.array ( Image.open(filename) ) 
        # return cv2.imread( filename )
        return read_image(filename) 
        
    def getProcessedMask(self, maskFileName):
        ''' 
        transform the mask image w.r.t CDNET properties
    
        property 1 : mask is a 3 channel in CDNET
        property 2: it has 5 labels : 0, 50, 85, 170, 255 

        '''
        # read 
        maskImage = self.readMask(maskFileName, channel=3)

        # convert all values less than 255 to 0 
        # convert dtype to float 
        maskImage = maskImage // 255.0

        return maskImage

    def getRandomMask(self):
        p = 0.01

        while True:
            filename = np.random.choice(self.masks, replace=False)
            img = self.readMask(filename)

            if (not np.all(img == 0)):
                return filename 

            if (np.random.rand() < p):
                return filename

    def __len__(self):
        return len(self.masks)

    def convert3channelTensorTo1channel(self, image):
        """ image : a channel first input"""
        image = channel_last(image)
        return image[:, :, 0]

        # transpose = lambda x: x.transpose((2, 1, 0))  # H x W x C => C x H x W
        # array = np.uint8(imageTensor.numpy())
        # array = transpose(array)
        # array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
        # array = np.float64(array)
        # return torch.tensor(array)


class SyntheticData(Dataset):

    def __init__(self,
                 path_to_COCO_file,
                 path_to_tuples,
                 path_to_CDNET,
                 mode="train",
                 data_mode="file",  # or zip
                 n_of_tuples=200_000,
                 cache_into_memory=False
                 ):

        self.mode = mode

        # get a zip file reader
        if data_mode == "zip":
            self.COCO = ZipSet(path_to_COCO_file, cache_into_memory=cache_into_memory)
        else:
            self.COCO = COCO(path_to_COCO_file)

        self.files_tuples = self.__getRandomTuples__(path_to_tuples=path_to_tuples,
                                                     n_of_tuples=n_of_tuples,
                                                     replace=True
                                                     )
        # getMasksList
        self.masksDataset = CDNETMasks(path_to_CDNET)

        print(f'''
                there are total masks = {len(self.masksDataset)} images
                and from COCO there are total images = {len(self.COCO.files)}
                we are generating : {n_of_tuples} tuples of two random images from COCO images.
               ''')

    def __getRandomTuples__(self, path_to_tuples, n_of_tuples, replace=True):
        '''generates tuples of format = ( randomImage1, randomImage2 ).
        path_to_tuples : path where to save or load from the tuples with extension .npy
        n_of_tuples : how many tuples of format to be generated. 
        replace : random sampling by replacement method.
        
        returns : a numpy array of shape ( n_of_tuples, 2) where each tuple is the file path of 
                two random images. 
        '''

        if os.path.exists(path_to_tuples):
            x = np.load(path_to_tuples)
            print(f"tuples loaded from file {path_to_tuples}")
        else:
            x = np.random.choice(a=self.COCO.files, size=(n_of_tuples, 2), replace=replace, p=None)
            # remove duplicates
            x = np.unique(x, axis=0)
            x = np.unique(x, axis=1)
            np.save(path_to_tuples, x)
            print(f"tuples saved to file {path_to_tuples}")
        return x

    def __len__(self):
        return self.files_tuples.shape[0]

    def transform(self, backgroundImage, randomImage2, mask, size=224):
        '''
        applies transformation on the input data
        1. make channel first
        2. normalise
        3. convert to tensor
        4. random crop of size
        5. for training, color jitter on background images.
        '''

        # make image channel first and normalise to [0,1]
        # print(f'''
        #         backgroundImage dimension : { backgroundImage.ndim}
        #         randomImage2 dimension : {randomImage2.ndim}
        #         mask dimension : {mask.ndim}
        #         ''')

        backgroundImage = channel_first(backgroundImage) / 255.0
        randomImage2 = channel_first(randomImage2) / 255.0
        if mask.ndim == 3:
            mask = channel_first(mask)

        # make tensor
        backgroundImage, randomImage2, mask = torch.from_numpy(backgroundImage), torch.from_numpy(
            randomImage2), torch.from_numpy(mask)

        # random crop of size
        cropper = RandomCrop(size=size, pad_if_needed=True, fill=0, padding_mode='constant')
        backgroundImage, randomImage2, mask = cropper(backgroundImage), cropper(randomImage2), cropper(mask)

        # training specific
        if self.mode == "train":
            colorJitter = ColorJitter(brightness=(0.5, 2), contrast=0.2, saturation=0.2)
            backgroundImage = colorJitter(backgroundImage)

        return backgroundImage, randomImage2, mask

    def __getitem__(self, idx):

        if torch.is_tensor(idx): idx = idx.to_list()

        # choose 3 images
        randomImage1_filename, randomImage2_filename = self.files_tuples[idx]
        mask_filename = self.masksDataset.getRandomMask()

        # read the three images
        backgroundImage = self.COCO(randomImage1_filename)
        randomImage2 = self.COCO(randomImage2_filename)
        mask = self.masksDataset.getProcessedMask(mask_filename)

        # get the transformed tensors
        backgroundImage, randomImage2, mask = self.transform(backgroundImage, randomImage2, mask, size=224)

        # formulate the target image 
        inputImage = backgroundImage * (1 - mask) + randomImage2 * (mask)

        # convert the mask to a single channel
        # couldn't do it before inputImage as inputImage is not formulated correctly. 
        if mask.ndim == 3:
            mask = torch.tensor ( self.masksDataset.convert3channelTensorTo1channel(mask) ) 

        return inputImage, backgroundImage, mask
