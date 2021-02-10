#---------------------
# Author : Rohit Kumar
# ---------------------
# example directory structure
#  
#  - DATA 
    #   - highway
            # - background
            # - groundtruth 
            # - input 
            # - ROI.bmp ( optional )
            # - ROI.jpg ( optional )
            # - temporalROI.txt 



import os
from glob import glob

import cv2
import torch
from skimage import io
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils

import torchvision.transforms.functional as TF 
import random 

import numpy as np  
from background_generator import background_gen 
from utils import * 

class Data(Dataset):

    def __init__ ( self, root_dir, folders_name, mode : str, outputs_dir = None ):
        '''
        root_dir : path to dataset 
        mode : one of train/ test 
        '''
        self.root_dir = root_dir 
        self.folders_name = folders_name 
        self.raw_images, self.gt_images, self.bg_images = self.list_of_images ( root_dir )
        self.mode = mode 
        self.outputs_dir = outputs_dir 
    
    def list_of_images(self, root_dir ):
        '''
        returns the list of 
        raw_images, gt_images, bg_images as per the file format 
        '''
        if not os.path.exists(root_dir) or len(os.listdir(root_dir)) == 0:
            raise(Exception(f"root_dir passed as {root_dir} either does not exists or has not sub data folder") ) 

        raw_images = [] ; gt_images = [] ; bg_images = {}

        for i, sub_dir in enumerate ( self.folders_name ) :

            path = os.path.join(root_dir, sub_dir)

            #-----------
            # background
            #------------

            # check that background image exist 
            if not os.path.exists ( os.path.join( path, 'background') ):
                background_gen ( root_dir= root_dir, n_to_consider_for_bg= 10, redo = False)  

            # add the bg_image to the bg_images
            bg_images[sub_dir] = sorted ( glob ( os.path.join (path, 'background') + "/*" )  )
            
            #------------------------
            # input  and groundtruth
            #------------------------
            temp_in = sorted ( glob ( os.path.join ( path, "input") + "/*") ) 
            temp_gt = sorted ( glob ( os.path.join ( path, "groundtruth") + "/*") ) 
        
            # read file for given valid range of images 
            temporalFile = os.path.join ( path, 'temporalROI.txt') 
            if not os.path.exists  ( temporalFile ):
                start = 0; end = len(temp_in) - 1 
            else:
                with open(temporalFile) as f:
                    start, end = f.readline().split() 
                    start = int ( start ) - 1 ; end = int(end) - 1  # indexing type 1 of temporalROI.txt 
            
            print(f"{i+1}. {sub_dir} : useful frames - start : {start}, end : { end } ")
            raw_images += temp_in[start:end]
            gt_images += temp_gt[start:end] 

        return sorted ( raw_images ) , sorted ( gt_images ) , bg_images  

    def __len__(self):
        return len(self.raw_images)

    def getBgImage(self, idx ):
        ''' 
        for now assuming the first frame of raw images is the background with no object 
        extension : select a random background image from list of background images 
        ''' 
        key = self.raw_images[idx].split("/")[-3] 
        bg_image_path = np.random.choice ( self.bg_images [ key ] ) 
        return io.imread ( bg_image_path ) 
    
    def transform(self, raw_image, bg_image, gt_image, idx ):
        '''
        a good discussion at https://discuss.pytorch.org/t/torchvision-transfors-how-to-perform-identical-transform-on-both-image-and-target/10606/16 
        a list of transforms in pytorch : https://pytorch.org/docs/stable/torchvision/transforms.html
        ''' 
        # raw_image, bg_image, gt_image = TF.to_pil_image( raw_image ) , TF.to_pil_image (bg_image), TF.to_pil_image ( gt_image )
        def randomCrop(raw_image, bg_image, gt_image, idx ):
            # Random crop 
            # ensure that random crops are not containing outside ROI image only
            # this will break the system as in loss there will be image with value 85 only which is not a mask

            i, j, h, w = transforms.RandomCrop ( size = 224, pad_if_needed= True ).get_params( raw_image, output_size=(224, 224) )
            temp_raw_image = TF.crop(raw_image, i, j, h, w)
            temp_bg_image = TF.crop(bg_image, i, j, h, w) 
            temp_gt_image = TF.crop( gt_image, i, j, h, w) 

            # this loop ensures that random crops will never be outside of ROI 
            
            count = 0 
            while torch.all(temp_gt_image == 0.85):
                count += 1 
                i, j, h, w = transforms.RandomCrop ( size = 224, pad_if_needed= True ).get_params( raw_image, output_size=(224, 224) )
                temp_raw_image = TF.crop(raw_image, i, j, h, w)
                temp_bg_image = TF.crop(bg_image, i, j, h, w) 
                temp_gt_image = TF.crop( gt_image, i, j, h, w) 

                if count >= 20:
                    print(f" gt_image unique values : {torch.unique(gt_image)}" )
                    print(f" filename : { self.gt_images [idx] } ")
                    raise(Warning('ensuring that region of ROI is present in the image is taking too long.\n Maybe better to remove these type of images  or change the algo.')) 
                
                elif count>= 100:
                    raise(Exception("ensuring that region of ROI is present in the image is taking too long")) 

            return temp_raw_image, temp_bg_image, temp_gt_image 


        #------
        # crop
        #------
        raw_image, bg_image, gt_image = randomCrop ( raw_image, bg_image, gt_image, idx ) 

        #---------------------------------
        # train dataset specific transform
        #----------------------------------
        if self.mode == 'train':
            #-----------------
            # horizontalFlip 
            #-----------------
            if random.random() > 0.5 :
                raw_image = TF.hflip(raw_image)
                bg_image = TF.hflip(bg_image)
                gt_image = TF.hflip(gt_image)

            #---------------
            # color jitter
            #---------------
            colorJitter = transforms.ColorJitter( brightness = (0.5, 2), contrast = 0.2, saturation= 0.2 )
            bg_image = colorJitter ( bg_image )  

        # join bg_image and raw_image
        input_hybrid = torch.cat([bg_image, raw_image], 0)

        return input_hybrid, gt_image.squeeze() 
        
    def __getitem__(self, idx ):
        '''
        idx : index of the the image file
        reads the single image and processes it 

        '''
        if torch.is_tensor(idx): idx = idx.to_list() 

        #------------------
        # ground truth mask
        #------------------
        # Tasks : 
        #       : read as grayscale 
        #       : make it binary map
        #       : dtype float32 

                #--------------------
                # for 170 analysis 
                #---------------------
                # gt_image = cv2.imread( self.gt_images [idx] )
                # indices_for_170 = None 
                # if np.any ( gt_image == 170 ) : 
                #     indices_for_170 = np.where ( gt_image == 170 )
                #     temp2 = gt_image.copy() 

        gt_image = cv2.imread( self.gt_images [idx], 0 ) 
        gt_image = np.float32 ( gt_image ) 
        gt_image = gt_image[np.newaxis, :]  

        # this is for CDNET as it has 5 value. read here http://changedetection.net/
        # the value of 85 has to be preserved for ROI clarification later 

                #------------
                # outside ROI 
                #  - mark 85 and 170 as outside ROI 
                #-------------
        
        gt_image [ gt_image == 170 ] = 85  # convert 170 to 85 
        indices_outside_roi = np.where ( gt_image == 85 ) 

        gt_image[ gt_image != 255 ] = 0 
        gt_image = torch.from_numpy ( gt_image ) / 255.0 

        gt_image[ indices_outside_roi ] = 0.85 

        #------------------
        # background image 
        #------------------
        # Tasks :
        #       : illumination processing
        #       : dtype float32
        #       : normalise 
        
        bg_image = self.getBgImage ( idx )  # processed 
        bg_image = np.float32 ( bg_image ) 
        bg_image = bg_image.transpose((2, 0, 1)) / 255.0 
        bg_image = torch.from_numpy(bg_image)
        
        #-----------------
        # composite input
        #-----------------
        raw_image = np.float32 ( io.imread ( self.raw_images[idx] ) ) / 255.0  
        
        #-------------------
        # for 170 analysis
        #--------------------
        # if indices_for_170 is not None:

        #     temp = raw_image.copy() 
        #     temp [ indices_for_170 ] = 0.0 
        #     temp = temp[ np.newaxis, : ] 
        #     temp2 = temp2[np.newaxis, : ]
        #     show( matrices = [ temp, temp2 ],
        #           title = None, 
        #           descriptions = ["raw", "gt"], 
        #           root_out_path= self.outputs_dir, 
        #           subfolderName= "analysis_for_170", 
        #           show_plots= False
        #           )
            

        raw_image = torch.from_numpy ( raw_image.transpose((2, 0, 1)) ) 

        # print(f" raw_image : {raw_image.shape}, bg_image : {bg_image.shape}, gt_image : {gt_image.shape}")
        input_hybrid, gt_image = self.transform ( raw_image, bg_image, gt_image, idx )
        
        return input_hybrid, gt_image
    


if __name__ == "__main__":
    root_dir = "../DATA"

    # folder names 
    test_folders = [ 'pedestrians', 'sofa' ]
    all_folders = os.listdir ( root_dir )
    train_folders = sorted ( list ( set ( all_folders).difference ( set (test_folders) ) ) ) 
    
    print ( f"test folders : no of folders : { len ( test_folders) }, { test_folders}" )
    print ( f"train folders : no of folders : { len ( train_folders) }, { train_folders}" )

    #----------------
    # train dataset
    #----------------

    data = Data( root_dir= root_dir, folders_name= train_folders,  mode = 'train')
    train_dataloader = DataLoader(  data, batch_size = 4 , shuffle = True, num_workers = 4 ) 

    for i, (inputs, outputs ) in enumerate ( train_dataloader ):
        if i == 100:
            break 

        continue 


    # # sample print 
    # print(f"no of images {len(data)} ")
    # indices = np.random.choice ( len(data ), size = 10 )
    # for i in indices:
    #     input_hybrid, gt_image = data[i] 

    #     print(f'''  
    #             raw_image : {data.raw_images[i]}
    #             gt_image : {data.gt_images[i]}
    #             input_hybrid_shape : {input_hybrid.shape}
    #             gt_image : { gt_image.shape }
    #             input_hybrid min, max = {(torch.min(input_hybrid), torch.max(input_hybrid))}
    #             gt_image min, max , unique = { ( torch.min(gt_image), torch.max(gt_image), torch.unique(gt_image))}
    #     ''')

    # #-------------
    # # test dataset
    # #--------------

    #  # train dataset
    # data = Data( root_dir= root_dir, folders_name= test_folders,  mode = 'test')

    # # sample print 
    # print(f"no of images {len(data)} ")
    # indices = np.random.choice ( len(data ), size = 10 )
    # for i in indices:
    #     input_hybrid, gt_image = data[i] 

    #     print(f'''  
    #             raw_image : {data.raw_images[i]}
    #             gt_image : {data.gt_images[i]}
    #             input_hybrid_shape : {input_hybrid.shape}
    #             gt_image : { gt_image.shape }
    #             input_hybrid min, max = {(torch.min(input_hybrid), torch.max(input_hybrid))}
    #             gt_image min, max , unique = { ( torch.min(gt_image), torch.max(gt_image), torch.unique(gt_image))}
    #     ''')
    