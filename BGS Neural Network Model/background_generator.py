#------------------------------------------------------------------
# code to generate the background of each data folder 
# running this before hand to generate the "background" folder
# or it will run for the first time the Dataset object is created. 
#-------------------------------------------------------------------

import os 
from glob import glob 
from skimage import io 
import numpy as np 
import shutil 

def background_gen ( root_dir : str , n_to_consider_for_bg = 10 , redo = False )-> None :
    '''
    root_dir : path to DATA folder, the main folder containing all datasets
    n_to_consider_for_bg : no of images to consider for basic background generation 

    method : mean over "n_to_consider_for_bg" images is the background 

    generates a background image for the dataset  
    '''
    if not os.path.exists(root_dir):
        raise(Exception(f'root dir : {root_dir} does not exist'))
    elif len(os.listdir(root_dir)) == 0:
        raise(Exception(f'no subdir of dir : {root_dir}') ) 
    
    for sub_dir in os.listdir ( root_dir ):
        path = os.path.join ( root_dir, sub_dir ) 
        background_path = os.path.join ( path, 'background') 

        if redo or not os.path.exists(  background_path ):
            print(f'generating background for {sub_dir}...', end = " ")

            # create the directory 
            # just in case of redo : directory already exists 
            if os.path.exists ( background_path ) :
                shutil.rmtree ( background_path )

            os.mkdir ( background_path )

            # read out n_to_consider_for_bg images and average them 
            gt_images = glob ( path + "/input/*")[:n_to_consider_for_bg]

            # extension of images
            ext = gt_images[0].split(".")[-1]

            images = [] 

            # check for all images read have same shape 
            shape = None 
            for gt_image in gt_images:
                images.append ( io.imread ( gt_image ) )
                if shape is None: shape = images[0].shape 
                else: 
                    assert ( images[-1].shape == shape ), 'all images shape must be consistent '  

            # average them 
            images = np.array ( images, dtype = np.float32 )
            images = np.mean ( images, axis = 0 )
            images = np.array ( images, dtype = np.uint8 ) 

            io.imsave( os.path.join ( background_path, "bg_" + sub_dir + "." + ext ) , images.squeeze() )
            print("done.")

            
if __name__ == "__main__":
    background_gen ( root_dir = "../DATA/", redo= True)

    
