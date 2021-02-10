from glob import glob 
import numpy as np 
import os 
from skimage import io 
#--------
# due to the fact that in CDNET
# there is a large part of data only for non-recognised motion with val 170
# we cannot use this for our data modelling. 
# this code verifies the same 

def correctTemporalROI ( folder ):
    
    temporalROIFile = os.path.join ( folder, "temporalROI.txt") 
    ground_truths = glob ( os.path.join ( folder, "groundtruth" ) + "/*" ) 

    start = None
    end = None 

    for index, f in enumerate ( ground_truths ) :
        
        img = io.imread ( f ) 
        unique_values = np.unique(img)

        if start is None and not np.all ( unique_values == 85 ):
            start = index + 1 
        
        elif end is None and np.all ( unique_values == 170 ):
            end = index + 1 
            break 

    if end is None:
        end = len ( ground_truths ) 

    with open ( temporalROIFile, "r") as f:
        data = f.readline() 

    dataset = folder.split()[-1]
    print(f'''
            dataset : {dataset}
            original temporal ROI : {data}
            found ROI : { start, end }
            ''')


if __name__ == "__main__":
    root = "../DATA/"
    for folder in os.listdir ( root ):
        if folder not in ["boats", "canoe", "fall", "fountain01", "fountain02", "overpass" ]:
            continue
        correctTemporalROI ( os.path.join ( root, folder ) ) 