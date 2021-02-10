import torch.nn as nn
import torch
import numpy as np 

# https://en.wikipedia.org/wiki/Jaccard_index 
# https://gist.github.com/wassname/f1452b748efcbeb4cb9b1d059dce6f96 

# intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
# sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
# jac = (intersection + smooth) / (sum_ - intersection + smooth)
# return (1 - jac) * smooth

class Jaccard(nn.Module):

    def __init__(self, filterROI : bool, smooth = 100  ):
        super().__init__()     
        self.smooth = smooth   
        self.filterROI = filterROI       

    def forward(self, y_true , y_pred):
        '''
        y_true : consists the roi details also with pixel value 85 indicating as outside ROI 
        y_pred : predicted vector of same shape as y_true. 
        '''

        batch_size = y_true.shape[0]
        
        # filter ROI
        if self.filterROI:
            roi_indices = torch.where ( y_true != 0.85 )
            y_true = y_true[roi_indices]
            y_pred = y_pred [ roi_indices ] 

        # jaccard intersection 
        intersection = (y_true * y_pred ).abs().sum() 
        sum_ =  ( y_true.abs() + y_pred.abs() ).sum() 
        jac = ( intersection + self.smooth ) / ( sum_ - intersection + self.smooth )

        # jaccard distance 
        d = ( 1 - jac ) * self.smooth / batch_size 
        return d 


if __name__ == "__main__":
    loss = Jaccard ()
    a = np.random.rand(2,5,5)
    b = np.random.rand(2,5,5)
    a [ a > 0.3 ] = 85 
    print(loss( torch.tensor(a), torch.tensor(b) ) )    