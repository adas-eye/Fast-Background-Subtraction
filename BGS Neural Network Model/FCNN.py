import torch
import torch.nn as nn 
import time 

class Net(nn.Module): 
    
    def __init__(self, in_channels = 6 ):

        super().__init__()
        self.b1 = self.block ( in_channels = in_channels, out_channels = 56 )
        self.b2 = self.block ( in_channels = 56, out_channels = 112 )
        self.b3 = self.block ( in_channels = 112, out_channels = 256 )
        self.b4 = self.block ( in_channels = 256, out_channels = 512 )
    
        self.final_layer = nn.Sequential ( 
            nn.Conv2d(512, 1, 1),
            nn.Sigmoid() 
        ) 


    def block(self, in_channels, out_channels):
        ''' a block consists of 
            convolution operation with 3 X 3 kernel, padding 1 
            relu layer
            batchNormalisation
        ''' 

        temp = nn.Sequential(
            nn.Conv2d ( in_channels = in_channels, out_channels = out_channels, kernel_size = 3, padding= 1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features= out_channels)
        )
        return temp 

    def forward ( self, image ):
        image = self.b1 ( image )
        image = self.b2 ( image )
        image = self.b3 ( image )
        image = self.b4 ( image )
        image = self.final_layer ( image )
        return image.squeeze ( dim = 1 )



if __name__ == "__main__":
    net = Net()
    net.eval() 

    for s in [ 37, 224, 512]:
        image = torch.rand(10, 6, s, s )

        tick = time.time()
        out = net ( image )
        tock = time.time()

        print(f'''
            time taken is : { tock - tick } 
            input size : {image.shape}
            output size : {out.shape}

            ''')
