"""
	   
Notes:

How to figure out number of neurons needed in the fully connected layer

Layer 1
Each input is 4 features, 180 sequence length (8 minute intervals)
4 channels in (for each feature)
maxpool is 2

output1 = out_channels1 x seq_length - (kernel_size - 1)
	= 16 x 180 - (2 - 1)
	= 16 x 179 	
divide dim=1 by maxpoolsize and take floor -> floor(179/2) = 89 
	= 16 x 89

layer 2
output2 = out_channels2 x output1(dim=1) - (kernel_size - 1)
	= 32 x 89 - (2 - 1)
	= 32 x 88
divide again by pool size
	= 32 x (floor(88/2))
	= 32 x 44
	
layer 3 
output3 = out_channels3 x output2(dim=1) - (kernel_size - 1)
	= 64 x 43 
divide again
	= 64 x floor(43/2)
	= 64 x 21
    
"""
