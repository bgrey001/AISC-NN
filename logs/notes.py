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
	
	
	
	
	
	
	
	
	
MODEL PARAMS mTANCNN

att.linears.0.weight
torch.Size([128, 128])

att.linears.0.bias
torch.Size([128])

att.linears.1.weight
torch.Size([128, 128])

att.linears.1.bias
torch.Size([128])

att.linears.2.weight
torch.Size([64, 4])

att.linears.2.bias
torch.Size([64])

resnet.conv_1.weight
torch.Size([64, 64, 3])

resnet.conv_1.bias
torch.Size([64])

resnet.batch_norm_1.weight
torch.Size([64])

resnet.batch_norm_1.bias
torch.Size([64])

resnet.res_block_1.conv_1.weight
torch.Size([64, 64, 3])

resnet.res_block_1.conv_1.bias
torch.Size([64])

resnet.res_block_1.batch_norm_1.weight
torch.Size([64])

resnet.res_block_1.batch_norm_1.bias
torch.Size([64])

resnet.res_block_1.conv_2.weight
torch.Size([64, 64, 3])

resnet.res_block_1.conv_2.bias
torch.Size([64])

resnet.res_block_1.batch_norm_2.weight
torch.Size([64])

resnet.res_block_1.batch_norm_2.bias
torch.Size([64])

resnet.res_block_2.conv_1.weight
torch.Size([64, 64, 3])

resnet.res_block_2.conv_1.bias
torch.Size([64])

resnet.res_block_2.batch_norm_1.weight
torch.Size([64])

resnet.res_block_2.batch_norm_1.bias
torch.Size([64])

resnet.res_block_2.conv_2.weight
torch.Size([64, 64, 3])

resnet.res_block_2.conv_2.bias
torch.Size([64])

resnet.res_block_2.batch_norm_2.weight
torch.Size([64])

resnet.res_block_2.batch_norm_2.bias
torch.Size([64])

resnet.conv_2.weight
torch.Size([128, 64, 3])

resnet.conv_2.bias
torch.Size([128])

resnet.batch_norm_2.weight
torch.Size([128])

resnet.batch_norm_2.bias
torch.Size([128])

resnet.fc_1.weight
torch.Size([128, 22784])

resnet.fc_1.bias
torch.Size([128])

resnet.fc_2.weight
torch.Size([64, 128])

resnet.fc_2.bias
torch.Size([64])

resnet.fc_3.weight
torch.Size([6, 64])

resnet.fc_3.bias
torch.Size([6])

periodic.weight
torch.Size([127, 1])

periodic.bias
torch.Size([127])

linear.weight
torch.Size([1, 1])

linear.bias
torch.Size([1])
	
	
	
	
    
"""
