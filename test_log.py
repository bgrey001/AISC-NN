"""
============================================================================================
----------------------------------------> CNN_1D <------------------------------------------
============================================================================================

TEST 1:
---------------------------------------->
Model: CNN_1D_v1 -> Hyperparamters: 
Learnig rate = 0.0003 
Optimiser = Adam 
Loss = CrossEntropyLoss 
conv_L1 = 16 
kernel_size = 3 
pool_size = 2 
Batch size = 32 
Epochs = 50 
Model structure 
CNN_1D(
  (conv_1): Conv1d(4, 16, kernel_size=(3,), stride=(1,))
  (relu_1): ReLU()
  (pool_1): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (batch_norm_2): BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (dropout_2): Dropout(p=0.1, inplace=False)
  (conv_2): Conv1d(16, 32, kernel_size=(3,), stride=(1,))
  (relu_2): ReLU()
  (pool_2): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (batch_norm_3): BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (dropout_3): Dropout(p=0.1, inplace=False)
  (conv_3): Conv1d(32, 64, kernel_size=(3,), stride=(1,))
  (relu_3): ReLU()
  (pool_3): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (fc_1): Linear(in_features=1280, out_features=128, bias=True)
  (fc_2): Linear(in_features=128, out_features=64, bias=True)
  (fc_3): Linear(in_features=64, out_features=6, bias=True)
  (softmax): LogSoftmax(dim=1)
)
Data: linear_interp, version 3, intervals of 8 minutes 
Sequence length = 180 
Batch size = 32

=====================================================================================================================
|  Training accuracy  |  Training loss  |  Validation accuracy  |  Validation loss  |  Test accuracy  |  Test loss  |
=====================================================================================================================
|        88.500       |      0.2841     |        66.672         |      1.4876       |      73.100     |    1.5737   |
=====================================================================================================================

=====================================================================================================================
|  Training accuracy  |  Training loss  |  Validation accuracy  |  Validation loss  |  Test accuracy  |  Test loss  |
=====================================================================================================================
|  {round(self.history["training_accuracy"][-1], 3)}  |  {round(self.history["training_loss"][-1], 3)}  |  {round(self.history["validation_accuracy"][-1], 3)}  |  {round(self.history["validation_loss"][-1], 3)}  |  {round(self.history["test_accuracy"][0], 3)}  |  {round(self.history["test_loss"][0], 3)}  |
=====================================================================================================================

Training accuracy = 88.50031460931244, average training loss = 0.2841220244765282
Validation accuracy = 66.67239010989012, average validation loss = 1.4876457929611206
Test accuracy = 73.09981684981685, average test loss = 1.5737431049346924

NOTES: Best test accuracy results so far, 




TEST 2:
---------------------------------------->
Model: CNN_1D_v2 -> Hyperparamters: 
Learnig rate = 0.0003 
Optimiser = RMSprop 
Loss = CrossEntropyLoss 
conv_L1 = 16 
kernel_size = 3 
pool_size = 2 
Batch size = 32 
Epochs = 50 
Model structure 
CNN_1D(
  (conv_1): Conv1d(4, 16, kernel_size=(3,), stride=(1,))
  (relu_1): ReLU()
  (pool_1): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv_2): Conv1d(16, 32, kernel_size=(3,), stride=(1,))
  (relu_2): ReLU()
  (pool_2): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv_3): Conv1d(32, 64, kernel_size=(3,), stride=(1,))
  (relu_3): ReLU()
  (pool_3): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (fc_1): Linear(in_features=1280, out_features=128, bias=True)
  (fc_2): Linear(in_features=128, out_features=64, bias=True)
  (fc_3): Linear(in_features=64, out_features=6, bias=True)
  (softmax): LogSoftmax(dim=1)
)
Data: linear_interp, version 3, intervals of 8 minutes 
Sequence length = 180 
Batch size = 32
Training accuracy = 90.93353163253633, average training loss = 0.25391184240579606
Validation accuracy = 69.27369505494505, average validation loss = 1.7757352590560913
Test accuracy = 71.91792582417582, average test loss = 2.0254493772983553
	
	
NOTES: 
No batch norm or dropout, best results yet.





TEST 3:
---------------------------------------->
Model: CNN_1D_v3 -> Hyperparamters: 
Learnig rate = 0.0003 
Optimiser = Adam 
Loss = CrossEntropyLoss 
conv_L1 = 32 
kernel_size = 3 
pool_size = 2 
Batch size = 64 
Epochs = 50 
Model structure 
CNN_1D(
  (conv_1): Conv1d(4, 32, kernel_size=(3,), stride=(1,))
  (relu_1): ReLU()
  (pool_1): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (batch_norm_2): BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (dropout_2): Dropout(p=0.1, inplace=False)
  (conv_2): Conv1d(32, 64, kernel_size=(3,), stride=(1,))
  (relu_2): ReLU()
  (pool_2): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (batch_norm_3): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (dropout_3): Dropout(p=0.1, inplace=False)
  (conv_3): Conv1d(64, 128, kernel_size=(3,), stride=(1,))
  (relu_3): ReLU()
  (pool_3): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (fc_1): Linear(in_features=2560, out_features=128, bias=True)
  (fc_2): Linear(in_features=128, out_features=64, bias=True)
  (fc_3): Linear(in_features=64, out_features=6, bias=True)
  (softmax): LogSoftmax(dim=1)
)
Data: linear_interp, version 3, intervals of 8 minutes 
Sequence length = 180 
Batch size = 64
Training accuracy = 92.20716533180779, average training loss = 0.22422077357769013
Validation accuracy = 65.96554487179486, average validation loss = 2.2816241979599
Test accuracy = 71.57451923076923, average test loss = 1.9592629432678224
	
	
	
TEST 4:
---------------------------------------->
Model: CNN_1D_v4 -> Hyperparamters: 
Learnig rate = 0.0003 
Optimiser = Adam 
Loss = CrossEntropyLoss 
conv_L1 = 32 
kernel_size = 4 
pool_size = 2 
Batch size = 32 
Epochs = 100 
Model structure 
CNN_1D(
  (batch_norm_1): BatchNorm1d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (dropout_1): Dropout(p=0.1, inplace=False)
  (conv_1): Conv1d(4, 32, kernel_size=(4,), stride=(1,))
  (relu_1): ReLU()
  (pool_1): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (batch_norm_2): BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (dropout_2): Dropout(p=0.1, inplace=False)
  (conv_2): Conv1d(32, 64, kernel_size=(4,), stride=(1,))
  (relu_2): ReLU()
  (pool_2): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (batch_norm_3): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (dropout_3): Dropout(p=0.1, inplace=False)
  (conv_3): Conv1d(64, 128, kernel_size=(4,), stride=(1,))
  (relu_3): ReLU()
  (pool_3): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (fc_1): Linear(in_features=2432, out_features=128, bias=True)
  (fc_2): Linear(in_features=128, out_features=64, bias=True)
  (fc_3): Linear(in_features=64, out_features=6, bias=True)
  (softmax): LogSoftmax(dim=1)
)
Data: linear_interp, version 3, intervals of 8 minutes 
Sequence length = 180 
Batch size = 32
Training accuracy = 86.2765987873241, average training loss = 0.2878228560090065
Validation accuracy = 66.42914377289377, average validation loss = 1.3804596662521362
Test accuracy = 63.77632783882784, average test loss = 2.3650902271270753
	
NOTES: Unexpectedly poor results, kernel size at 4 and dropout, batch normalisation in the first layer are the contributing factors	
	
TEST 5:
---------------------------------------->
Done import stats



TEST 6:
---------------------------------------->
Model: CNN_1D_v6 -> Hyperparamters: 
Learnig rate = 0.0003 
Optimiser = Adam 
Loss = CrossEntropyLoss 
conv_L1 = 32 
kernel_size = 3 
pool_size = 2 
Batch size = 64 
Epochs = 50 
Model structure 
CNN_1D(
  (conv_1): Conv1d(4, 32, kernel_size=(3,), stride=(1,))
  (relu_1): ReLU()
  (pool_1): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (batch_norm_2): BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (dropout_2): Dropout(p=0.1, inplace=False)
  (conv_2): Conv1d(32, 64, kernel_size=(3,), stride=(1,))
  (relu_2): ReLU()
  (pool_2): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (batch_norm_3): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (dropout_3): Dropout(p=0.1, inplace=False)
  (conv_3): Conv1d(64, 128, kernel_size=(3,), stride=(1,))
  (relu_3): ReLU()
  (pool_3): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (fc_1): Linear(in_features=2560, out_features=128, bias=True)
  (fc_2): Linear(in_features=128, out_features=64, bias=True)
  (fc_3): Linear(in_features=64, out_features=6, bias=True)
  (softmax): LogSoftmax(dim=1)
)
Data: linear_interp, version 3, intervals of 8 minutes 
Sequence length = 180 
Batch size = 64
Training accuracy = 92.81464530892448, average training loss = 0.21901892721652985
Validation accuracy = 68.34077380952381, average validation loss = 1.8865823149681091
Test accuracy = 74.41334706959707, average test loss = 1.4456888794898988

NOTES: Best results yet on training, validation and test sets -> running for another 50 epochs on a shuffled training and dataset (same distributions just in different order)
NOTES: Trying to recreate results from TEST 3




TEST 8:
---------------------------------------->
Model: CNN_1D_v8 -> Hyperparamters: 
Learnig rate = 0.0003 
Optimiser = Adam 
Loss = CrossEntropyLoss 
conv_L1 = 32 
kernel_size = 3 
pool_size = 2 
Batch size = 64 
Epochs = 150 
Model structure 
CNN_1D(
  (conv_1): Conv1d(4, 32, kernel_size=(3,), stride=(1,))
  (relu_1): ReLU()
  (pool_1): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (batch_norm_2): BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (dropout_2): Dropout(p=0.1, inplace=False)
  (conv_2): Conv1d(32, 64, kernel_size=(3,), stride=(1,))
  (relu_2): ReLU()
  (pool_2): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (batch_norm_3): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (dropout_3): Dropout(p=0.1, inplace=False)
  (conv_3): Conv1d(64, 128, kernel_size=(3,), stride=(1,))
  (relu_3): ReLU()
  (pool_3): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (fc_1): Linear(in_features=2560, out_features=128, bias=True)
  (fc_2): Linear(in_features=128, out_features=64, bias=True)
  (fc_3): Linear(in_features=64, out_features=6, bias=True)
  (softmax): LogSoftmax(dim=1)
)
Data: linear_interp, version 3, intervals of 8 minutes 
Sequence length = 180 
Batch size = 64
=====================================================================================================================
|  Training accuracy  |  Training loss  |  Validation accuracy  |  Validation loss  |  Test accuracy  |  Test loss  |
=====================================================================================================================
|        94.5988       |     0.119      |        68.441         |      2.6355       |      71.300     |    2.8886   |
=====================================================================================================================


















	
============================================================================================
----------------------------------------> GRU-RNN <------------------------------------------
============================================================================================









"""
