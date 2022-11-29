"""

DATA
----------------------------------------> <----------------------------------------
Samples = sequence length * features
		 = 24,800,000 (roughly)

Training = 19840000
Validation = 2480000
Testing = 2480000

BEFORE PADDING


============================================================================================
----------------------------------------> CNN_1D <------------------------------------------
============================================================================================

TEST 1:
---------------------------------------->
Model: CNN_1D_v0 -> Hyperparamters: 
Learnig rate = 0.0003 
Optimiser = AdamW 
Loss = CrossEntropyLoss 
conv_L1 = 8 
kernel_size = 3 
pool_size = 2 
Batch size = 128 
Epochs = 10 
Model structure 
CNN_1D(
  (batch_norm_1): BatchNorm1d(5, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (dropout_1): Dropout(p=0.1, inplace=False)
  (conv_1): Conv1d(5, 8, kernel_size=(3,), stride=(1,))
  (relu_1): ReLU()
  (pool_1): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (batch_norm_2): BatchNorm1d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (dropout_2): Dropout(p=0.1, inplace=False)
  (conv_2): Conv1d(8, 16, kernel_size=(3,), stride=(1,))
  (relu_2): ReLU()
  (pool_2): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (batch_norm_3): BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (dropout_3): Dropout(p=0.1, inplace=False)
  (conv_3): Conv1d(16, 32, kernel_size=(3,), stride=(1,))
  (relu_3): ReLU()
  (pool_3): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (fc_1): Linear(in_features=11648, out_features=128, bias=True)
  (fc_2): Linear(in_features=128, out_features=64, bias=True)
  (fc_3): Linear(in_features=64, out_features=6, bias=True)
  (softmax): LogSoftmax(dim=1)
)
Data: padded, v2, varying intervals 
Sequence length = 2931 
Batch size = 128 
Shuffled = True

Metric table
=====================================================================================================================
|  Training accuracy  |  Training loss  |  Validation accuracy  |  Validation loss  |  Test accuracy  |  Test loss  |=====================================================================================================================
|        46.790%      |     1.790       |        45.929%        |      1.790        |      46.475%    |    1.790    |=====================================================================================================================

Class accuracy table
=====================================================================================================================
|   Drifting longlines   |    Fixed gear    |   Pole and line   |   Purse seines   |    Trawlers    |    Trollers   |=====================================================================================================================
|          100%          |       0.000%     |       0.000%      |      0.000%      |      0.000%    |     0.000%    |=====================================================================================================================

NOTES: Common sense baseline -> the most frequent class is 'drifting_longlines' and as a result this model is trained only predicting this class


TEST 2:
---------------------------------------->
Model: CNN_1D_v0 -> Hyperparamters: 
Learnig rate = 0.0003 
Optimiser = AdamW 
Loss = CrossEntropyLoss 
conv_L1 = 4 
kernel_size = 3 
pool_size = 2 
Batch size = 128 
Epochs = 10 
Model structure 
CNN_1D(
  (conv_1): Conv1d(5, 4, kernel_size=(3,), stride=(1,))
  (relu_1): ReLU()
  (pool_1): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (fc_1): Linear(in_features=5856, out_features=128, bias=True)
  (fc_2): Linear(in_features=128, out_features=64, bias=True)
  (fc_3): Linear(in_features=64, out_features=6, bias=True)
  (softmax): LogSoftmax(dim=1)
)
Data: padded, v2, varying intervals 
Sequence length = 2931 
Batch size = 128 
Shuffled = True

Metric table
=====================================================================================================================
|  Training accuracy  |  Training loss  |  Validation accuracy  |  Validation loss  |  Test accuracy  |  Test loss  |=====================================================================================================================
|        87.662%      |      0.250      |        87.295%        |       0.260       |      87.540%    |     0.235   |=====================================================================================================================

Class accuracy table
=====================================================================================================================
|   Drifting longlines   |    Fixed gear    |   Pole and line   |   Purse seines   |    Trawlers    |    Trollers   |=====================================================================================================================
|           99%          |        86%       |        0%         |       0%         |        98%     |       20%     |=====================================================================================================================

NOTES: Simple 1DCNN with only one conv layer and max pool, not great results

TEST :
---------------------------------------->
Model: CNN_1D_v0 -> Hyperparamters: 
Learnig rate = 0.0003 
Optimiser = AdamW 
Loss = CrossEntropyLoss 
conv_L1 = 8 
kernel_size = 3 
pool_size = 2 
Batch size = 64 
Epochs = 10 
Model structure 
CNN_1D(
  (dropout_1): Dropout(p=0.1, inplace=False)
  (conv_1): Conv1d(5, 8, kernel_size=(3,), stride=(1,))
  (relu_1): ReLU()
  (pool_1): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (dropout_2): Dropout(p=0.1, inplace=False)
  (conv_2): Conv1d(8, 16, kernel_size=(3,), stride=(1,))
  (relu_2): ReLU()
  (pool_2): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (dropout_3): Dropout(p=0.1, inplace=False)
  (conv_3): Conv1d(16, 32, kernel_size=(3,), stride=(1,))
  (relu_3): ReLU()
  (pool_3): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (fc_1): Linear(in_features=11648, out_features=128, bias=True)
  (fc_2): Linear(in_features=128, out_features=64, bias=True)
  (fc_3): Linear(in_features=64, out_features=6, bias=True)
  (softmax): LogSoftmax(dim=1)
)
Data: padded, v2, varying intervals 
Sequence length = 2931 
Batch size = 64 
Shuffled = True

Metric table
=====================================================================================================================
|  Training accuracy  |  Training loss  |  Validation accuracy  |  Validation loss  |  Test accuracy  |  Test loss  |=====================================================================================================================
|        88.012%      |      0.248      |        87.240%        |       0.243       |      87.463%    |     0.253   |=====================================================================================================================

Class accuracy table
=====================================================================================================================
|   Drifting longlines   |    Fixed gear    |   Pole and line   |   Purse seines   |    Trawlers    |    Trollers   |=====================================================================================================================
|           100%         |        99%       |        0%         |       0%         |        94%     |       0%      |=====================================================================================================================

NOTES: No batch normalisation layers, poor results


TEST :
---------------------------------------->
Model: CNN_1D_v4 -> Hyperparamters: 
Learnig rate = 0.0003 
Optimiser = AdamW 
Loss = CrossEntropyLoss 
conv_L1 = 8 
kernel_size = 3 
pool_size = 2 
Batch size = 128 
Epochs = 5 (initial) + 3 
Model structure 
CNN_1D(
  (batch_norm_1): BatchNorm1d(5, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (dropout_1): Dropout(p=0.1, inplace=False)
  (conv_1): Conv1d(5, 8, kernel_size=(3,), stride=(1,))
  (relu_1): ReLU()
  (pool_1): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (batch_norm_2): BatchNorm1d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (dropout_2): Dropout(p=0.1, inplace=False)
  (conv_2): Conv1d(8, 16, kernel_size=(3,), stride=(1,))
  (relu_2): ReLU()
  (pool_2): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (batch_norm_3): BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (dropout_3): Dropout(p=0.1, inplace=False)
  (conv_3): Conv1d(16, 32, kernel_size=(3,), stride=(1,))
  (relu_3): ReLU()
  (pool_3): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (fc_1): Linear(in_features=11648, out_features=128, bias=True)
  (fc_2): Linear(in_features=128, out_features=64, bias=True)
  (fc_3): Linear(in_features=64, out_features=6, bias=True)
  (softmax): LogSoftmax(dim=1)
)
Data: padded, v2, varying intervals 
Sequence length = 2931 
Batch size = 128 
Shuffled = True

Metric table
=====================================================================================================================
|  Training accuracy  |  Training loss  |  Validation accuracy  |  Validation loss  |  Test accuracy  |  Test loss  |=====================================================================================================================
|        97.092%      |     0.071       |        97.480%        |      0.068        |      97.605%    |    0.062    |=====================================================================================================================

Class accuracy table
=====================================================================================================================
|   Drifting longlines   |    Fixed gear    |   Pole and line   |   Purse seines   |    Trawlers    |    Trollers   |=====================================================================================================================
|           99%          |        96%       |        91%        |       96%        |        97%     |       79%     |=====================================================================================================================
 
NOTES: Highest validation accuracy and lowest validation loss obtained so far



TEST :
---------------------------------------->
Model: CNN_1D_v5 -> Hyperparamters: 
Learnig rate = 0.0003 
Optimiser = AdamW 
Loss = CrossEntropyLoss 
conv_L1 = 8 
kernel_size = 3 
pool_size = 2 
Batch size = 128 
Epochs = 5 (initial) + 10
Model structure 
CNN_1D(
  (batch_norm_1): BatchNorm1d(5, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (dropout_1): Dropout(p=0.1, inplace=False)
  (conv_1): Conv1d(5, 8, kernel_size=(3,), stride=(1,))
  (relu_1): ReLU()
  (pool_1): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (batch_norm_2): BatchNorm1d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (dropout_2): Dropout(p=0.1, inplace=False)
  (conv_2): Conv1d(8, 16, kernel_size=(3,), stride=(1,))
  (relu_2): ReLU()
  (pool_2): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (batch_norm_3): BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (dropout_3): Dropout(p=0.1, inplace=False)
  (conv_3): Conv1d(16, 32, kernel_size=(3,), stride=(1,))
  (relu_3): ReLU()
  (pool_3): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (fc_1): Linear(in_features=11648, out_features=128, bias=True)
  (fc_2): Linear(in_features=128, out_features=64, bias=True)
  (fc_3): Linear(in_features=64, out_features=6, bias=True)
  (softmax): LogSoftmax(dim=1)
)
Data: padded, v2, varying intervals 
Sequence length = 2931 
Batch size = 128 
Shuffled = True

Metric table
=====================================================================================================================
|  Training accuracy  |  Training loss  |  Validation accuracy  |  Validation loss  |  Test accuracy  |  Test loss  |=====================================================================================================================
|        97.480%      |      0.057      |        97.451%        |       0.065       |      97.488%    |     0.045   |=====================================================================================================================

Class accuracy table
=====================================================================================================================
|   Drifting longlines   |    Fixed gear    |   Pole and line   |   Purse seines   |    Trawlers    |    Trollers   |=====================================================================================================================
|           99%          |        97%       |        95%        |       98%        |        95%     |       86%     |=====================================================================================================================



TEST :
---------------------------------------->
Model: CNN_1D_v6 -> Hyperparamters: 
Learnig rate = 0.0003 
Optimiser = AdamW 
Loss = CrossEntropyLoss 
conv_L1 = 8 
kernel_size = 3 
pool_size = 2 
Batch size = 128 
Epochs = 5 + 10 (with training and validation combined)
Model structure 
CNN_1D(
  (batch_norm_1): BatchNorm1d(5, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (dropout_1): Dropout(p=0.1, inplace=False)
  (conv_1): Conv1d(5, 8, kernel_size=(3,), stride=(1,))
  (relu_1): ReLU()
  (pool_1): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (batch_norm_2): BatchNorm1d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (dropout_2): Dropout(p=0.1, inplace=False)
  (conv_2): Conv1d(8, 16, kernel_size=(3,), stride=(1,))
  (relu_2): ReLU()
  (pool_2): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (batch_norm_3): BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (dropout_3): Dropout(p=0.1, inplace=False)
  (conv_3): Conv1d(16, 32, kernel_size=(3,), stride=(1,))
  (relu_3): ReLU()
  (pool_3): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (fc_1): Linear(in_features=11648, out_features=128, bias=True)
  (fc_2): Linear(in_features=128, out_features=64, bias=True)
  (fc_3): Linear(in_features=64, out_features=6, bias=True)
  (softmax): LogSoftmax(dim=1)
)
Data: padded, v2, varying intervals 
Sequence length = 2931 
Batch size = 128 
Shuffled = True

Metric table
=====================================================================================================================
|  Training accuracy  |  Training loss  |  Validation accuracy  |  Validation loss  |  Test accuracy  |  Test loss  |=====================================================================================================================
|        97.538%      |      0.063      |        96.497%        |       0.091       |      96.200%    |     0.116   |=====================================================================================================================

Class accuracy table
=====================================================================================================================
|   Drifting longlines   |    Fixed gear    |   Pole and line   |   Purse seines   |    Trawlers    |    Trollers   |=====================================================================================================================
|           99%          |        98%       |        95%        |       98%        |        91%     |       83%     |=====================================================================================================================

TEST :
---------------------------------------->
Model: CNN_1D_v7 -> Hyperparamters: 
Learnig rate = 0.0003 
Optimiser = AdamW 
Loss = CrossEntropyLoss 
conv_L1 = 16 
kernel_size = 3 
pool_size = 2 
Batch size = 128 
Epochs = 10 (with training data only) + 10 (restart with training and validation data combined for training)
Model structure 
CNN_1D(
  (batch_norm_1): BatchNorm1d(5, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (dropout_1): Dropout(p=0.1, inplace=False)
  (conv_1): Conv1d(5, 16, kernel_size=(3,), stride=(1,))
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
  (fc_1): Linear(in_features=23296, out_features=128, bias=True)
  (fc_2): Linear(in_features=128, out_features=64, bias=True)
  (fc_3): Linear(in_features=64, out_features=6, bias=True)
  (softmax): LogSoftmax(dim=1)
)
Data: padded, v2, varying intervals 
Sequence length = 2931 
Batch size = 128 
Shuffled = True

Metric table
=====================================================================================================================
|  Training accuracy  |  Training loss  |  Validation accuracy  |  Validation loss  |  Test accuracy  |  Test loss  |=====================================================================================================================
|        98.348%      |      0.036      |        97.770%        |       0.062       |      98.581%    |     0.043   |=====================================================================================================================

Class accuracy table
=====================================================================================================================
|   Drifting longlines   |    Fixed gear    |   Pole and line   |   Purse seines   |    Trawlers    |    Trollers   |=====================================================================================================================
|           100%          |        96%       |        96%        |       99%        |        98%     |       91%     |=====================================================================================================================

NOTES: Unprecendented results, the restart with full data proving to push even further. Another note is that the first conv layer is outputting 16 channels, seems to not only speed up results but mitigate overfitting. 



"""

