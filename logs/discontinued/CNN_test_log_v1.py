"""
============================================================================================
----------------------------------------> CNN_1D <------------------------------------------
============================================================================================


TEST 0:
---------------------------------------->
Model: CNN_1D_v0 -> Hyperparamters: 
Learnig rate = 0.0003 
Optimiser = AdamW 
Loss = CrossEntropyLoss 
conv_l1 = 64 
kernel_size = 3 
pool_size = 2 
Batch size = 128 
Epochs = 2 
Model structure:
CNN_1D_v1(
  (batch_norm_1): BatchNorm1d(5, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (conv_1): Conv1d(5, 64, kernel_size=(3,), stride=(1,))
  (relu_1): ReLU()
  (pool_1): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (fc_1): Linear(in_features=93696, out_features=128, bias=True)
  (fc_2): Linear(in_features=128, out_features=64, bias=True)
  (fc_3): Linear(in_features=64, out_features=6, bias=True)
  (softmax): LogSoftmax(dim=1)
)
Data: padded, v3, varying intervals 
Sequence length = 2931 
Batch size = 128 
Shuffled = True

Metric table
=====================================================================================================================
|  Training accuracy  |  Training loss  |  Validation accuracy  |  Validation loss  |  Test accuracy  |  Test loss  |
=====================================================================================================================
|        56.298%      |      1.795      |        55.965%        |       1.796       |      56.606%    |     1.795   |
=====================================================================================================================

Class F1-score table
=====================================================================================================================
|   Drifting longlines   |    Fixed gear    |   Pole and line   |   Purse seines   |    Trawlers    |    Trollers   |
=====================================================================================================================
|         72.310%        |      0.000%      |      0.000%       |     0.000%       |     0.000%     |    0.000%     |
=====================================================================================================================

Predict     0           1           2           3           4           5           
Actual
0           12800       0           0           0           0           0           

1           2371        0           0           0           0           0           

2           293         0           0           0           0           0           

3           2280        0           0           0           0           0           

4           4620        0           0           0           0           0           

5           239         0           0           0           0           0           


Overall Statistics : 

ACC Macro                                                         0.85543
F1 Macro                                                          0.12052
FPR Macro                                                         0.16667
Kappa                                                             0.0
Overall ACC                                                       0.5663
PPV Macro                                                         None
SOA1(Landis & Koch)                                               Slight
TPR Macro                                                         0.16667
Zero-one Loss                                                     9803

Class Statistics :

Classes                                                           0             1             2             3             4             5             
ACC(Accuracy)                                                     0.5663        0.8951        0.98704       0.89913       0.7956        0.98943       
AUC(Area under the ROC curve)                                     0.5           0.5           0.5           0.5           0.5           0.5           
AUCI(AUC value interpretation)                                    Poor          Poor          Poor          Poor          Poor          Poor          
F1(F1 score - harmonic mean of precision and sensitivity)         0.7231        0.0           0.0           0.0           0.0           0.0           
FN(False negative/miss/type 2 error)                              0             2371          293           2280          4620          239           
FP(False positive/type 1 error/false alarm)                       9803          0             0             0             0             0             
FPR(Fall-out or false positive rate)                              1.0           0.0           0.0           0.0           0.0           0.0           
N(Condition negative)                                             9803          20232         22310         20323         17983         22364         
P(Condition positive or support)                                  12800         2371          293           2280          4620          239           
POP(Population)                                                   22603         22603         22603         22603         22603         22603         
PPV(Precision or positive predictive value)                       0.5663        None          None          None          None          None          
TN(True negative/correct rejection)                               0             20232         22310         20323         17983         22364         
TON(Test outcome negative)                                        0             22603         22603         22603         22603         22603         
TOP(Test outcome positive)                                        22603         0             0             0             0             0             
TP(True positive/hit)                                             12800         0             0             0             0             0             
TPR(Sensitivity, recall, hit rate, or true positive rate)         1.0           0.0           0.0           0.0           0.0           0.0   

NOTES: Common sense baseline -> the most frequent class is 'drifting_longlines' and as a result this model is trained only predicting this class (updated)


TEST 1:
---------------------------------------->
Model: CNN_1D_v0 -> Hyperparamters: 
Learnig rate = 0.0003 
Optimiser = AdamW 
Loss = CrossEntropyLoss 
conv_L1 = 16 
kernel_size = 3 
pool_size = 2 
Batch size = 128 
Epochs = 1 
Model structure 
CNN_1D(
  (batch_norm_1): BatchNorm1d(5, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (conv_1): Conv1d(5, 16, kernel_size=(3,), stride=(1,))
  (relu_1): ReLU()
  (pool_1): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (batch_norm_2): BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (conv_2): Conv1d(16, 32, kernel_size=(3,), stride=(1,))
  (relu_2): ReLU()
  (pool_2): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (batch_norm_3): BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (conv_3): Conv1d(32, 64, kernel_size=(3,), stride=(1,))
  (relu_3): ReLU()
  (pool_3): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (fc_1): Linear(in_features=23296, out_features=128, bias=True)
  (fc_2): Linear(in_features=128, out_features=64, bias=True)
  (fc_3): Linear(in_features=64, out_features=6, bias=True)
  (softmax): LogSoftmax(dim=1)
)
Data: padded, v3, varying intervals 
Sequence length = 2931 
Batch size = 128 
Shuffled = True

Metric table
=====================================================================================================================
|  Training accuracy  |  Training loss  |  Validation accuracy  |  Validation loss  |  Test accuracy  |  Test loss  |
=====================================================================================================================
|        56.284%      |      1.817      |        55.840%        |       1.815       |      56.497%    |     1.817   |
=====================================================================================================================

Class accuracy table
=====================================================================================================================
|   Drifting longlines   |    Fixed gear    |   Pole and line   |   Purse seines   |    Trawlers    |    Trollers   |
=====================================================================================================================
|           100%          |        0%       |        0%         |       0%         |        0%      |       0%      |
=====================================================================================================================

TEST 2:
---------------------------------------->
Model: CNN_1D_v9 -> Hyperparamters: 
Learnig rate = 0.0003 
Optimiser = AdamW 
Loss = CrossEntropyLoss 
conv_L1 = 16 
kernel_size = 3 
pool_size = 2 
Batch size = 128 
Epochs = 55 
Model structure 
CNN_1D(
  (batch_norm_1): BatchNorm1d(5, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (conv_1): Conv1d(5, 16, kernel_size=(3,), stride=(1,))
  (relu_1): ReLU()
  (pool_1): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (batch_norm_2): BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (conv_2): Conv1d(16, 32, kernel_size=(3,), stride=(1,))
  (relu_2): ReLU()
  (pool_2): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (batch_norm_3): BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (conv_3): Conv1d(32, 64, kernel_size=(3,), stride=(1,))
  (relu_3): ReLU()
  (pool_3): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (fc_1): Linear(in_features=23296, out_features=128, bias=True)
  (fc_2): Linear(in_features=128, out_features=64, bias=True)
  (fc_3): Linear(in_features=64, out_features=6, bias=True)
  (softmax): LogSoftmax(dim=1)
)
Data: padded, v3, varying intervals 
Sequence length = 2931 
Batch size = 128 
Shuffled = True

Metric table
=====================================================================================================================
|  Training accuracy  |  Training loss  |  Validation accuracy  |  Validation loss  |  Test accuracy  |  Test loss  |
=====================================================================================================================
|        94.044%      |      0.158      |        91.252%        |       0.203       |      91.565%    |     0.223   |
=====================================================================================================================

Class accuracy table
=====================================================================================================================
|   Drifting longlines   |    Fixed gear    |   Pole and line   |   Purse seines   |    Trawlers    |    Trollers   |
=====================================================================================================================
|           97%          |        76%       |        67%        |       82%        |        91%     |       82%     |
=====================================================================================================================





TEST 3:
---------------------------------------->
Model: CNN_1D_v0 -> Hyperparamters: 
Learnig rate = 0.0003 
Optimiser = AdamW 
Loss = CrossEntropyLoss 
conv_l1 = 16 
kernel_size = 3 
pool_size = 2 
Batch size = 128 
Epochs = 25 
Model structure 
CNN_1D_v2(
  (conv_1): Conv1d(5, 16, kernel_size=(3,), stride=(1,))
  (batch_norm_1): BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (relu): ReLU()
  (maxpool): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (res_block_1): ResBlock(
    (conv_1): Conv1d(16, 16, kernel_size=(3,), stride=(1,), padding=(1,))
    (batch_norm_1): BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (conv_2): Conv1d(16, 16, kernel_size=(3,), stride=(1,), padding=(1,))
    (batch_norm_2): BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu): LeakyReLU(negative_slope=0.01)
  )
  (res_block_2): ResBlock(
    (conv_1): Conv1d(16, 16, kernel_size=(3,), stride=(1,), padding=(1,))
    (batch_norm_1): BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (conv_2): Conv1d(16, 16, kernel_size=(3,), stride=(1,), padding=(1,))
    (batch_norm_2): BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu): LeakyReLU(negative_slope=0.01)
  )
  (avgpool): AvgPool1d(kernel_size=(2,), stride=(2,), padding=(0,))
  (conv_2): Conv1d(16, 32, kernel_size=(3,), stride=(1,))
  (batch_norm_2): BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (res_block_3): ResBlock(
    (conv_1): Conv1d(32, 32, kernel_size=(3,), stride=(1,), padding=(1,))
    (batch_norm_1): BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (conv_2): Conv1d(32, 32, kernel_size=(3,), stride=(1,), padding=(1,))
    (batch_norm_2): BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu): LeakyReLU(negative_slope=0.01)
  )
  (res_block_4): ResBlock(
    (conv_1): Conv1d(32, 32, kernel_size=(3,), stride=(1,), padding=(1,))
    (batch_norm_1): BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (conv_2): Conv1d(32, 32, kernel_size=(3,), stride=(1,), padding=(1,))
    (batch_norm_2): BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu): LeakyReLU(negative_slope=0.01)
  )
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (fc_1): Linear(in_features=5824, out_features=128, bias=True)
  (fc_2): Linear(in_features=128, out_features=64, bias=True)
  (fc_3): Linear(in_features=64, out_features=6, bias=True)
  (softmax): LogSoftmax(dim=1)
)
Data: padded, v3, varying intervals 
Sequence length = 2931 
Batch size = 128 
Shuffled = True

Metric table
=====================================================================================================================
|  Training accuracy  |  Training loss  |  Validation accuracy  |  Validation loss  |  Test accuracy  |  Test loss  |
=====================================================================================================================
|        92.184%      |      0.194      |        91.128%        |       0.212       |      91.128%    |     0.213   |
=====================================================================================================================

Class accuracy table
=====================================================================================================================
|   Drifting longlines   |    Fixed gear    |   Pole and line   |   Purse seines   |    Trawlers    |    Trollers   |
=====================================================================================================================
|           97%          |        87%       |        62%        |       81%        |        86%     |       74%     |
=====================================================================================================================





TEST 4:
---------------------------------------->
Model: CNN_1D_v1 -> Hyperparamters: 
Learnig rate = 0.0003 
Optimiser = AdamW 
Loss = CrossEntropyLoss 
conv_l1 = 64 
kernel_size = 3 
pool_size = 2 
Batch size = 128 
Epochs = 25 
Model structure 
CNN_1D_v2(
  (conv_1): Conv1d(5, 64, kernel_size=(3,), stride=(1,))
  (batch_norm_1): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (relu): ReLU()
  (maxpool): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (res_block_1): ResBlock(
    (conv_1): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(1,))
    (batch_norm_1): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (conv_2): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(1,))
    (batch_norm_2): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu): LeakyReLU(negative_slope=0.01)
  )
  (res_block_2): ResBlock(
    (conv_1): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(1,))
    (batch_norm_1): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (conv_2): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(1,))
    (batch_norm_2): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu): LeakyReLU(negative_slope=0.01)
  )
  (avgpool): AvgPool1d(kernel_size=(2,), stride=(2,), padding=(0,))
  (conv_2): Conv1d(64, 128, kernel_size=(3,), stride=(1,))
  (batch_norm_2): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (fc_1): Linear(in_features=46720, out_features=128, bias=True)
  (fc_2): Linear(in_features=128, out_features=64, bias=True)
  (fc_3): Linear(in_features=64, out_features=6, bias=True)
  (softmax): LogSoftmax(dim=1)
)
Data: padded, v3, varying intervals 
Sequence length = 2931 
Batch size = 128 
Shuffled = True

Metric table
=====================================================================================================================
|  Training accuracy  |  Training loss  |  Validation accuracy  |  Validation loss  |  Test accuracy  |  Test loss  |
=====================================================================================================================
|        94.541%      |      0.136      |        92.245%        |       0.196       |      92.377%    |     0.226   |
=====================================================================================================================

Class accuracy table
=====================================================================================================================
|   Drifting longlines   |    Fixed gear    |   Pole and line   |   Purse seines   |    Trawlers    |    Trollers   |
=====================================================================================================================
|           97%          |        84%       |        75%        |       87%        |        89%     |       77%     |
=====================================================================================================================



NOTES: Residual Blocks adding stability and high results


TEST 5:
---------------------------------------->
Model: CNN_1D_v2 -> Hyperparamters: 
Learnig rate = 0.0003 
Optimiser = AdamW 
Loss = CrossEntropyLoss 
conv_l1 = 64 
kernel_size = 3 
pool_size = 2 
Batch size = 128 
Epochs = 25 
Model structure 
CNN_1D_v2(
  (conv_1): Conv1d(5, 64, kernel_size=(3,), stride=(1,))
  (batch_norm_1): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (relu): ReLU()
  (maxpool): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (res_block_1): ResBlock(
    (conv_1): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(1,))
    (batch_norm_1): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (conv_2): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(1,))
    (batch_norm_2): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu): LeakyReLU(negative_slope=0.01)
  )
  (res_block_2): ResBlock(
    (conv_1): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(1,))
    (batch_norm_1): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (conv_2): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(1,))
    (batch_norm_2): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu): LeakyReLU(negative_slope=0.01)
  )
  (avgpool): AvgPool1d(kernel_size=(2,), stride=(2,), padding=(0,))
  (conv_2): Conv1d(64, 128, kernel_size=(3,), stride=(1,))
  (batch_norm_2): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (fc_1): Linear(in_features=46720, out_features=128, bias=True)
  (fc_2): Linear(in_features=128, out_features=64, bias=True)
  (fc_3): Linear(in_features=64, out_features=6, bias=True)
  (softmax): LogSoftmax(dim=1)
)
Data: padded, v3, varying intervals 
Sequence length = 2931 
Batch size = 128 
Shuffled = True

Metric table
=====================================================================================================================
|  Training accuracy  |  Training loss  |  Validation accuracy  |  Validation loss  |  Test accuracy  |  Test loss  |
=====================================================================================================================
|        94.993%      |      0.125      |        92.549%        |       0.175       |      92.629%    |     0.147   |
=====================================================================================================================

Class accuracy table
=====================================================================================================================
|   Drifting longlines   |    Fixed gear    |   Pole and line   |   Purse seines   |    Trawlers    |    Trollers   |
=====================================================================================================================
|           97%          |        82%       |        85%        |       86%        |        92%     |       87%     |
=====================================================================================================================





TEST 6:
---------------------------------------->
Model: CNN_1D_v2 -> Hyperparamters: 
Learnig rate = 0.0003 
Optimiser = AdamW 
Loss = CrossEntropyLoss 
conv_l1 = 64 
kernel_size = 3 
pool_size = 2 
Batch size = 128 
Epochs = 3 (random init) + 25 (training set) + 15 (training and validation combined) , total = 43 epochs
Model structure 
CNN_1D_v2(
  (conv_1): Conv1d(5, 64, kernel_size=(3,), stride=(1,))
  (batch_norm_1): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (relu): ReLU()
  (maxpool): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (res_block_1): ResBlock(
    (conv_1): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(1,))
    (batch_norm_1): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (conv_2): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(1,))
    (batch_norm_2): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu): LeakyReLU(negative_slope=0.01)
  )
  (res_block_2): ResBlock(
    (conv_1): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(1,))
    (batch_norm_1): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (conv_2): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(1,))
    (batch_norm_2): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu): LeakyReLU(negative_slope=0.01)
  )
  (avgpool): AvgPool1d(kernel_size=(2,), stride=(2,), padding=(0,))
  (conv_2): Conv1d(64, 128, kernel_size=(3,), stride=(1,))
  (batch_norm_2): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (fc_1): Linear(in_features=46720, out_features=128, bias=True)
  (fc_2): Linear(in_features=128, out_features=64, bias=True)
  (fc_3): Linear(in_features=64, out_features=6, bias=True)
  (softmax): LogSoftmax(dim=1)
)
Data: padded, v3, varying intervals 
Sequence length = 2931 
Batch size = 128 
Shuffled = True

Metric table
=====================================================================================================================
|  Training accuracy  |  Training loss  |  Validation accuracy  |  Validation loss  |  Test accuracy  |  Test loss  |
=====================================================================================================================
|        96.062%      |      0.096      |        92.549%        |       0.175       |      93.260%    |     0.201   |
=====================================================================================================================

Class accuracy table
=====================================================================================================================
|   Drifting longlines   |    Fixed gear    |   Pole and line   |   Purse seines   |    Trawlers    |    Trollers   |
=====================================================================================================================
|           97%          |        87%       |        75%        |       87%        |        92%     |       81%     |
=====================================================================================================================

NOTES: Strongest results so far with the CNN architecture




"""
