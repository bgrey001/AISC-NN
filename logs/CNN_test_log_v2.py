"""
====================================================================================================================================
----------------------------------------> CNN_1D TEST LOG FOR LINEARLY INTERPOLATED DATA <------------------------------------------
====================================================================================================================================

TEST 1:
---------------------------------------->
Model: CNN_1D_v0 -> Hyperparamters: 
Learnig rate = 0.0003 
Optimiser = AdamW 
Loss = CrossEntropyLoss 
conv_l1 = 64 
kernel_size = 3 
pool_size = 2 
Batch size = 128 
Epochs = 25 
Model structure: 
CNN_1D(
  (conv_1): Conv1d(4, 64, kernel_size=(3,), stride=(1,))
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
  (fc_1): Linear(in_features=4352, out_features=128, bias=True)
  (fc_2): Linear(in_features=128, out_features=64, bias=True)
  (fc_3): Linear(in_features=64, out_features=6, bias=True)
  (softmax): LogSoftmax(dim=1)
) 
Total parameters = 641670
Data: linearly interpolated, v4
Sequence length = 288.0 
Batch size = 128 
Shuffled = True

Metric table
=====================================================================================================================
|  Training accuracy  |  Training loss  |  Validation accuracy  |  Validation loss  |  Test accuracy  |  Test loss  |
=====================================================================================================================
|        88.686%      |      0.277      |        87.477%        |       0.312       |      87.482%    |     0.289   |
=====================================================================================================================

Class F1-score table
=====================================================================================================================
|   Drifting longlines   |    Fixed gear    |   Pole and line   |   Purse seines   |    Trawlers    |    Trollers   |
=====================================================================================================================
|         94.036%        |      78.619%     |      0.000%       |     80.190%      |     87.138%    |    80.362%    |
=====================================================================================================================

NOTES: Model failing to learn on one of the smallest classes, either trollers or pole and line. No nonrandom init for this model, trying that next 



TEST 2:
---------------------------------------->
Model: CNN_1D_v2 -> Hyperparamters: 
Learnig rate = 0.0003 
Optimiser = AdamW 
Loss = CrossEntropyLoss 
Batch size = 128 
Epochs = 28 
Model structure: 
CNN_1D(
  (conv_1): Conv1d(4, 64, kernel_size=(3,), stride=(1,))
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
  (fc_1): Linear(in_features=4352, out_features=128, bias=True)
  (fc_2): Linear(in_features=128, out_features=64, bias=True)
  (fc_3): Linear(in_features=64, out_features=6, bias=True)
  (softmax): LogSoftmax(dim=1)
) 
Total parameters = 641670
Data: linear_interp, v4
Sequence length = 288.0 
Batch size = 128 
Shuffled = True

Metric table
=====================================================================================================================
|  Training accuracy  |  Training loss  |  Validation accuracy  |  Validation loss  |  Test accuracy  |  Test loss  |
=====================================================================================================================
|        91.057%      |      0.221      |        90.838%        |       0.235       |      90.619%    |     0.219   |
=====================================================================================================================

Class F1-score table
=====================================================================================================================
|   Drifting longlines   |    Fixed gear    |   Pole and line   |   Purse seines   |    Trawlers    |    Trollers   |
=====================================================================================================================
|         95.947%        |      80.638%     |      87.415%      |     82.411%      |     89.561%    |    87.340%    |
=====================================================================================================================


Predict     0           1           2           3           4           5           
Actual
0           16595       177         34          166         137         0           

1           129         3867        1           172         580         88          

2           112         4           646         11          0           0           

3           410         336         21          3172        135         3           

4           237         339         3           98          6791        35          

5           0           31          0           2           19          614         


Overall Statistics : 

ACC Macro                                                         0.96873
F1 Macro                                                          0.87219
FPR Macro                                                         0.0218
Kappa                                                             0.86135
Overall ACC                                                       0.90619
PPV Macro                                                         0.8785
SOA1(Landis & Koch)                                               Almost Perfect
TPR Macro                                                         0.86836
Zero-one Loss                                                     3280

Class Statistics :

Classes                                                           0             1             2             3             4             5             
ACC(Accuracy)                                                     0.9599        0.94689       0.99468       0.96128       0.95473       0.99491       
AUC(Area under the ROC curve)                                     0.96011       0.88501       0.91699       0.88174       0.93669       0.95912       
AUCI(AUC value interpretation)                                    Excellent     Very Good     Excellent     Very Good     Excellent     Excellent     
F1(F1 score - harmonic mean of precision and sensitivity)         0.95947       0.80638       0.87415       0.82411       0.89561       0.8734        
FN(False negative/miss/type 2 error)                              514           970           127           905           712           52            
FP(False positive/type 1 error/false alarm)                       888           887           59            449           871           126           
FPR(Fall-out or false positive rate)                              0.04973       0.02944       0.00173       0.01454       0.03172       0.00367       
N(Condition negative)                                             17856         30128         34192         30888         27462         34299         
P(Condition positive or support)                                  17109         4837          773           4077          7503          666           
POP(Population)                                                   34965         34965         34965         34965         34965         34965         
PPV(Precision or positive predictive value)                       0.94921       0.81342       0.91631       0.876         0.88632       0.82973       
TN(True negative/correct rejection)                               16968         29241         34133         30439         26591         34173         
TON(Test outcome negative)                                        17482         30211         34260         31344         27303         34225         
TOP(Test outcome positive)                                        17483         4754          705           3621          7662          740           
TP(True positive/hit)                                             16595         3867          646           3172          6791          614           
TPR(Sensitivity, recall, hit rate, or true positive rate)         0.96996       0.79946       0.83571       0.77802       0.9051        0.92192 


NOTES: Decent model after nonrandom initialisation, using linearly interpolated 5 minute intervals


TEST 3:
---------------------------------------->

Model: CNN_1D_v0 -> Hyperparamters: 
Learnig rate = 0.0003 
Optimiser = AdamW 
Loss = CrossEntropyLoss 
Batch size = 128 
Epochs = 0 
Model structure: 
CNN_1D(
  (conv_1): Conv1d(4, 64, kernel_size=(3,), stride=(1,))
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
  (fc_1): Linear(in_features=22784, out_features=128, bias=True)
  (fc_2): Linear(in_features=128, out_features=64, bias=True)
  (fc_3): Linear(in_features=64, out_features=6, bias=True)
) 
Total parameters = 3000966
Data: linear_interp, v5 
Sequence length = 1440.0 
Batch size = 128 
Shuffled = True


"""
