"""

=============================================================================================================================
----------------------------------------> CNN_1D TEST LOG FOR VARYING PADDED DATA <------------------------------------------
=============================================================================================================================

TEST 1:
---------------------------------------->
Model: CNN_1D_v0 -> Hyperparamters: 
Learnig rate = 3e-05 
Optimiser = SGD 
Loss = CrossEntropyLoss 
conv_l1 = 6 
kernel_size = 3 
pool_size = 2 
Batch size = 64 
Epochs = 5 
Model structure: 
CNN_1D_v1(
  (conv_1): Conv1d(5, 6, kernel_size=(3,), stride=(1,))
  (relu_1): ReLU()
  (pool_1): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (fc_1): Linear(in_features=8784, out_features=6, bias=True)
  (softmax): LogSoftmax(dim=1)
) 
Total parameters = 52806
Data: padded, v3, varying intervals 
Sequence length = 2931 
Batch size = 64 
Shuffled = True

Metric table
=====================================================================================================================
|  Training accuracy  |  Training loss  |  Validation accuracy  |  Validation loss  |  Test accuracy  |  Test loss  |
=====================================================================================================================
|        56.886%      |      1.149      |        57.399%        |       1.131       |      57.880%    |     1.109   |
=====================================================================================================================

Class F1-score table
=====================================================================================================================
|   Drifting longlines   |    Fixed gear    |   Pole and line   |   Purse seines   |    Trawlers    |    Trollers   |
=====================================================================================================================
|         73.716%        |      0.000%      |      0.000%       |     0.000%       |     13.392%    |    0.000%     |
=====================================================================================================================


Predict     0           1           2           3           4           5           
Actual
0           12726       0           0           0           74          0           

1           1982        0           0           0           389         0           

2           293         0           0           0           0           0           

3           2260        0           0           0           20          0           

4           4252        0           0           0           368         0           

5           214         0           0           0           25          0           


Overall Statistics : 

ACC Macro                                                         0.85977
F1 Macro                                                          0.14518
FPR Macro                                                         0.15774
Kappa                                                             0.06038
Overall ACC                                                       0.5793
PPV Macro                                                         None
SOA1(Landis & Koch)                                               Slight
TPR Macro                                                         0.17898
Zero-one Loss                                                     9509

Class Statistics :

Classes                                                           0             1             2             3             4             5             
ACC(Accuracy)                                                     0.5985        0.8951        0.98704       0.89913       0.78941       0.98943       
AUC(Area under the ROC curve)                                     0.53802       0.5           0.5           0.5           0.5257        0.5           
AUCI(AUC value interpretation)                                    Poor          Poor          Poor          Poor          Poor          Poor          
F1(F1 score - harmonic mean of precision and sensitivity)         0.73716       0.0           0.0           0.0           0.13392       0.0           
FN(False negative/miss/type 2 error)                              74            2371          293           2280          4252          239           
FP(False positive/type 1 error/false alarm)                       9001          0             0             0             508           0             
FPR(Fall-out or false positive rate)                              0.91819       0.0           0.0           0.0           0.02825       0.0           
N(Condition negative)                                             9803          20232         22310         20323         17983         22364         
P(Condition positive or support)                                  12800         2371          293           2280          4620          239           
POP(Population)                                                   22603         22603         22603         22603         22603         22603         
PPV(Precision or positive predictive value)                       0.58572       None          None          None          0.42009       None          
TN(True negative/correct rejection)                               802           20232         22310         20323         17475         22364         
TON(Test outcome negative)                                        876           22603         22603         22603         21727         22603         
TOP(Test outcome positive)                                        21727         0             0             0             876           0             
TP(True positive/hit)                                             12726         0             0             0             368           0             
TPR(Sensitivity, recall, hit rate, or true positive rate)         0.99422       0.0           0.0           0.0           0.07965       0.0          

NOTES: Minimal model         



TEST 2:
---------------------------------------->
Model: CNN_1D_v6 -> Hyperparamters: 
Learnig rate = 0.0003 
Optimiser = AdamW
Loss = CrossEntropyLoss 
conv_l1 = 64 
kernel_size = 3 
pool_size = 2 
Batch size = 128 
Epochs = 30 
Model structure: 
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
Total parameters = 6064966
Data: padded, v3, varying intervals 
Sequence length = 2931 
Batch size = 128 
Shuffled = True
Post training L1-norm global unstructured trainable weight pruning: 0.51

Metric table
=====================================================================================================================
|  Training accuracy  |  Training loss  |  Validation accuracy  |  Validation loss  |  Test accuracy  |  Test loss  |
=====================================================================================================================
|        95.439%      |      0.111      |        93.029%        |       0.175       |      93.265%    |     0.227   |
=====================================================================================================================

Class F1-score table
=====================================================================================================================
|   Drifting longlines   |    Fixed gear    |   Pole and line   |   Purse seines   |    Trawlers    |    Trollers   |
=====================================================================================================================
|         97.029%        |      83.821%     |      78.644%      |     86.747%      |     92.160%    |    86.681%    |
=====================================================================================================================


Predict     0           1           2           3           4           5           
Actual
0           12426       123         49          93          109         0           

1           54          2036        1           64          193         23          

2           54          0           232         6           1           0           

3           176         110         14          1908        68          4           

4           103         200         1           41          4273        2           

5           0           18          0           7           9           205         


Overall Statistics : 

ACC Macro                                                         0.97754
F1 Macro                                                          0.87514
FPR Macro                                                         0.01625
Kappa                                                             0.89055
Overall ACC                                                       0.93262
PPV Macro                                                         0.8774
SOA1(Landis & Koch)                                               Almost Perfect
TPR Macro                                                         0.87346
Zero-one Loss                                                     1523

Class Statistics :

Classes                                                           0             1             2             3             4             5             
ACC(Accuracy)                                                     0.96633       0.96523       0.99443       0.97421       0.96784       0.99721       
AUC(Area under the ROC curve)                                     0.96565       0.91821       0.89445       0.91323       0.95188       0.92822       
AUCI(AUC value interpretation)                                    Excellent     Excellent     Very Good     Excellent     Excellent     Excellent     
F1(F1 score - harmonic mean of precision and sensitivity)         0.97029       0.83821       0.78644       0.86747       0.9216        0.86681       
FN(False negative/miss/type 2 error)                              374           335           61            372           347           34            
FP(False positive/type 1 error/false alarm)                       387           451           65            211           380           29            
FPR(Fall-out or false positive rate)                              0.03948       0.02229       0.00291       0.01038       0.02113       0.0013        
N(Condition negative)                                             9803          20232         22310         20323         17983         22364         
P(Condition positive or support)                                  12800         2371          293           2280          4620          239           
POP(Population)                                                   22603         22603         22603         22603         22603         22603         
PPV(Precision or positive predictive value)                       0.9698        0.81866       0.78114       0.90042       0.91833       0.87607       
TN(True negative/correct rejection)                               9416          19781         22245         20112         17603         22335         
TON(Test outcome negative)                                        9790          20116         22306         20484         17950         22369         
TOP(Test outcome positive)                                        12813         2487          297           2119          4653          234           
TP(True positive/hit)                                             12426         2036          232           1908          4273          205           
TPR(Sensitivity, recall, hit rate, or true positive rate)         0.97078       0.85871       0.79181       0.83684       0.92489       0.85774  



TEST 2:
---------------------------------------->
Model: CNN_1D_v4 -> Hyperparamters: 
Learnig rate = 0.0003 
Optimiser = AdamW 
Loss = CrossEntropyLoss 
conv_l1 = 64 
kernel_size = 3 
pool_size = 2 
Batch size = 128 
Epochs = 5 
Model structure: 
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
Total parameters = 6064966
Data: padded, v3, varying intervals 
Sequence length = 2931 
Batch size = 128 
Shuffled = True

Metric table
=====================================================================================================================
|  Training accuracy  |  Training loss  |  Validation accuracy  |  Validation loss  |  Test accuracy  |  Test loss  |
=====================================================================================================================
|        95.439%      |      0.111      |        93.029%        |       0.175       |      93.305%    |     0.194   |
=====================================================================================================================

Class F1-score table
=====================================================================================================================
|   Drifting longlines   |    Fixed gear    |   Pole and line   |   Purse seines   |    Trawlers    |    Trollers   |
=====================================================================================================================
|         97.060%        |      83.855%     |      78.859%      |     86.791%      |     92.220%    |    86.864%    |
=====================================================================================================================


Predict     0           1           2           3           4           5           
Actual
0           12428       121         52          93          106         0           

1           54          2036        1           67          191         22          

2           51          0           235         6           1           0           

3           173         110         14          1912        67          4           

4           103         200         1           41          4273        2           

5           0           18          0           7           9           205         


Overall Statistics : 

ACC Macro                                                         0.97767
F1 Macro                                                          0.87608
FPR Macro                                                         0.01611
Kappa                                                             0.89122
Overall ACC                                                       0.93302
PPV Macro                                                         0.87731
SOA1(Landis & Koch)                                               Almost Perfect
TPR Macro                                                         0.87549
Zero-one Loss                                                     1514

Class Statistics :

Classes                                                           0             1             2             3             4             5             
ACC(Accuracy)                                                     0.96669       0.96531       0.99443       0.97425       0.9681        0.99726       
AUC(Area under the ROC curve)                                     0.96604       0.91826       0.8995        0.91403       0.95205       0.92824       
AUCI(AUC value interpretation)                                    Excellent     Excellent     Very Good     Excellent     Excellent     Excellent     
F1(F1 score - harmonic mean of precision and sensitivity)         0.9706        0.83855       0.78859       0.86791       0.9222        0.86864       
FN(False negative/miss/type 2 error)                              372           335           58            368           347           34            
FP(False positive/type 1 error/false alarm)                       381           449           68            214           374           28            
FPR(Fall-out or false positive rate)                              0.03887       0.02219       0.00305       0.01053       0.0208        0.00125       
N(Condition negative)                                             9803          20232         22310         20323         17983         22364         
P(Condition positive or support)                                  12800         2371          293           2280          4620          239           
POP(Population)                                                   22603         22603         22603         22603         22603         22603         
PPV(Precision or positive predictive value)                       0.97026       0.81932       0.77558       0.89934       0.91952       0.87983       
TN(True negative/correct rejection)                               9422          19783         22242         20109         17609         22336         
TON(Test outcome negative)                                        9794          20118         22300         20477         17956         22370         
TOP(Test outcome positive)                                        12809         2485          303           2126          4647          233           
TP(True positive/hit)                                             12428         2036          235           1912          4273          205           
TPR(Sensitivity, recall, hit rate, or true positive rate)         0.97094       0.85871       0.80205       0.8386        0.92489       0.85774    

NOTES: Added pruning, marginal increase in test accuracy





"""

