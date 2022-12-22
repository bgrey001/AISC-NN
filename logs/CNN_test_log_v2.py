"""


============================================================================================
----------------------------------------> CNN_1D <------------------------------------------
============================================================================================


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
Epochs = 2 
Model structure 
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


TEST 2:
---------------------------------------->
Model: CNN_1D_v0 -> Hyperparamters: 
Learnig rate = 0.0003 
Optimiser = SGD 
Loss = CrossEntropyLoss 
conv_l1 = 8 
kernel_size = 3 
pool_size = 2 
Batch size = 64 
Epochs = 35 
Model structure 
CNN_1D_v1(
  (conv_1): Conv1d(5, 8, kernel_size=(3,), stride=(1,))
  (relu_1): ReLU()
  (pool_1): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (fc_1): Linear(in_features=11712, out_features=6, bias=True)
  (softmax): LogSoftmax(dim=1)
)
Data: padded, v3, varying intervals 
Sequence length = 2931 
Batch size = 64 
Shuffled = True

Metric table
=====================================================================================================================
|  Training accuracy  |  Training loss  |  Validation accuracy  |  Validation loss  |  Test accuracy  |  Test loss  |
=====================================================================================================================
|        68.748%      |      0.866      |        68.512%        |       0.910       |      69.118%    |     0.840   |
=====================================================================================================================

Class F1-score table
=====================================================================================================================
|   Drifting longlines   |    Fixed gear    |   Pole and line   |   Purse seines   |    Trawlers    |    Trollers   |
=====================================================================================================================
|         84.361%        |      31.783%     |      0.000%      |     0.000%      |     60.358%    |    0.000%    |
=====================================================================================================================


Predict     0           1           2           3           4           5           
Actual
0           11916       253         0           1           630         0           

1           420         632         0           0           1319        0           

2           207         42          0           0           44          0           

3           1694        270         0           0           316         0           

4           1180        372         0           0           3068        0           

5           33          37          0           0           169         0           


Overall Statistics : 

ACC Macro                                                         0.89696
F1 Macro                                                          0.29417
FPR Macro                                                         0.09108
Kappa                                                             0.44334
Overall ACC                                                       0.69088
PPV Macro                                                         None
SOA1(Landis & Koch)                                               Moderate
TPR Macro                                                         0.31026
Zero-one Loss                                                     6987

Class Statistics :

Classes                                                           0             1             2             3             4             5             
ACC(Accuracy)                                                     0.80454       0.87997       0.98704       0.89908       0.82171       0.98943       
AUC(Area under the ROC curve)                                     0.78522       0.60921       0.5           0.49998       0.76314       0.5           
AUCI(AUC value interpretation)                                    Good          Fair          Poor          Poor          Good          Poor          
F1(F1 score - harmonic mean of precision and sensitivity)         0.84361       0.31783       0.0           0.0           0.60358       0.0           
FN(False negative/miss/type 2 error)                              884           1739          293           2280          1552          239           
FP(False positive/type 1 error/false alarm)                       3534          974           0             1             2478          0             
FPR(Fall-out or false positive rate)                              0.3605        0.04814       0.0           5e-05         0.1378        0.0           
N(Condition negative)                                             9803          20232         22310         20323         17983         22364         
P(Condition positive or support)                                  12800         2371          293           2280          4620          239           
POP(Population)                                                   22603         22603         22603         22603         22603         22603         
PPV(Precision or positive predictive value)                       0.77126       0.39352       None          0.0           0.55319       None          
TN(True negative/correct rejection)                               6269          19258         22310         20322         15505         22364         
TON(Test outcome negative)                                        7153          20997         22603         22602         17057         22603         
TOP(Test outcome positive)                                        15450         1606          0             1             5546          0             
TP(True positive/hit)                                             11916         632           0             0             3068          0             
TPR(Sensitivity, recall, hit rate, or true positive rate)         0.93094       0.26655       0.0           0.0           0.66407       0.0           

NOTES: Minimal model         



TEST 3:
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
"""

