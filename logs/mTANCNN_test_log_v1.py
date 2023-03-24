"""
=================================================================================================================================
----------------------------------------> mTAN_CNN TEST LOG  <------------------------------------------
=================================================================================================================================

TEST 1:
---------------------------------------->


TEST 2:
---------------------------------------->



TEST 3:
---------------------------------------->
Model: mTANCNN_v12 -> Hyperparamters: 
Learnig rate = 0.0003 
Optimiser = AdamW 
Loss = CrossEntropyLoss 
Batch size = 10 
Epochs = 0 
Model structure: 
mTAN_enc(
  (att): multiTimeAttention(
    (linears): ModuleList(
      (0): Linear(in_features=128, out_features=128, bias=True)
      (1): Linear(in_features=128, out_features=128, bias=True)
      (2): Linear(in_features=4, out_features=64, bias=True)
    )
  )
  (resnet): mTANCNN(
    (conv_1): Conv1d(64, 64, kernel_size=(3,), stride=(1,))
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
  (periodic): Linear(in_features=1, out_features=127, bias=True)
  (linear): Linear(in_features=1, out_features=1, bias=True)
) 
Total parameters = 3046086
Data: non_linear, v1 
Sequence length = 2931 
Batch size = 10 
Shuffled = True

Metric table
=====================================================================================================================
|  Training accuracy  |  Training loss  |  Validation accuracy  |  Validation loss  |  Test accuracy  |  Test loss  |
=====================================================================================================================
|        90.981%      |      0.232      |        91.650%        |       0.229       |      91.664%    |     0.159   |
=====================================================================================================================

Class F1-score table
=====================================================================================================================
|   Drifting longlines   |    Fixed gear    |   Pole and line   |   Purse seines   |    Trawlers    |    Trollers   |
=====================================================================================================================
|         96.228%        |      80.713%     |      82.012%      |     81.089%      |     90.371%    |    90.488%    |
=====================================================================================================================


Predict     0           1           2           3           4           5           
Actual
0           12423       107         13          185         92          0           

1           75          1948        0           137         275         9           

2           62          0           212         15          0           0           

3           287         91          0           1831        61          0           

4           153         220         2           72          4125        3           

5           0           17          1           6           1           177         


Overall Statistics : 

ACC Macro                                                         0.97221
F1 Macro                                                          0.86825
FPR Macro                                                         0.02101
Kappa                                                             0.86364
Overall ACC                                                       0.91664
PPV Macro                                                         0.8934
SOA1(Landis & Koch)                                               Almost Perfect
TPR Macro                                                         0.84736
Zero-one Loss                                                     1884

Class Statistics :

Classes                                                           0             1             2             3             4             5             
ACC(Accuracy)                                                     0.9569        0.95881       0.99588       0.96221       0.96111       0.99836       
AUC(Area under the ROC curve)                                     0.95502       0.88774       0.86642       0.8931        0.93892       0.93785       
AUCI(AUC value interpretation)                                    Excellent     Very Good     Very Good     Very Good     Excellent     Excellent     
F1(F1 score - harmonic mean of precision and sensitivity)         0.96228       0.80713       0.82012       0.81089       0.90371       0.90537       
FN(False negative/miss/type 2 error)                              397           496           77            439           450           25            
FP(False positive/type 1 error/false alarm)                       577           435           16            415           429           12            
FPR(Fall-out or false positive rate)                              0.059         0.02158       0.00072       0.02041       0.0238        0.00054       
N(Condition negative)                                             9780          20156         22311         20330         18025         22398         
P(Condition positive or support)                                  12820         2444          289           2270          4575          202           
POP(Population)                                                   22600         22600         22600         22600         22600         22600         
PPV(Precision or positive predictive value)                       0.95562       0.81746       0.92982       0.81523       0.9058        0.93651       
TN(True negative/correct rejection)                               9203          19721         22295         19915         17596         22386         
TON(Test outcome negative)                                        9600          20217         22372         20354         18046         22411         
TOP(Test outcome positive)                                        13000         2383          228           2246          4554          189           
TP(True positive/hit)                                             12423         1948          212           1831          4125          177           
TPR(Sensitivity, recall, hit rate, or true positive rate)         0.96903       0.79705       0.73356       0.80661       0.90164       0.87624       



NOTES: CHECKPOINT MODEL

"""
