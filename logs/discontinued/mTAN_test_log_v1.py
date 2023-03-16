"""
=================================================================================================================================
----------------------------------------> mTANGRU TEST LOG  <------------------------------------------
=================================================================================================================================

TEST 1:
---------------------------------------->
mTAN_v1 state dictionary successfully loaded
mTAN_v1 history loaded

Model: mTAN_v1 -> Hyperparamters: 
Learnig rate = 0.0003 
Optimiser = AdamW 
Loss = CrossEntropyLoss 
Batch size = 30 
Epochs = 17 
Model structure: 
mTAN_enc(
  (att): multiTimeAttention(
    (linears): ModuleList(
      (0): Linear(in_features=16, out_features=16, bias=True)
      (1): Linear(in_features=16, out_features=16, bias=True)
      (2): Linear(in_features=4, out_features=64, bias=True)
    )
  )
  (GRU): GRU(64, 64, num_layers=2, batch_first=True, bidirectional=True)
  (periodic): Linear(in_features=1, out_features=15, bias=True)
  (linear): Linear(in_features=1, out_features=1, bias=True)
  (classifier): Sequential(
    (0): Linear(in_features=128, out_features=128, bias=True)
    (1): Linear(in_features=128, out_features=64, bias=True)
    (2): Linear(in_features=64, out_features=6, bias=True)
  )
) 
Total parameters = 150470
Data: non_linear, v1 
Sequence length = 2931 
Batch size = 30 
Shuffled = True

Metric table
=====================================================================================================================
|  Training accuracy  |  Training loss  |  Validation accuracy  |  Validation loss  |  Test accuracy  |  Test loss  |
=====================================================================================================================
|        89.008%      |      0.274      |        88.774%        |       0.323       |      88.920%    |     0.266   |
=====================================================================================================================

Class F1-score table
=====================================================================================================================
|   Drifting longlines   |    Fixed gear    |   Pole and line   |   Purse seines   |    Trawlers    |    Trollers   |
=====================================================================================================================
|         93.845%        |      76.975%     |      77.778%      |     77.252%      |     86.921%    |    89.209%    |
=====================================================================================================================

NOTES: First run looks promising, massive epoch time of 1hr 8 minutes




TEST 2:
---------------------------------------->
Model: mTAN_v1 -> Hyperparamters: 
Learnig rate = 0.0003 
Optimiser = AdamW 
Loss = CrossEntropyLoss 
Batch size = 18 
Epochs = 15 
Model structure: 
mTAN_enc(
  (att): multiTimeAttention(
    (linears): ModuleList(
      (0): Linear(in_features=16, out_features=16, bias=True)
      (1): Linear(in_features=16, out_features=16, bias=True)
      (2): Linear(in_features=4, out_features=64, bias=True)
    )
  )
  (GRU): GRU(64, 64, num_layers=2, batch_first=True, bidirectional=True)
  (periodic): Linear(in_features=1, out_features=15, bias=True)
  (linear): Linear(in_features=1, out_features=1, bias=True)
  (classifier): Sequential(
    (0): Linear(in_features=128, out_features=128, bias=True)
    (1): Linear(in_features=128, out_features=64, bias=True)
    (2): Linear(in_features=64, out_features=6, bias=True)
  )
) 
Total parameters = 150470
Data: non_linear, v1 
Sequence length = 2931 
Batch size = 18 
Shuffled = True

Metric table
=====================================================================================================================
|  Training accuracy  |  Training loss  |  Validation accuracy  |  Validation loss  |  Test accuracy  |  Test loss  |
=====================================================================================================================
|        88.543%      |      0.315      |        88.203%        |       0.358       |      88.380%    |     0.324   |
=====================================================================================================================

Class F1-score table
=====================================================================================================================
|   Drifting longlines   |    Fixed gear    |   Pole and line   |   Purse seines   |    Trawlers    |    Trollers   |
=====================================================================================================================
|         93.942%        |      76.528%     |      81.111%      |     75.468%      |     84.966%    |    90.625%    |
=====================================================================================================================


Predict     0           1           2           3           4           5           
Actual
0           12397       124         26          160         108         0           

1           204         1895        1           137         200         8           

2           49          0           219         21          0           0           

3           460         119         1           1631        58          0           

4           465         354         4           99          3648        0           

5           3           13          0           7           5           174         


Overall Statistics : 

ACC Macro                                                         0.96125
F1 Macro                                                          0.83771
FPR Macro                                                         0.03239
Kappa                                                             0.8068
Overall ACC                                                       0.88375
PPV Macro                                                         0.86657
SOA1(Landis & Koch)                                               Almost Perfect
TPR Macro                                                         0.81311
Zero-one Loss                                                     2626

Class Statistics :

Classes                                                           0             1             2             3             4             5             
ACC(Accuracy)                                                     0.92922       0.94865       0.99548       0.95299       0.94276       0.99841       
AUC(Area under the ROC curve)                                     0.92328       0.87239       0.87818       0.84898       0.88883       0.93051       
AUCI(AUC value interpretation)                                    Excellent     Very Good     Very Good     Very Good     Very Good     Excellent     
F1(F1 score - harmonic mean of precision and sensitivity)         0.93942       0.76566       0.81111       0.75439       0.84946       0.90625       
FN(False negative/miss/type 2 error)                              418           550           70            638           922           28            
FP(False positive/type 1 error/false alarm)                       1181          610           32            424           371           8             
FPR(Fall-out or false positive rate)                              0.12082       0.03028       0.00143       0.02087       0.02059       0.00036       
N(Condition negative)                                             9775          20145         22301         20321         18020         22388         
P(Condition positive or support)                                  12815         2445          289           2269          4570          202           
POP(Population)                                                   22590         22590         22590         22590         22590         22590         
PPV(Precision or positive predictive value)                       0.91302       0.75649       0.87251       0.79367       0.90769       0.95604       
TN(True negative/correct rejection)                               8594          19535         22269         19897         17649         22380         
TON(Test outcome negative)                                        9012          20085         22339         20535         18571         22408         
TOP(Test outcome positive)                                        13578         2505          251           2055          4019          182           
TP(True positive/hit)                                             12397         1895          219           1631          3648          174           
TPR(Sensitivity, recall, hit rate, or true positive rate)         0.96738       0.77505       0.75779       0.71882       0.79825       0.86139      

NOTES: No nonrandom init, unstable training, training loss seems to be hitting a wall, unexpected. Batch size reduction made quicker epoch time (48 mins), next to try is batch size of 1 and nonrandom init





TEST 2:
---------------------------------------->
Model: mTAN_v3 -> Hyperparamters: 
Learnig rate = 0.0003 
Optimiser = AdamW 
Loss = CrossEntropyLoss 
Batch size = 10 
Epochs = 32 
Model structure: 
mTAN_enc(
  (att): multiTimeAttention(
    (linears): ModuleList(
      (0): Linear(in_features=16, out_features=16, bias=True)
      (1): Linear(in_features=16, out_features=16, bias=True)
      (2): Linear(in_features=4, out_features=64, bias=True)
    )
  )
  (GRU): GRU(64, 64, num_layers=2, batch_first=True, bidirectional=True)
  (periodic): Linear(in_features=1, out_features=15, bias=True)
  (linear): Linear(in_features=1, out_features=1, bias=True)
  (classifier): Sequential(
    (0): Linear(in_features=128, out_features=128, bias=True)
    (1): Linear(in_features=128, out_features=64, bias=True)
    (2): Linear(in_features=64, out_features=6, bias=True)
  )
) 
Total parameters = 150470
Data: non_linear, v1 
Sequence length = 2931 
Batch size = 10 
Shuffled = True

Metric table
=====================================================================================================================
|  Training accuracy  |  Training loss  |  Validation accuracy  |  Validation loss  |  Test accuracy  |  Test loss  |
=====================================================================================================================
|        95.087%      |      0.119      |        92.717%        |       0.164       |      92.832%    |     0.266   |
=====================================================================================================================

Class F1-score table
=====================================================================================================================
|   Drifting longlines   |    Fixed gear    |   Pole and line   |   Purse seines   |    Trawlers    |    Trollers   |
=====================================================================================================================
|         96.753%        |      83.235%     |      86.301%      |     82.864%      |     91.921%    |    95.025%    |
=====================================================================================================================








"""
