"""
=================================================================================================================================
----------------------------------------> mTAN GRU TEST LOG  <------------------------------------------
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



TEST 3:
---------------------------------------->
Model: mTAN_v3 (old) -> Hyperparamters: 
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





TEST 4:
---------------------------------------->
Model: mTAN_v3 -> Hyperparamters: 
Learnig rate = 0.0003 
Optimiser = AdamW 
Loss = CrossEntropyLoss 
Batch size = 10 
Epochs = 30 
Model structure: 
mTAN_enc(
  (att): multiTimeAttention(
    (linears): ModuleList(
      (0): Linear(in_features=256, out_features=256, bias=True)
      (1): Linear(in_features=256, out_features=256, bias=True)
      (2): Linear(in_features=4, out_features=64, bias=True)
    )
  )
  (GRU): GRU(64, 64, num_layers=2, batch_first=True, bidirectional=True)
  (periodic): Linear(in_features=1, out_features=255, bias=True)
  (linear): Linear(in_features=1, out_features=1, bias=True)
  (classifier): Sequential(
    (0): Linear(in_features=128, out_features=128, bias=True)
    (1): Linear(in_features=128, out_features=64, bias=True)
    (2): Linear(in_features=64, out_features=6, bias=True)
  )
) 
Total parameters = 281990
Data: non_linear, v1 
Sequence length = 2931 
Batch size = 10 
Shuffled = True

Metric table
=====================================================================================================================
|  Training accuracy  |  Training loss  |  Validation accuracy  |  Validation loss  |  Test accuracy  |  Test loss  |
=====================================================================================================================
|        93.717%      |      0.162      |        93.743%        |       0.268       |      93.765%    |     0.137   |
=====================================================================================================================

Class F1-score table
=====================================================================================================================
|   Drifting longlines   |    Fixed gear    |   Pole and line   |   Purse seines   |    Trawlers    |    Trollers   |
=====================================================================================================================
|         97.205%        |      86.075%     |      86.406%      |     84.830%      |     92.970%    |    97.229%    |
=====================================================================================================================


Predict     0           1           2           3           4           5           
Actual
0           12448       122         10          161         78          0           

1           37          2115        0           106         187         0           

2           41          0           232         15          1           0           

3           185         70          6           1918        90          1           

4           82          156         0           51          4285        1           

5           0           6           0           1           2           193         


Overall Statistics : 

ACC Macro                                                         0.97922
F1 Macro                                                          0.90787
FPR Macro                                                         0.01499
Kappa                                                             0.89867
Overall ACC                                                       0.93765
PPV Macro                                                         0.92158
SOA1(Landis & Koch)                                               Almost Perfect
TPR Macro                                                         0.89597
Zero-one Loss                                                     1409

Class Statistics :

Classes                                                           0             1             2             3             4             5             
ACC(Accuracy)                                                     0.96832       0.96973       0.99677       0.96965       0.97133       0.99951       
AUC(Area under the ROC curve)                                     0.96789       0.92373       0.90103       0.91425       0.95838       0.97768       
AUCI(AUC value interpretation)                                    Excellent     Excellent     Excellent     Excellent     Excellent     Excellent     
F1(F1 score - harmonic mean of precision and sensitivity)         0.97204       0.86081       0.86406       0.8483        0.9297        0.97229       
FN(False negative/miss/type 2 error)                              371           330           57            352           290           9             
FP(False positive/type 1 error/false alarm)                       345           354           16            334           358           2             
FPR(Fall-out or false positive rate)                              0.03527       0.01756       0.00072       0.01643       0.01986       9e-05         
N(Condition negative)                                             9781          20155         22311         20330         18025         22398         
P(Condition positive or support)                                  12819         2445          289           2270          4575          202           
POP(Population)                                                   22600         22600         22600         22600         22600         22600         
PPV(Precision or positive predictive value)                       0.97303       0.85662       0.93548       0.85169       0.92289       0.98974       
TN(True negative/correct rejection)                               9436          19801         22295         19996         17667         22396         
TON(Test outcome negative)                                        9807          20131         22352         20348         17957         22405         
TOP(Test outcome positive)                                        12793         2469          248           2252          4643          195           
TP(True positive/hit)                                             12448         2115          232           1918          4285          193           
TPR(Sensitivity, recall, hit rate, or true positive rate)         0.97106       0.86503       0.80277       0.84493       0.93661       0.95545       


NOTES: Best performance since the GRU with linearly interpolated, let's continue training to push it further - no callbacks used!
Key point, this is used with reference time points for non linear interpolation to compress the time series into 1 minute bins per 24 hours.



TEST 5:
---------------------------------------->
Model: mTAN_v5 -> Hyperparamters: 
Learnig rate = 0.0003 
Optimiser = AdamW 
Loss = CrossEntropyLoss 
Batch size = 10 
Epochs = 0 
Model structure: 
mTAN_enc(
  (att): multiTimeAttention(
    (linears): ModuleList(
      (0): Linear(in_features=256, out_features=256, bias=True)
      (1): Linear(in_features=256, out_features=256, bias=True)
      (2): Linear(in_features=4, out_features=64, bias=True)
    )
  )
  (GRU): GRU(64, 64, num_layers=2, batch_first=True, bidirectional=True)
  (periodic): Linear(in_features=1, out_features=255, bias=True)
  (linear): Linear(in_features=1, out_features=1, bias=True)
  (classifier): Sequential(
    (0): Linear(in_features=128, out_features=128, bias=True)
    (1): Linear(in_features=128, out_features=64, bias=True)
    (2): Linear(in_features=64, out_features=6, bias=True)
  )
) 
Total parameters = 281990
Data: non_linear, v1 
Sequence length = 2931 
Batch size = 10 
Shuffled = True

Metric table
=====================================================================================================================
|  Training accuracy  |  Training loss  |  Validation accuracy  |  Validation loss  |  Test accuracy  |  Test loss  |
=====================================================================================================================
|        94.489%      |      0.137      |        94.226%        |       0.146       |      94.292%    |     0.248   |
=====================================================================================================================

Class F1-score table
=====================================================================================================================
|   Drifting longlines   |    Fixed gear    |   Pole and line   |   Purse seines   |    Trawlers    |    Trollers   |
=====================================================================================================================
|         97.478%        |      87.452%     |      85.470%      |     85.406%      |     93.812%    |    97.487%    |
=====================================================================================================================


Predict     0           1           2           3           4           5           
Actual
0           12488       127         24          117         65          0           

1           20          2156        0           116         152         0           

2           32          0           250         7           0           0           

3           187         62          22          1902        96          1           

4           74          139         0           40          4320        1           

5           0           3           0           2           3           194         


Overall Statistics : 

ACC Macro                                                         0.98097
F1 Macro                                                          0.91183
FPR Macro                                                         0.01366
Kappa                                                             0.90723
Overall ACC                                                       0.94292
PPV Macro                                                         0.91326
SOA1(Landis & Koch)                                               Almost Perfect
TPR Macro                                                         0.91066
Zero-one Loss                                                     1290

Class Statistics :

Classes                                                           0             1             2             3             4             5             
ACC(Accuracy)                                                     0.97142       0.97261       0.99624       0.97124       0.97478       0.99956       
AUC(Area under the ROC curve)                                     0.97101       0.93287       0.9315        0.91201       0.96347       0.98015       
AUCI(AUC value interpretation)                                    Excellent     Excellent     Excellent     Excellent     Excellent     Excellent     
F1(F1 score - harmonic mean of precision and sensitivity)         0.97479       0.87447       0.8547        0.85406       0.93811       0.97487       
FN(False negative/miss/type 2 error)                              333           288           39            368           254           8             
FP(False positive/type 1 error/false alarm)                       313           331           46            282           316           2             
FPR(Fall-out or false positive rate)                              0.03201       0.01642       0.00206       0.01387       0.01753       9e-05         
N(Condition negative)                                             9779          20156         22311         20330         18026         22398         
P(Condition positive or support)                                  12821         2444          289           2270          4574          202           
POP(Population)                                                   22600         22600         22600         22600         22600         22600         
PPV(Precision or positive predictive value)                       0.97555       0.86691       0.84459       0.87088       0.93184       0.9898        
TN(True negative/correct rejection)                               9466          19825         22265         20048         17710         22396         
TON(Test outcome negative)                                        9799          20113         22304         20416         17964         22404         
TOP(Test outcome positive)                                        12801         2487          296           2184          4636          196           
TP(True positive/hit)                                             12488         2156          250           1902          4320          194           
TPR(Sensitivity, recall, hit rate, or true positive rate)         0.97403       0.88216       0.86505       0.83789       0.94447       0.9604        


NOTES: CHECKPOINT MODEL


















"""
