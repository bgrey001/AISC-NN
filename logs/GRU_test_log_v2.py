"""
============================================================================================
----------------------------------------> GRU <------------------------------------------
============================================================================================

TEST 1:
---------------------------------------->
Model: GRU_v3 -> Hyperparamters: 
Learnig rate = 0.0003 
Optimiser = AdamW 
Loss = CrossEntropyLoss 
Batch size = 64 
Epochs = 50 
Model structure 
GRU(
  (gru): GRU(5, 64, num_layers=2, batch_first=True, dropout=0.1, bidirectional=True)
  (fc_1): Linear(in_features=128, out_features=128, bias=True)
  (fc_2): Linear(in_features=128, out_features=64, bias=True)
  (fc_3): Linear(in_features=64, out_features=6, bias=True)
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
|        95.894%      |      0.110      |        94.626%        |       0.167       |      95.007%    |     0.123   |
=====================================================================================================================

Class F1-score table
=====================================================================================================================
|   Drifting longlines   |    Fixed gear    |   Pole and line   |   Purse seines   |    Trawlers    |    Trollers   |
=====================================================================================================================
|         97.755%        |      87.841%     |      87.356%      |     89.075%      |     94.714%    |    94.492%    |
=====================================================================================================================


Predict     0           1           2           3           4           5           
Actual
0           12434       117         39          149         53          0           

1           33          2142        0           76          113         7           

2           19          0           266         8           0           0           

3           87          84          11          2070        25          2           

4           74          151         0           64          4329        1           

5           0           12          0           2           2           222         


Overall Statistics : 

ACC Macro                                                         0.98334
F1 Macro                                                          0.91868
FPR Macro                                                         0.01131
Kappa                                                             0.91935
Overall ACC                                                       0.95003
PPV Macro                                                         0.91128
SOA1(Landis & Koch)                                               Almost Perfect
TPR Macro                                                         0.92693
Zero-one Loss                                                     1129

Class Statistics :

Classes                                                           0             1             2             3             4             5             
ACC(Accuracy)                                                     0.97473       0.97375       0.99659       0.97751       0.97862       0.99885       
AUC(Area under the ROC curve)                                     0.97514       0.94271       0.9528        0.94679       0.96324       0.96616       
AUCI(AUC value interpretation)                                    Excellent     Excellent     Excellent     Excellent     Excellent     Excellent     
F1(F1 score - harmonic mean of precision and sensitivity)         0.97755       0.87841       0.87356       0.89071       0.94716       0.94468       
FN(False negative/miss/type 2 error)                              358           229           27            209           290           16            
FP(False positive/type 1 error/false alarm)                       213           364           50            299           193           10            
FPR(Fall-out or false positive rate)                              0.02173       0.018         0.00224       0.01472       0.01074       0.00045       
N(Condition negative)                                             9800          20221         22299         20313         17973         22354         
P(Condition positive or support)                                  12792         2371          293           2279          4619          238           
POP(Population)                                                   22592         22592         22592         22592         22592         22592         
PPV(Precision or positive predictive value)                       0.98316       0.85475       0.84177       0.87379       0.95732       0.9569        
TN(True negative/correct rejection)                               9587          19857         22249         20014         17780         22344         
TON(Test outcome negative)                                        9945          20086         22276         20223         18070         22360         
TOP(Test outcome positive)                                        12647         2506          316           2369          4522          232           
TP(True positive/hit)                                             12434         2142          266           2070          4329          222           
TPR(Sensitivity, recall, hit rate, or true positive rate)         0.97201       0.90342       0.90785       0.90829       0.93722       0.93277 


"""
