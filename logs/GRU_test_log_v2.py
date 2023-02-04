"""
=================================================================================================================================
----------------------------------------> GRU TEST LOG FOR LINEARLY INTERPOLATED DATA <------------------------------------------
=================================================================================================================================

TEST 1:
---------------------------------------->
Model: GRU_v1 -> Hyperparamters: 
Learnig rate = 0.0003 
Optimiser = AdamW 
Loss = CrossEntropyLoss 
Batch size = 64 
Epochs = 28 
Model structure: 
GRU(
  (gru): GRU(4, 64, num_layers=2, batch_first=True, bidirectional=True)
  (fc_1): Linear(in_features=128, out_features=128, bias=True)
  (fc_2): Linear(in_features=128, out_features=64, bias=True)
  (fc_3): Linear(in_features=64, out_features=6, bias=True)
  (softmax): LogSoftmax(dim=1)
) 
Total parameters = 126534
Data: linear_interp, v4, 
Sequence length = 288.0 
Batch size = 64 
Shuffled = True

Metric table
=====================================================================================================================
|  Training accuracy  |  Training loss  |  Validation accuracy  |  Validation loss  |  Test accuracy  |  Test loss  |
=====================================================================================================================
|        91.108%      |      0.221      |        90.659%        |       0.241       |      90.848%    |     0.228   |
=====================================================================================================================

Class F1-score table
=====================================================================================================================
|   Drifting longlines   |    Fixed gear    |   Pole and line   |   Purse seines   |    Trawlers    |    Trollers   |
=====================================================================================================================
|         96.055%        |      82.523%     |      89.288%      |     80.290%      |     89.918%    |    92.414%    |
=====================================================================================================================


Predict     0           1           2           3           4           5           
Actual
0           16419       202         67          257         150         0           

1           53          4177        1           115         459         31          

2           65          4           696         8           0           0           

3           414         390         21          3046        203         0           

4           141         465         1           87          6800        7           

5           0           48          0           2           11          604         


Overall Statistics : 

ACC Macro                                                         0.96946
F1 Macro                                                          0.88416
FPR Macro                                                         0.02058
Kappa                                                             0.86532
Overall ACC                                                       0.90837
PPV Macro                                                         0.88929
SOA1(Landis & Koch)                                               Almost Perfect
TPR Macro                                                         0.88118
Zero-one Loss                                                     3202

Class Statistics :

Classes                                                           0             1             2             3             4             5             
ACC(Accuracy)                                                     0.9614        0.9494        0.99522       0.95716       0.95639       0.99717       
AUC(Area under the ROC curve)                                     0.96138       0.91345       0.94888       0.86624       0.93828       0.95358       
AUCI(AUC value interpretation)                                    Excellent     Excellent     Excellent     Very Good     Excellent     Excellent     
F1(F1 score - harmonic mean of precision and sensitivity)         0.96054       0.82533       0.89288       0.80274       0.89923       0.92425       
FN(False negative/miss/type 2 error)                              676           659           77            1028          701           61            
FP(False positive/type 1 error/false alarm)                       673           1109          90            469           823           38            
FPR(Fall-out or false positive rate)                              0.03771       0.03683       0.00263       0.01519       0.02999       0.00111       
N(Condition negative)                                             17849         30108         34171         30870         27443         34279         
P(Condition positive or support)                                  17095         4836          773           4074          7501          665           
POP(Population)                                                   34944         34944         34944         34944         34944         34944         
PPV(Precision or positive predictive value)                       0.96062       0.7902        0.8855        0.86657       0.89204       0.94081       
TN(True negative/correct rejection)                               17176         28999         34081         30401         26620         34241         
TON(Test outcome negative)                                        17852         29658         34158         31429         27321         34302         
TOP(Test outcome positive)                                        17092         5286          786           3515          7623          642           
TP(True positive/hit)                                             16419         4177          696           3046          6800          604           
TPR(Sensitivity, recall, hit rate, or true positive rate)         0.96046       0.86373       0.90039       0.74767       0.90655       0.90827   

NOTES: Strong start for the linearly interpolated data with 5 minute intervals


TEST 1:
---------------------------------------->
Model: GRU_v2 -> Hyperparamters: 
Learnig rate = 0.0003 
Optimiser = AdamW 
Loss = CrossEntropyLoss 
Batch size = 128 
Epochs = 50 
Model structure: 
GRU(
  (gru): GRU(4, 64, num_layers=2, batch_first=True, bidirectional=True)
  (fc_1): Linear(in_features=128, out_features=128, bias=True)
  (fc_2): Linear(in_features=128, out_features=64, bias=True)
  (fc_3): Linear(in_features=64, out_features=6, bias=True)
  (softmax): LogSoftmax(dim=1)
) 
Total parameters = 126534
Data: linear_interp, v4, 
Sequence length = 288.0 
Batch size = 128 
Shuffled = True

Metric table
=====================================================================================================================
|  Training accuracy  |  Training loss  |  Validation accuracy  |  Validation loss  |  Test accuracy  |  Test loss  |
=====================================================================================================================
|        92.129%      |      0.189      |        92.302%        |       0.210       |      92.356%    |     0.190   |
=====================================================================================================================

Class F1-score table
=====================================================================================================================
|   Drifting longlines   |    Fixed gear    |   Pole and line   |   Purse seines   |    Trawlers    |    Trollers   |
=====================================================================================================================
|         96.377%        |      85.714%     |      90.135%      |     83.800%      |     91.840%    |    96.557%    |
=====================================================================================================================


Predict     0           1           2           3           4           5           
Actual
0           16601       152         34          196         117         0           

1           83          4137        1           271         324         19          

2           79          3           667         22          1           0           

3           403         166         6           3368        130         0           

4           183         350         0           105         6852        8           

5           1           12          0           5           1           647         


Overall Statistics : 

ACC Macro                                                         0.97451
F1 Macro                                                          0.90731
FPR Macro                                                         0.01782
Kappa                                                             0.88727
Overall ACC                                                       0.92353
PPV Macro                                                         0.91483
SOA1(Landis & Koch)                                               Almost Perfect
TPR Macro                                                         0.90044
Zero-one Loss                                                     2672

Class Statistics :

Classes                                                           0             1             2             3             4             5             
ACC(Accuracy)                                                     0.96429       0.96048       0.99582       0.96268       0.96512       0.99868       
AUC(Area under the ROC curve)                                     0.96442       0.91648       0.93139       0.90375       0.94648       0.98534       
AUCI(AUC value interpretation)                                    Excellent     Excellent     Excellent     Excellent     Excellent     Excellent     
F1(F1 score - harmonic mean of precision and sensitivity)         0.96377       0.85697       0.90135       0.83781       0.91831       0.96567       
FN(False negative/miss/type 2 error)                              499           698           105           705           646           19            
FP(False positive/type 1 error/false alarm)                       749           683           41            599           573           27            
FPR(Fall-out or false positive rate)                              0.04197       0.02268       0.0012        0.0194        0.02088       0.00079       
N(Condition negative)                                             17844         30109         34172         30871         27446         34278         
P(Condition positive or support)                                  17100         4835          772           4073          7498          666           
POP(Population)                                                   34944         34944         34944         34944         34944         34944         
PPV(Precision or positive predictive value)                       0.95683       0.8583        0.94209       0.849         0.92283       0.95994       
TN(True negative/correct rejection)                               17095         29426         34131         30272         26873         34251         
TON(Test outcome negative)                                        17594         30124         34236         30977         27519         34270         
TOP(Test outcome positive)                                        17350         4820          708           3967          7425          674           
TP(True positive/hit)                                             16601         4137          667           3368          6852          647           
TPR(Sensitivity, recall, hit rate, or true positive rate)         0.97082       0.85564       0.86399       0.82691       0.91384       0.97147       

"""
