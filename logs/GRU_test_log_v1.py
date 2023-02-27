"""
=======================================================================================================================
----------------------------------------> GRU TEST LOG FOR ZERO-PADDED DATA <------------------------------------------
=======================================================================================================================

TEST 1:
---------------------------------------->
Model: GRU_v0 -> Hyperparamters: 
Learnig rate = 0.0003 
Optimiser = AdamW 
Loss = CrossEntropyLoss 
Batch size = 64 
Epochs = 20 
Model structure: 
GRU(
  (gru): GRU(5, 16, batch_first=True)
  (fc_1): Linear(in_features=16, out_features=128, bias=True)
  (fc_2): Linear(in_features=128, out_features=64, bias=True)
  (fc_3): Linear(in_features=64, out_features=6, bias=True)
  (softmax): LogSoftmax(dim=1)
)
Total parameters = 
Data: padded, v3, varying intervals 
Sequence length = 2931 
Batch size = 64 
Shuffled = True

Metric table
=====================================================================================================================
|  Training accuracy  |  Training loss  |  Validation accuracy  |  Validation loss  |  Test accuracy  |  Test loss  |
=====================================================================================================================
|        72.590%      |      0.784      |        73.969%        |       0.774       |      74.907%    |     0.776   |
=====================================================================================================================

Class F1-score table
=====================================================================================================================
|   Drifting longlines   |    Fixed gear    |   Pole and line   |   Purse seines   |    Trawlers    |    Trollers   |
=====================================================================================================================
|         87.523%        |      45.945%     |      33.516%      |     66.378%      |     63.247%    |    23.741%    |
=====================================================================================================================


Predict     0           1           2           3           4           5           
Actual
0           10898       144         7           294         1452        1           

1           36          911         0           33          1389        1           

2           181         0           61          17          34          0           

3           276         363         0           1379        257         3           

4           703         137         3           130         3642        1           

5           11          43          0           24          128         33          


Overall Statistics : 

ACC Macro                                                         0.91637
F1 Macro                                                          0.53387
FPR Macro                                                         0.06063
Kappa                                                             0.5942
Overall ACC                                                       0.74911
PPV Macro                                                         0.73967
SOA1(Landis & Koch)                                               Moderate
TPR Macro                                                         0.49611
Zero-one Loss                                                     5668

Class Statistics :

Classes                                                           0             1             2             3             4             5             
ACC(Accuracy)                                                     0.86256       0.90501       0.98929       0.93816       0.81259       0.99062       
AUC(Area under the ROC curve)                                     0.86423       0.67521       0.60387       0.79042       0.80382       0.5689        
AUCI(AUC value interpretation)                                    Very Good     Fair          Fair          Good          Very Good     Poor          
F1(F1 score - harmonic mean of precision and sensitivity)         0.87531       0.45917       0.33516       0.66378       0.6324        0.23741       
FN(False negative/miss/type 2 error)                              1898          1459          232           899           974           206           
FP(False positive/type 1 error/false alarm)                       1207          687           10            498           3260          6             
FPR(Fall-out or false positive rate)                              0.12321       0.03397       0.00045       0.02452       0.18135       0.00027       
N(Condition negative)                                             9796          20222         22299         20314         17976         22353         
P(Condition positive or support)                                  12796         2370          293           2278          4616          239           
POP(Population)                                                   22592         22592         22592         22592         22592         22592         
PPV(Precision or positive predictive value)                       0.90029       0.57009       0.85915       0.73468       0.52767       0.84615       
TN(True negative/correct rejection)                               8589          19535         22289         19816         14716         22347         
TON(Test outcome negative)                                        10487         20994         22521         20715         15690         22553         
TOP(Test outcome positive)                                        12105         1598          71            1877          6902          39            
TP(True positive/hit)                                             10898         911           61            1379          3642          33            
TPR(Sensitivity, recall, hit rate, or true positive rate)         0.85167       0.38439       0.20819       0.60536       0.78899       0.13808 


NOTES: Minimal GRU model, 1 GRU unit, 16 hidden units, monodirectional, 3 fully connected layers as this worked best for the CNN, using AdamW, not using Reed and Marks nonrandom initialisation with validation accuracy metric



TEST 2:
---------------------------------------->
Model: GRU_v1 -> Hyperparamters: 
Learnig rate = 0.0003 
Optimiser = AdamW 
Loss = CrossEntropyLoss 
Batch size = 64 
Epochs = 20 
Model structure: 
GRU(
  (gru): GRU(5, 32, num_layers=2, batch_first=True)
  (fc_1): Linear(in_features=32, out_features=128, bias=True)
  (fc_2): Linear(in_features=128, out_features=64, bias=True)
  (fc_3): Linear(in_features=64, out_features=6, bias=True)
  (softmax): LogSoftmax(dim=1)
)
Total parameters = 
Data: padded, v3, varying intervals 
Sequence length = 2931 
Batch size = 64 
Shuffled = True

Metric table
=====================================================================================================================
|  Training accuracy  |  Training loss  |  Validation accuracy  |  Validation loss  |  Test accuracy  |  Test loss  |
=====================================================================================================================
|        90.572%      |      0.238      |        90.178%        |       0.229       |      90.271%    |     0.240   |
=====================================================================================================================

Class F1-score table
=====================================================================================================================
|   Drifting longlines   |    Fixed gear    |   Pole and line   |   Purse seines   |    Trawlers    |    Trollers   |
=====================================================================================================================
|         96.020%        |      75.787%     |      74.858%      |     83.009%      |     86.854%    |    76.033%    |
=====================================================================================================================


Predict     0           1           2           3           4           5           
Actual
0           12412       140         24          118         101         0           

1           64          1818        0           163         272         52          

2           78          0           198         9           8           0           

3           216         97          11          1898        52          5           

4           285         336         3           103         3887        3           

5           3           36          0           4           12          184         


Overall Statistics : 

ACC Macro                                                         0.96761
F1 Macro                                                          0.82124
FPR Macro                                                         0.02412
Kappa                                                             0.84117
Overall ACC                                                       0.90284
PPV Macro                                                         0.83616
SOA1(Landis & Koch)                                               Almost Perfect
TPR Macro                                                         0.80964
Zero-one Loss                                                     2195

Class Statistics :

Classes                                                           0             1             2             3             4             5             
ACC(Accuracy)                                                     0.95445       0.94865       0.99411       0.96556       0.94799       0.99491       
AUC(Area under the ROC curve)                                     0.95206       0.86865       0.83703       0.90664       0.90857       0.8836        
AUCI(AUC value interpretation)                                    Excellent     Very Good     Very Good     Excellent     Excellent     Very Good     
F1(F1 score - harmonic mean of precision and sensitivity)         0.9602        0.75813       0.74858       0.82991       0.8687        0.7619        
FN(False negative/miss/type 2 error)                              383           551           95            381           730           55            
FP(False positive/type 1 error/false alarm)                       646           609           38            397           445           60            
FPR(Fall-out or false positive rate)                              0.06594       0.03011       0.0017        0.01954       0.02476       0.00268       
N(Condition negative)                                             9797          20223         22299         20313         17975         22353         
P(Condition positive or support)                                  12795         2369          293           2279          4617          239           
POP(Population)                                                   22592         22592         22592         22592         22592         22592         
PPV(Precision or positive predictive value)                       0.95053       0.74907       0.83898       0.82702       0.89728       0.7541        
TN(True negative/correct rejection)                               9151          19614         22261         19916         17530         22293         
TON(Test outcome negative)                                        9534          20165         22356         20297         18260         22348         
TOP(Test outcome positive)                                        13058         2427          236           2295          4332          244           
TP(True positive/hit)                                             12412         1818          198           1898          3887          184           
TPR(Sensitivity, recall, hit rate, or true positive rate)         0.97007       0.76741       0.67577       0.83282       0.84189       0.76987       


NOTES: Slightly expanded model, using 2 GRU units and larger hidden dimension of 32, converging much faster on a high accuracy score -> clearly well suited for the task, continue to explore hyperparameters for improved performance



TEST 3:
---------------------------------------->
Model: GRU_v0 -> Hyperparamters: 
Learnig rate = 0.0003 
Optimiser = AdamW 
Loss = CrossEntropyLoss 
Batch size = 64 
Epochs = 20 
Model structure: 
GRU(
  (gru): GRU(5, 32, num_layers=2, batch_first=True, bidirectional=True)
  (fc_1): Linear(in_features=64, out_features=128, bias=True)
  (fc_2): Linear(in_features=128, out_features=64, bias=True)
  (fc_3): Linear(in_features=64, out_features=6, bias=True)
  (softmax): LogSoftmax(dim=1)
)
Total parameters = 
Data: padded, v3, varying intervals 
Sequence length = 2931 
Batch size = 64 
Shuffled = True

Metric table
=====================================================================================================================
|  Training accuracy  |  Training loss  |  Validation accuracy  |  Validation loss  |  Test accuracy  |  Test loss  |
=====================================================================================================================
|        92.491%      |      0.200      |        92.303%        |       0.199       |      92.236%    |     0.246   |
=====================================================================================================================

Class F1-score table
=====================================================================================================================
|   Drifting longlines   |    Fixed gear    |   Pole and line   |   Purse seines   |    Trawlers    |    Trollers   |
=====================================================================================================================
|         96.720%        |      80.972%     |      82.310%      |     84.425%      |     90.356%    |    84.494%    |
=====================================================================================================================


Predict     0           1           2           3           4           5           
Actual
0           12457       144         28          95          69          0           

1           58          1998        0           43          257         12          

2           49          0           229         15          0           0           

3           204         170         5           1804        92          4           

4           197         223         1           35          4162        2           

5           1           34          0           3           13          188         


Overall Statistics : 

ACC Macro                                                         0.97412
F1 Macro                                                          0.86549
FPR Macro                                                         0.01932
Kappa                                                             0.87322
Overall ACC                                                       0.92236
PPV Macro                                                         0.88871
SOA1(Landis & Koch)                                               Almost Perfect
TPR Macro                                                         0.84635
Zero-one Loss                                                     1754

Class Statistics :

Classes                                                           0             1             2             3             4             5             
ACC(Accuracy)                                                     0.9626        0.95835       0.99566       0.97052       0.96065       0.99695       
AUC(Area under the ROC curve)                                     0.9609        0.90776       0.89002       0.89109       0.93844       0.8929        
AUCI(AUC value interpretation)                                    Excellent     Excellent     Very Good     Very Good     Excellent     Very Good     
F1(F1 score - harmonic mean of precision and sensitivity)         0.9672        0.8094        0.82374       0.84417       0.90351       0.84494       
FN(False negative/miss/type 2 error)                              336           370           64            475           458           51            
FP(False positive/type 1 error/false alarm)                       509           571           34            191           431           18            
FPR(Fall-out or false positive rate)                              0.05194       0.02823       0.00152       0.0094        0.02398       0.00081       
N(Condition negative)                                             9799          20224         22299         20313         17972         22353         
P(Condition positive or support)                                  12793         2368          293           2279          4620          239           
POP(Population)                                                   22592         22592         22592         22592         22592         22592         
PPV(Precision or positive predictive value)                       0.96074       0.77773       0.87072       0.90426       0.90616       0.91262       
TN(True negative/correct rejection)                               9290          19653         22265         20122         17541         22335         
TON(Test outcome negative)                                        9626          20023         22329         20597         17999         22386         
TOP(Test outcome positive)                                        12966         2569          263           1995          4593          206           
TP(True positive/hit)                                             12457         1998          229           1804          4162          188           
TPR(Sensitivity, recall, hit rate, or true positive rate)         0.97374       0.84375       0.78157       0.79158       0.90087       0.78661       


NOTES: Introduced bidirectionality with the same previous hyperparameters to see the effects: decent improvement over the monodirectional model, will expand hidden units and run for more epochs in future iterations




TEST 4:
---------------------------------------->
Model: GRU_v3 -> Hyperparamters: 
Learnig rate = 0.0003 
Optimiser = AdamW 
Loss = CrossEntropyLoss 
Batch size = 64 
Epochs = 50 
Model structure: 
GRU(
  (gru): GRU(5, 64, num_layers=2, batch_first=True, dropout=0.1, bidirectional=True)
  (fc_1): Linear(in_features=128, out_features=128, bias=True)
  (fc_2): Linear(in_features=128, out_features=64, bias=True)
  (fc_3): Linear(in_features=64, out_features=6, bias=True)
  (softmax): LogSoftmax(dim=1)
)
Total parameters = 126918
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
TPR(Sensitivity, recall, hit rate, or true positive rate)         



NOTES: Strongest results overall for either models, beating Kalaiselvi et al.'s results -> 95% accuracy overall will be difficult to beat in the following months, this proves the hypothesis that there exists a representative relationship between movement patterns and fishing method


TEST 5:
---------------------------------------->
Model: GRU_v6 -> Hyperparamters: 
Learnig rate = 0.0003 
Optimiser = AdamW 
Loss = CrossEntropyLoss 
Batch size = 64 
Epochs = 10 
Model structure: 
GRU(
  (gru): GRU(5, 64, num_layers=2, batch_first=True, bidirectional=True)
  (fc_1): Linear(in_features=128, out_features=128, bias=True)
  (fc_2): Linear(in_features=128, out_features=64, bias=True)
  (fc_3): Linear(in_features=64, out_features=6, bias=True)
  (softmax): LogSoftmax(dim=1)
) 
Total parameters = 126918
Data: padded, v3, varying intervals 
Sequence length = 2931 
Batch size = 64 
Shuffled = True

Metric table
=====================================================================================================================
|  Training accuracy  |  Training loss  |  Validation accuracy  |  Validation loss  |  Test accuracy  |  Test loss  |
=====================================================================================================================
|        96.529%      |      0.081      |        95.100%        |       0.132       |      95.073%    |     0.125   |
=====================================================================================================================

Class F1-score table
=====================================================================================================================
|   Drifting longlines   |    Fixed gear    |   Pole and line   |   Purse seines   |    Trawlers    |    Trollers   |
=====================================================================================================================
|         97.750%        |      88.298%     |      87.002%      |     89.612%      |     94.531%    |    95.445%    |
=====================================================================================================================


Predict     0           1           2           3           4           5           
Actual
0           12445       110         32          120         86          0           

1           44          2145        0           80          101         1           

2           29          0           251         13          0           0           

3           85          75          1           2066        52          1           

4           68          146         0           50          4353        0           

5           0           14          0           3           2           219         


Overall Statistics : 

ACC Macro                                                         0.98358
F1 Macro                                                          0.92089
FPR Macro                                                         0.01137
Kappa                                                             0.92036
Overall ACC                                                       0.95073
PPV Macro                                                         0.92531
SOA1(Landis & Koch)                                               Almost Perfect
TPR Macro                                                         0.91721
Zero-one Loss                                                     1113

Class Statistics :

Classes                                                           0             1             2             3             4             5             
ACC(Accuracy)                                                     0.97459       0.97473       0.99668       0.97875       0.97765       0.99907       
AUC(Area under the ROC curve)                                     0.97487       0.94381       0.92759       0.94652       0.96471       0.96004       
AUCI(AUC value interpretation)                                    Excellent     Excellent     Excellent     Excellent     Excellent     Excellent     
F1(F1 score - harmonic mean of precision and sensitivity)         0.97746       0.88253       0.87002       0.89592       0.94517       0.95425       
FN(False negative/miss/type 2 error)                              348           226           42            214           264           19            
FP(False positive/type 1 error/false alarm)                       226           345           33            266           241           2             
FPR(Fall-out or false positive rate)                              0.02306       0.01706       0.00148       0.0131        0.01341       9e-05         
N(Condition negative)                                             9799          20221         22299         20312         17975         22354         
P(Condition positive or support)                                  12793         2371          293           2280          4617          238           
POP(Population)                                                   22592         22592         22592         22592         22592         22592         
PPV(Precision or positive predictive value)                       0.98216       0.86145       0.8838        0.88593       0.94754       0.99095       
TN(True negative/correct rejection)                               9573          19876         22266         20046         17734         22352         
TON(Test outcome negative)                                        9921          20102         22308         20260         17998         22371         
TOP(Test outcome positive)                                        12671         2490          284           2332          4594          221           
TP(True positive/hit)                                             12445         2145          251           2066          4353          219           
TPR(Sensitivity, recall, hit rate, or true positive rate)         0.9728        0.90468       0.85666       0.90614       0.94282       0.92017       


model.save_model(6)
GRU_v6 state_dict successfully saved
GRU_v6 history saved


TEST 6:
---------------------------------------->
Model: GRU_v7 -> Hyperparamters: 
Learnig rate = 0.0003 
Optimiser = AdamW 
Loss = CrossEntropyLoss 
Batch size = 64 
Epochs = 70 
Model structure: 
GRU(
  (gru): GRU(5, 64, num_layers=2, batch_first=True, bidirectional=True)
  (fc_1): Linear(in_features=128, out_features=128, bias=True)
  (fc_2): Linear(in_features=128, out_features=64, bias=True)
  (fc_3): Linear(in_features=64, out_features=6, bias=True)
  (softmax): LogSoftmax(dim=1)
) 
Total parameters = 126918
Data: padded, v3, varying intervals 
Sequence length = 2931 
Batch size = 64 
Shuffled = True
Post training L1-norm global unstructured trainable weight pruning: 0.2

Metric table
=====================================================================================================================
|  Training accuracy  |  Training loss  |  Validation accuracy  |  Validation loss  |  Test accuracy  |  Test loss  |
=====================================================================================================================
|        96.500%      |      0.088      |        95.100%        |       0.132       |      95.122%    |     0.130   |
=====================================================================================================================

Class F1-score table
=====================================================================================================================
|   Drifting longlines   |    Fixed gear    |   Pole and line   |   Purse seines   |    Trawlers    |    Trollers   |
=====================================================================================================================
|         97.714%        |      88.266%     |      86.515%      |     89.851%      |     94.657%    |    95.708%    |
=====================================================================================================================


Predict     0           1           2           3           4           5           
Actual
0           12499       127         29          86          54          0           

1           36          2173        0           43          115         3           

2           35          0           248         10          0           0           

3           127         100         3           2013        34          1           

4           91          145         0           47          4334        0           

5           0           9           0           4           3           223         


Overall Statistics : 

ACC Macro                                                         0.98374
F1 Macro                                                          0.92125
FPR Macro                                                         0.01179
Kappa                                                             0.92087
Overall ACC                                                       0.95122
PPV Macro                                                         0.92745
SOA1(Landis & Koch)                                               Almost Perfect
TPR Macro                                                         0.91593
Zero-one Loss                                                     1102

Class Statistics :

Classes                                                           0             1             2             3             4             5             
ACC(Accuracy)                                                     0.97411       0.97442       0.99659       0.97986       0.97836       0.99911       
AUC(Area under the ROC curve)                                     0.97368       0.94902       0.92249       0.93716       0.96362       0.96644       
AUCI(AUC value interpretation)                                    Excellent     Excellent     Excellent     Excellent     Excellent     Excellent     
F1(F1 score - harmonic mean of precision and sensitivity)         0.97713       0.88262       0.86562       0.89846       0.9466        0.95708       
FN(False negative/miss/type 2 error)                              296           197           45            265           283           16            
FP(False positive/type 1 error/false alarm)                       289           381           32            190           206           4             
FPR(Fall-out or false positive rate)                              0.0295        0.01884       0.00144       0.00935       0.01146       0.00018       
N(Condition negative)                                             9797          20222         22299         20314         17975         22353         
P(Condition positive or support)                                  12795         2370          293           2278          4617          239           
POP(Population)                                                   22592         22592         22592         22592         22592         22592         
PPV(Precision or positive predictive value)                       0.9774        0.85082       0.88571       0.91375       0.95463       0.98238       
TN(True negative/correct rejection)                               9508          19841         22267         20124         17769         22349         
TON(Test outcome negative)                                        9804          20038         22312         20389         18052         22365         
TOP(Test outcome positive)                                        12788         2554          280           2203          4540          227           
TP(True positive/hit)                                             12499         2173          248           2013          4334          223           
TPR(Sensitivity, recall, hit rate, or true positive rate)         0.97687       0.91688       0.84642       0.88367       0.9387        0.93305       


NOTES: Strongest results yet, improved upon v3 using nonrandom initialisation and weight pruning.


TEST 6:
---------------------------------------->
Model: GRU_v10 -> Hyperparamters: 
Learnig rate = 0.0003 
Optimiser = AdamW 
Loss = CrossEntropyLoss 
Batch size = 64 
Epochs = 50 
Model structure: 
GRU(
  (gru): GRU(5, 64, num_layers=2, batch_first=True, bidirectional=True)
  (relu): ReLU()
  (fc_1): Linear(in_features=128, out_features=128, bias=True)
  (fc_2): Linear(in_features=128, out_features=64, bias=True)
  (fc_3): Linear(in_features=64, out_features=6, bias=True)
) 
Total parameters = 126918
Data: varying, v3 
Sequence length = 2931 
Batch size = 64 
Shuffled = True

Metric table
=====================================================================================================================
|  Training accuracy  |  Training loss  |  Validation accuracy  |  Validation loss  |  Test accuracy  |  Test loss  |
=====================================================================================================================
|        96.029%      |      0.098      |        94.175%        |       0.159       |      94.272%    |     0.135   |
=====================================================================================================================

Class F1-score table
=====================================================================================================================
|   Drifting longlines   |    Fixed gear    |   Pole and line   |   Purse seines   |    Trawlers    |    Trollers   |
=====================================================================================================================
|         97.568%        |      85.623%     |      85.714%      |     88.988%      |     93.005%    |    93.157%    |
=====================================================================================================================


Predict     0           1           2           3           4           5           
Actual
0           12461       124         50          93          67          0           

1           56          2145        0           71          94          2           

2           22          0           261         10          0           0           

3           106         98          5           2005        65          1           

4           103         254         0           45          4214        1           

5           0           21          0           2           4           212         


Overall Statistics : 

ACC Macro                                                         0.98091
F1 Macro                                                          0.90684
FPR Macro                                                         0.01337
Kappa                                                             0.90728
Overall ACC                                                       0.94272
PPV Macro                                                         0.90763
SOA1(Landis & Koch)                                               Almost Perfect
TPR Macro                                                         0.90827
Zero-one Loss                                                     1294

Class Statistics :

Classes                                                           0             1             2             3             4             5             
ACC(Accuracy)                                                     0.97251       0.96813       0.99615       0.97805       0.97198       0.99863       
AUC(Area under the ROC curve)                                     0.9723        0.94063       0.94416       0.93425       0.94996       0.94343       
AUCI(AUC value interpretation)                                    Excellent     Excellent     Excellent     Excellent     Excellent     Excellent     
F1(F1 score - harmonic mean of precision and sensitivity)         0.97569       0.85629       0.85714       0.88992       0.93014       0.93187       
FN(False negative/miss/type 2 error)                              334           223           32            275           403           27            
FP(False positive/type 1 error/false alarm)                       287           497           55            221           230           4             
FPR(Fall-out or false positive rate)                              0.02929       0.02457       0.00247       0.01088       0.0128        0.00018       
N(Condition negative)                                             9797          20224         22299         20312         17975         22353         
P(Condition positive or support)                                  12795         2368          293           2280          4617          239           
POP(Population)                                                   22592         22592         22592         22592         22592         22592         
PPV(Precision or positive predictive value)                       0.97749       0.81188       0.82595       0.90072       0.94824       0.98148       
TN(True negative/correct rejection)                               9510          19727         22244         20091         17745         22349         
TON(Test outcome negative)                                        9844          19950         22276         20366         18148         22376         
TOP(Test outcome positive)                                        12748         2642          316           2226          4444          216           
TP(True positive/hit)                                             12461         2145          261           2005          4214          212           
TPR(Sensitivity, recall, hit rate, or true positive rate)         0.9739        0.90583       0.89078       0.87939       0.91271       0.88703       


NOTES: removed softmax and slicing from forward pass final step and changed according to torch documentation, results are still good. 



TEST 7:
---------------------------------------->
Model: GRU_v11 -> Hyperparamters: 
Learnig rate = 0.0003 
Optimiser = AdamW 
Loss = CrossEntropyLoss 
Batch size = 64 
Epochs = 52 
Model structure: 
GRU(
  (gru): GRU(5, 64, num_layers=2, batch_first=True, bidirectional=True)
  (relu): ReLU()
  (fc_1): Linear(in_features=128, out_features=128, bias=True)
  (fc_2): Linear(in_features=128, out_features=64, bias=True)
  (fc_3): Linear(in_features=64, out_features=6, bias=True)
) 
Total parameters = 126918
Data: varying, v3 
Sequence length = 2931 
Batch size = 64 
Shuffled = True

Metric table
=====================================================================================================================
|  Training accuracy  |  Training loss  |  Validation accuracy  |  Validation loss  |  Test accuracy  |  Test loss  |
=====================================================================================================================
|        96.228%      |      0.093      |        95.113%        |       0.141       |      95.038%    |     0.110   |
=====================================================================================================================

Class F1-score table
=====================================================================================================================
|   Drifting longlines   |    Fixed gear    |   Pole and line   |   Purse seines   |    Trawlers    |    Trollers   |
=====================================================================================================================
|         97.704%        |      87.471%     |      89.005%      |     89.373%      |     94.683%    |    94.894%    |
=====================================================================================================================


Predict     0           1           2           3           4           5           
Actual
0           12529       102         19          94          50          0           

1           84          2097        0           58          125         5           

2           27          0           255         11          0           0           

3           126         86          5           2010        52          1           

4           87          129         0           43          4356        2           

5           0           12          1           2           1           223         


Overall Statistics : 

ACC Macro                                                         0.98345
F1 Macro                                                          0.92188
FPR Macro                                                         0.01229
Kappa                                                             0.91923
Overall ACC                                                       0.95034
PPV Macro                                                         0.92862
SOA1(Landis & Koch)                                               Almost Perfect
TPR Macro                                                         0.91548
Zero-one Loss                                                     1122

Class Statistics :

Classes                                                           0             1             2             3             4             5             
ACC(Accuracy)                                                     0.97393       0.9734        0.99721       0.97884       0.97836       0.99894       
AUC(Area under the ROC curve)                                     0.97311       0.93446       0.93459       0.93567       0.96539       0.96635       
AUCI(AUC value interpretation)                                    Excellent     Excellent     Excellent     Excellent     Excellent     Excellent     
F1(F1 score - harmonic mean of precision and sensitivity)         0.97703       0.87466       0.89005       0.89373       0.94685       0.94894       
FN(False negative/miss/type 2 error)                              265           272           38            270           261           16            
FP(False positive/type 1 error/false alarm)                       324           329           25            208           228           8             
FPR(Fall-out or false positive rate)                              0.03307       0.01627       0.00112       0.01024       0.01268       0.00036       
N(Condition negative)                                             9798          20223         22299         20312         17975         22353         
P(Condition positive or support)                                  12794         2369          293           2280          4617          239           
POP(Population)                                                   22592         22592         22592         22592         22592         22592         
PPV(Precision or positive predictive value)                       0.97479       0.86439       0.91071       0.90622       0.95026       0.96537       
TN(True negative/correct rejection)                               9474          19894         22274         20104         17747         22345         
TON(Test outcome negative)                                        9739          20166         22312         20374         18008         22361         
TOP(Test outcome positive)                                        12853         2426          280           2218          4584          231           
TP(True positive/hit)                                             12529         2097          255           2010          4356          223           
TPR(Sensitivity, recall, hit rate, or true positive rate)         0.97929       0.88518       0.87031       0.88158       0.94347       0.93305     



TEST 8:
---------------------------------------->
Model: GRU_v12 -> Hyperparamters: 
Learnig rate = 0.0003 
Optimiser = AdamW 
Loss = CrossEntropyLoss 
Batch size = 64 
Epochs = 60 
Model structure: 
GRU(
  (gru): GRU(5, 64, num_layers=2, batch_first=True, bidirectional=True)
  (relu): ReLU()
  (fc_1): Linear(in_features=128, out_features=128, bias=True)
  (fc_2): Linear(in_features=128, out_features=64, bias=True)
  (fc_3): Linear(in_features=64, out_features=6, bias=True)
) 
Total parameters = 126918
Data: varying, v3 
Sequence length = 2931 
Batch size = 64 
Shuffled = True

Metric table
=====================================================================================================================
|  Training accuracy  |  Training loss  |  Validation accuracy  |  Validation loss  |  Test accuracy  |  Test loss  |
=====================================================================================================================
|        96.469%      |      0.088      |        95.113%        |       0.141       |      95.379%    |     0.140   |
=====================================================================================================================

Class F1-score table
=====================================================================================================================
|   Drifting longlines   |    Fixed gear    |   Pole and line   |   Purse seines   |    Trawlers    |    Trollers   |
=====================================================================================================================
|         97.849%        |      89.256%     |      88.165%      |     89.962%      |     94.918%    |    94.628%    |
=====================================================================================================================


Predict     0           1           2           3           4           5           
Actual
0           12511       126         29          72          55          0           

1           22          2189        0           44          108         8           

2           27          0           257         9           0           0           

3           132         92          3           2003        48          2           

4           88          119         1           43          4360        6           

5           0           8           0           2           0           228         


Overall Statistics : 

ACC Macro                                                         0.9846
F1 Macro                                                          0.92457
FPR Macro                                                         0.01114
Kappa                                                             0.92505
Overall ACC                                                       0.95379
PPV Macro                                                         0.92317
SOA1(Landis & Koch)                                               Almost Perfect
TPR Macro                                                         0.92653
Zero-one Loss                                                     1044

Class Statistics :

Classes                                                           0             1             2             3             4             5             
ACC(Accuracy)                                                     0.97561       0.97667       0.99695       0.98021       0.97928       0.99885       
AUC(Area under the ROC curve)                                     0.97525       0.95309       0.93783       0.93507       0.9663        0.97863       
AUCI(AUC value interpretation)                                    Excellent     Excellent     Excellent     Excellent     Excellent     Excellent     
F1(F1 score - harmonic mean of precision and sensitivity)         0.97845       0.89256       0.88165       0.89962       0.94906       0.94606       
FN(False negative/miss/type 2 error)                              282           182           36            277           257           10            
FP(False positive/type 1 error/false alarm)                       269           345           33            170           211           16            
FPR(Fall-out or false positive rate)                              0.02745       0.01706       0.00148       0.00837       0.01174       0.00072       
N(Condition negative)                                             9799          20221         22299         20312         17975         22354         
P(Condition positive or support)                                  12793         2371          293           2280          4617          238           
POP(Population)                                                   22592         22592         22592         22592         22592         22592         
PPV(Precision or positive predictive value)                       0.97895       0.86385       0.88621       0.92177       0.95384       0.93443       
TN(True negative/correct rejection)                               9530          19876         22266         20142         17764         22338         
TON(Test outcome negative)                                        9812          20058         22302         20419         18021         22348         
TOP(Test outcome positive)                                        12780         2534          290           2173          4571          244           
TP(True positive/hit)                                             12511         2189          257           2003          4360          228           
TPR(Sensitivity, recall, hit rate, or true positive rate)         0.97796       0.92324       0.87713       0.87851       0.94434       0.95798 



"""
