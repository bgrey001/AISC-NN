"""
=================================================================================================================================
----------------------------------------> SVM TEST LOG  <------------------------------------------
=================================================================================================================================

TEST 1:
---------------------------------------->
Support vector machine:
C: 0.1 -> Regularization parameter. The strength of the regularization is inversely proportional to C. Must be strictly positive. The penalty is a squared l2 penalty.
kernel: radial basis function -> Specifies the kernel type to be used in the algorithm. 
gamma: scale -> Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’


Predict     0.0         1.0         2.0         3.0         4.0         5.0         
Actual
0.0         40283       815         118         704         774         2           

1.0         299         6139        0           592         4847        0           

2.0         1336        18          536         52          25          0           

3.0         1840        2014        12          5054        1291        0           

4.0         1645        2790        5           1055        13543       6           

5.0         55          60          1           256         1217        30     


ACC Macro                                                         0.91676
F1 Macro                                                          0.51715
FPR Macro                                                         0.05776
Kappa                                                             0.62337
Overall ACC                                                       0.75028
PPV Macro                                                         0.71189
SOA1(Landis & Koch)                                               Substantial
TPR Macro                                                         0.49292
Zero-one Loss                                                     21829

Class Statistics :

Classes                                                           0.0           1.0           2.0           3.0           4.0           5.0           
ACC(Accuracy)                                                     0.91319       0.86919       0.98207       0.91059       0.84379       0.98173       
AUC(Area under the ROC curve)                                     0.91388       0.72073       0.63545       0.73026       0.79594       0.50922       
AUCI(AUC value interpretation)                                    Excellent     Good          Fair          Good          Good          Poor          
F1(F1 score - harmonic mean of precision and sensitivity)         0.91392       0.51778       0.40621       0.56394       0.66483       0.03621       
FN(False negative/miss/type 2 error)                              2413          5738          1431          5157          5501          1589          
FP(False positive/type 1 error/false alarm)                       5175          5697          136           2659          8154          8             
FPR(Fall-out or false positive rate)                              0.11573       0.07542       0.00159       0.03444       0.11926       9e-05         
N(Condition negative)                                             44718         75537         85447         77203         68370         85795         
P(Condition positive or support)                                  42696         11877         1967          10211         19044         1619          
POP(Population)                                                   87414         87414         87414         87414         87414         87414         
PPV(Precision or positive predictive value)                       0.88616       0.51867       0.79762       0.65526       0.62419       0.78947       
TN(True negative/correct rejection)                               39543         69840         85311         74544         60216         85787         
TON(Test outcome negative)                                        41956         75578         86742         79701         65717         87376         
TOP(Test outcome positive)                                        45458         11836         672           7713          21697         38            
TP(True positive/hit)                                             40283         6139          536           5054          13543         30            
TPR(Sensitivity, recall, hit rate, or true positive rate)         0.94348       0.51688       0.2725        0.49496       0.71114       0.01853  

NOTES: Not great but didn't expect much from Sanchez pedroche et al. results

TEST 2:
---------------------------------------->
C: 0.1


Overall Statistics : 

ACC Macro                                                         0.90448
F1 Macro                                                          0.40256
FPR Macro                                                         0.06834
Kappa                                                             0.5606
Overall ACC                                                       0.71343
PPV Macro                                                         None
SOA1(Landis & Koch)                                               Moderate
TPR Macro                                                         0.40337
Zero-one Loss                                                     25050

Class Statistics :

Classes                                                           0.0           1.0           2.0           3.0           4.0           5.0           
ACC(Accuracy)                                                     0.88967       0.86384       0.97748       0.90037       0.81402       0.98148       
AUC(Area under the ROC curve)                                     0.89072       0.65938       0.49999       0.66469       0.79032       0.5           
AUCI(AUC value interpretation)                                    Very Good     Fair          Poor          Fair          Good          Poor          
F1(F1 score - harmonic mean of precision and sensitivity)         0.89234       0.43042       0.0           0.45579       0.63679       0.0           
FN(False negative/miss/type 2 error)                              2727          7380          1967          6564          4793          1619          
FP(False positive/type 1 error/false alarm)                       6917          4522          2             2145          11464         0             
FPR(Fall-out or false positive rate)                              0.15468       0.05986       2e-05         0.02778       0.16768       0.0           
N(Condition negative)                                             44718         75537         85447         77203         68370         85795         
P(Condition positive or support)                                  42696         11877         1967          10211         19044         1619          
POP(Population)                                                   87414         87414         87414         87414         87414         87414         
PPV(Precision or positive predictive value)                       0.85247       0.49861       0.0           0.62966       0.55419       None          
TN(True negative/correct rejection)                               37801         71015         85445         75058         56906         85795         
TON(Test outcome negative)                                        40528         78395         87412         81622         61699         87414         
TOP(Test outcome positive)                                        46886         9019          2             5792          25715         0             
TP(True positive/hit)                                             39969         4497          0             3647          14251         0             
TPR(Sensitivity, recall, hit rate, or true positive rate)         0.93613       0.37863       0.0           0.35716       0.74832       0.0           

"""
