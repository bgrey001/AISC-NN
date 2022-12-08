"""
============================================================================================
----------------------------------------> GRU <------------------------------------------
============================================================================================

TEST 1:
---------------------------------------->
Model: GRU_v2 -> Hyperparamters: 
Learnig rate = 0.0003 
Optimiser = AdamW 
Loss = CrossEntropyLoss 
Hidden dimension = 32 
Number of layers = 1
Batch size = 64 
Epochs = 25 
Model structure 
GRU(
  (gru): GRU(5, 32, batch_first=True)
  (fc_1): Linear(in_features=32, out_features=128, bias=True)
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
|        85.425%      |      0.635      |        85.579%        |       0.655       |      85.265%    |     0.653   |
=====================================================================================================================

Class accuracy table
=====================================================================================================================
|   Drifting longlines   |    Fixed gear    |   Pole and line   |   Purse seines   |    Trawlers    |    Trollers   |
=====================================================================================================================
|           91%          |        73%       |        84%        |       91%        |        73%     |       88%     |
=====================================================================================================================

NOTES: One layer GRU, good results however not achieving the high accuracies of the 1DCNN with ResBlocks.




TEST 2:
---------------------------------------->
Model: GRU_v1 -> Hyperparamters: 
Learnig rate = 0.0003 
Optimiser = AdamW 
Loss = CrossEntropyLoss 
Hidden dimension = 64 
Number of layers = 2
Batch size = 64 
Epochs = 5 
Model structure 
GRU(
  (gru): GRU(5, 64, num_layers=2, batch_first=True, bidirectional=True)
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
|        94.645%      |      0.140      |        93.281%        |       0.184       |      93.431%    |     0.140   |
=====================================================================================================================

Class accuracy table
=====================================================================================================================
|   Drifting longlines   |    Fixed gear    |   Pole and line   |   Purse seines   |    Trawlers    |    Trollers   |
=====================================================================================================================
|           97%          |        90%       |        66%        |       88%        |        91%     |       78%     |
=====================================================================================================================

NOTES: Encouraging results, val accuracy peaked agonisingly close to 94% (93.89%), more to come


TEST 3:
---------------------------------------->
Model: GRU_v0 -> Hyperparamters: 
Learnig rate = 0.0003 
Optimiser = AdamW 
Loss = CrossEntropyLoss 
Hidden dimension = 64 
Number of layers = 2
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
|        95.894%      |      0.110      |        94.626%        |       0.167       |      95.003%    |     0.133   |
=====================================================================================================================

Class accuracy table
=====================================================================================================================
|   Drifting longlines   |    Fixed gear    |   Pole and line   |   Purse seines   |    Trawlers    |    Trollers   |
=====================================================================================================================
|           97%          |        90%       |        91%        |       91%        |        94%     |       93%     |
=====================================================================================================================


"""
