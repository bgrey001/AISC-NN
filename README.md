# Automatic identification system classification neural network (AISC-NN) research project

### Code base for AISC-NN: an investigation into computational methods for classifying fishing vessels to identify illegal, unreported and unregulated fishing activity

*System architecture*

AIS data is highly irregular in terms of sampling frequency, as a result a system with an innate ability to model irregular time series data has been built.
Data processing has been seperated into three channels with differing techniques for handling irregularity.
* Linear interpolation 
* Zero padding with time difference feature
* Non-linear interpolation using Shukla and Marlin's multi-time attention network (https://github.com/reml-lab/mTAN)

Two classification neural network architectures are leveraged on the three pipelines of AIS data:
* 1D convolutional neural network (CNN) with residual blocks (identity connections)
* Stacked, bidirectional gated recurrent unit network (GRU)


Although primarily a deep learning methodology is followed in this research, logistic regression support vector machines (SVMs) hve been implemented as alternatives. In order to implement these models an additional data processing method has been used - feature extraction.


[<img src="https://github.com/bgrey001/AISC-NN/blob/main/plots/figures/system_architecture_v2_wb.png" width="500" />](https://github.com/bgrey001/AISC-NN/blob/main/plots/figures/system_architecture_v2_wb.png)

*Code walkthrough*

Before attempting to run the code, the requirements must be installed using the command:
```
pip3 install -r requirements.txt
```

In order to effectively use the code, firstly, the AIS csv data must be loaded into the data/csv directory of the repository. Next, the preprocessing must be conducted using the combine\_datasets.py, this script calls heavily on the pre\_processing\_module.py script containing a utility class with various functionality. combine\_datastes.py contains all preprocessing pipelines, all can be accessed using a match case switch statement. Furthermore, the script has streamlined producing different versions of each dataset for testing performance on varying preprocessing methods. 

When the data has been processed, the functionality for models is located within scr/models. All the models are divided into separate scripts. PyTorch has been used extensively for the neural network architectures. Each PyTorch model is contained within a custom wrapper class to create consistent interface for a multitude of tasks such as training the model, saving model state dictionaries, plotting results, saving results, creating checkpoints, nonrandom initialisation and much more.

All development was conducted using the Spyder IDE with a virtual environment containing all the necessary dependencies on a local machine (CPU: Ryzen 5950X, GPU: NVIDIA RTX3090 24GB VRAM, RAM: 80GB). 

Results on the AIS test data are shown here:


| Data pipeline            | Model                       | Overall accuracy   | Macro F1-score     | Training time<br>(hh:mm:ss)<br>* per epoch |
|--------------------------|-----------------------------|--------------------|--------------------|--------------------------------------------|
| Zero padding             | GRU<br>1D CNN               | 95.379%<br>93.302% | 0.92457<br>0.87608 | 00:01:38 *<br>00:01:01 *                   |
| Linear interpolation     | GRU<br>1D CNN               | 94.01%<br>90.069%  | 0.92868<br>0.87219 | 00:01:21 *<br>00:00:51 *                   |
| Non-linear interpolation | GRU<br>1D CNN               | 94.292%<br>91.664% | 0.91183<br>0.87219 | 00:24:43 *<br>00:24:27 *                   |
| Feature extracted        | SVM <br>Logistic regression | 75.028%<br>60.65%  | 0.51715<br>0.2897  | 05:25:39<br>00:00:53                       |
