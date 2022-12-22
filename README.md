# Automatic identification system classification neural network (AISC-NN)

### Welcome to the code base for AISC-NN: an investigation into computational methods for classifying fishing vessels to identify illegal, unreported and unregulated fishing activity

Although primarily a deep learning methodology is followed in the paper and the codebase, support vector machines (SVMs) will be implemented as a non deep learning approach. The deep learning classification models implemented are as follows:
* 1D convolutional neural network (CNN) with residual blocks (identity connections)
* Stacked, bidirectional gated recurrent unit network (GRU)


[<img src="https://github.com/bgrey001/AISC-NN/blob/main/plots/figures/system_architecture_wb.png" width="500" />](https://github.com/bgrey001/AISC-NN/blob/main/plots/figures/system_architecture_wb.png)

*System architecture*


AIS data is highly irregular in terms of sampling frequency, as a result a system with an innate ability to model irregular time series data has been built.
Data processing has been seperated into three channels with differing techniques for handling the irregularity.
* Linear interpolation 
* Zero padding and time difference featurising
* Non-linear interpolation

[<img src="https://github.com/bgrey001/AISC-NN/blob/main/plots/figures/data_pipeline_wb.png" width="500" />](https://github.com/bgrey001/AISC-NN/blob/main/plots/figures/data_pipeline_wb.png)

*Data processing pipelines*

The workflow for taking data from csv and classifying using the models is as follows:
* Pass the source csv through src/preprocessing/combine_datasets_v2.py (this file will process the data through the pre_processing_module.py script, normalise, segment and export into a format for the AIS_loader.py class)
* Next, the script for the model desired for classification i.e GRU_v1.py has to specify the source data for model training or prediction
* Results can be examined through the print_summary() method in the model wrapper class



