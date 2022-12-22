# Automatic identification system classification neural network (AISC-NN)


[<img src="https://github.com/bgrey001/AISC-NN/blob/main/plots/figures/system_architecture_wb.png" width="30" />](https://github.com/bgrey001/AISC-NN/blob/main/plots/figures/system_architecture_wb.png)

![Image](https://github.com/bgrey001/AISC-NN/blob/main/plots/figures/system_architecture_wb.png =100x20)


![Image](https://github.com/bgrey001/AISC-NN/blob/main/plots/figures/data_pipeline_wb.png =100x20)


Welcome to the code base for AISC-NN, the workflow for taking data from csv and classifying using the models is as follows:
* Pass the source csv through src/preprocessing/combine_datasets_v2.py (this file will process the data through the pre_processing_module.py script, normalise, segment and export into a format for the AIS_loader.py class)
* Next, the script for the model desired for classification i.e GRU_v1.py has to specify the source data for model training or prediction
* Results can be examined through the print_summary() method in the model wrapper class

