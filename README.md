# ONNX SAGENet Driver Attention-Estimation
 Python scripts for performing driver attention estimation using the SAGENet model in onnx
 
![SAGENet Driver Attention Estimation ONNX](https://github.com/ibaiGorordo/ONNX-SAGENet-Driver-Attention-Estimation/blob/main/doc/img/output.jpg)
*Original image:https://commons.wikimedia.org/wiki/File:Salt_transport_by_a_camel_train_on_Lake_Assale_(Karum)_in_Ethiopia.jpg*

# Requirements

 * Check the **requirements.txt** file. 
 * Additionally, **pafy** and **youtube-dl** are required for youtube video inference.
 
# Installation
```
pip install -r requirements.txt
pip install pafy youtube_dl=>2021.12.17
```

# ONNX model
The original models were converted to different formats (including .onnx) by [PINTO0309](https://github.com/PINTO0309), download the models from [his repository](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/257_PiCANet) and save them into the **[models](Https://github.com/ibaiGorordo/ONNX-SAGENet-Driver-Attention-Estimation/tree/main/models)** folder.


# Original Repository
The [original repository](https://github.com/anwesanpal/SAGENet_demo) also contains code for estimating the driver's attention in Pytorch.
 
# Examples

 * **Image inference**:
 
 ```
 python image_attention_estimation.py
 ```
 
  * **Video inference**:
 
 ```
 python video_attention_estimation.py
 ```
 
 * **Webcam inference**:
 
 ```
 python webcam_attention_estimation.py
 ```
 
# Inference video Example: https://youtu.be/2g-yWc_AsU0
 ![SAGENet Driver Attention Estimation ONNX](https://github.com/ibaiGorordo/ONNX-SAGENet-Driver-Attention-Estimation/blob/main/doc/img/sagenet-attention-heatmap.gif)

*Original video: https://youtu.be/MAj6y23vNuU*

# References:
* SAGENet Model: https://github.com/anwesanpal/SAGENet_demo
* SAGENet Demo Pytorch: https://github.com/ibaiGorordo/Pytorch-SAGENet-Driver-Attention-Estimation
* PINTO0309's model zoo: https://github.com/PINTO0309/PINTO_model_zoo
* PINTO0309's model conversion tool: https://github.com/PINTO0309/openvino2tensorflow
* Original paper: https://arxiv.org/abs/1911.10455
 
