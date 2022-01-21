# LRP for EfficientNet-B0 of 

### Acknowledgements

This is a wrapper for the EffNet-B0 by
[LukeMelas/EfficientNet-PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch)
Thank him for training the models! I just wanted to see LRP on a more modern net. 



### Quickstart
Copy files into the path of `EfficientNet-PyTorch-master`.
Set the paths to imagenet validation data.
Run `efficientnettestcode2.py`.

### What it supports
It supports for conv layers: beta0, beta and adaptive beta. See the code for `lrp_layer2method['nn.Conv2d']`. You can code others for other layers. Uses conv-batchnorm canonization. This code is from July 2021. It works with Pytorch 1.10. Heatmaps tend to be more visually neat than with resnets.
    
