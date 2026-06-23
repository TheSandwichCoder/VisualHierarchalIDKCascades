Creating IDK Hierarchy Based on ImageNet Dataset

Trained on ImageNet subtrain
Tested on ImageNetV2 (confidence threshold determined by ImageNetV2)

Base Classes: 1000
Total Groups: 73

Specialized Classifier:
 - Base Model
    - MobileNetV3_small, MobileNetV3_large
 - Modifications
    - Fine Tuned on ImageNet-subtrain Dataset
    - modified FC layer

Intermediate Classifier:
 - Base Model
    - MobileNetV3_small, MobileNetV3_large
 - Modifications
    - Fine Tuned on ImageNet-subtrain Dataset
    - added MLP after class prediction (hidden=256) 

Global Classifier:
 - Base Model
    - ResNet18, ResNet34
 - Modifications
    - None

Deterministic Classifier:
 - Base Model:
    - ResNet152
 - Modifications
    - None

