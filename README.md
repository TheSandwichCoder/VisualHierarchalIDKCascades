#Creating IDK Hierarchy Based on ImageNet Dataset#

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
    - varied confidence thresholds based on precision = [0.80, 0.95]

Intermediate Classifier:
 - Base Model
    - ResNet18, ResNet34, ResNet152
 - Modifications
    - Fine Tuned on ImageNet-subtrain Dataset
    - added MLP after class prediction (hidden=256) 
    - varied confidence thresholds based on precision = [0.75, 0.80, 0.85, 0.90, 0.95]

Global Classifier:
 - Base Model
    - ResNet18, ResNet34
 - Modifications
    - varied confidence thresholds based on precision = [0.75, 0.80, 0.85, 0.90, 0.95]

Deterministic Classifier:
 - Base Model:
    - ResNet152
 - Modifications
    - None

Optimizer Algorithm (WIP):
1. Assume completely deterministic
2. Optimize Specialized Classifiers
3. Calculate Intermediate Classifier Cost
4. Optimize Entire Cascade