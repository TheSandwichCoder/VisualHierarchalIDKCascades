import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
Cascade Setup:
resnet18 c:0.009130139080917973 cost:0.0057900139100733215 idk-prob:0.3508 recall:0.6492 precision:0.75
resnet18_logit_router c:0.009521451456227625 cost:0.005893903880089056 idk-prob:0.23460000000000003 recall:0.7654 precision:0.7500653253200941
resnet34_logit_router c:0.01546269214040311 cost:0.011174382630025502 idk-prob:0.09799999999999998 recall:0.902 precision:0.75
resnet152 c:0.043758260309975594 cost:0.043758260309975594 idk-prob:0.0 recall:1.0 precision:0.711

Resnet + Logit Router Identifier:
{'min_precision': 0.75, 'accuracy': 0.5731, 'average_runtime': 0.017884820430725812, 'correct': 5731, 'total': 10000}
{'min_precision': 0.8, 'accuracy': 0.6157, 'average_runtime': 0.030045030650508123, 'correct': 6157, 'total': 10000}
{'min_precision': 0.85, 'accuracy': 0.6898, 'average_runtime': 0.04321998993996531, 'correct': 6898, 'total': 10000}
{'min_precision': 0.9, 'accuracy': 0.7037, 'average_runtime': 0.052964112740918064, 'correct': 7037, 'total': 10000}
{'min_precision': 0.95, 'accuracy': 0.7092, 'average_runtime': 0.06587598415991525, 'correct': 7092, 'total': 10000}

MobileNet Identifier:
{'min_precision': 0.75, 'accuracy': 0.4869, 'average_runtime': 0.022439317870954982, 'correct': 4869, 'total': 10000}
{'min_precision': 0.8, 'accuracy': 0.5325, 'average_runtime': 0.028964980739704334, 'correct': 5325, 'total': 10000}
{'min_precision': 0.85, 'accuracy': 0.623, 'average_runtime': 0.035370395080000165, 'correct': 6230, 'total': 10000}
{'min_precision': 0.9, 'accuracy': 0.649, 'average_runtime': 0.0504827225303161, 'correct': 6490, 'total': 10000}
{'min_precision': 0.95, 'accuracy': 0.6653, 'average_runtime': 0.057720866779924836, 'correct': 6653, 'total': 10000}

MobileNet + MLP head Identifier
{'min_precision': 0.75, 'accuracy': 0.4496, 'average_runtime': 0.02144019855982624, 'correct': 4496, 'total': 10000}
{'min_precision': 0.8, 'accuracy': 0.49, 'average_runtime': 0.028365807860530914, 'correct': 4900, 'total': 10000}
{'min_precision': 0.85, 'accuracy': 0.5553, 'average_runtime': 0.03887959836043883, 'correct': 5553, 'total': 10000}
{'min_precision': 0.9, 'accuracy': 0.605, 'average_runtime': 0.0499492357987212, 'correct': 6050, 'total': 10000}
{'min_precision': 0.95, 'accuracy': 0.6497, 'average_runtime': 0.06283279696934624, 'correct': 6497, 'total': 10000}

Proper Identifier
{'accuracy': 0.544, 'average_runtime': 0.03327682137048105, 'correct': 5440, 'total': 10000, 'expected_cost': 0.01291060629929683}
"""