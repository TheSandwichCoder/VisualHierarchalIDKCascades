import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
Cascade Setup:
resnet18 c:0.009130139080917973 cost:0.0057900139100733215 idk-prob:0.3508 recall:0.6492 precision:0.75
resnet18_logit_router c:0.009521451456227625 cost:0.005893903880089056 idk-prob:0.23460000000000003 recall:0.7654 precision:0.7500653253200941
resnet34_logit_router c:0.01546269214040311 cost:0.011174382630025502 idk-prob:0.09799999999999998 recall:0.902 precision:0.75
resnet152 c:0.043758260309975594 cost:0.043758260309975594 idk-prob:0.0 recall:1.0 precision:0.711

{'min_precision': 0.75, 'accuracy': 0.5731, 'average_runtime': 0.017884820430725812, 'correct': 5731, 'total': 10000}
{'min_precision': 0.8, 'accuracy': 0.6157, 'average_runtime': 0.030045030650508123, 'correct': 6157, 'total': 10000}
{'min_precision': 0.85, 'accuracy': 0.6898, 'average_runtime': 0.04321998993996531, 'correct': 6898, 'total': 10000}
{'min_precision': 0.9, 'accuracy': 0.7037, 'average_runtime': 0.052964112740918064, 'correct': 7037, 'total': 10000}
{'min_precision': 0.95, 'accuracy': 0.7092, 'average_runtime': 0.06587598415991525, 'correct': 7092, 'total': 10000}

"""