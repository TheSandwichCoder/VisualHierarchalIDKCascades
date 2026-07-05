from empirical_cascade_optimizer.empirical_cascade_builder import *
from model_trainer.get_skipper_dataset import *
from model_trainer.train_rf_skipper import *
from model_trainer.train_mlp_skipper import *

if __name__ == "__main__":
    benchmark_empirical_cascade(
        mlp_skipper_path="models/skipper/mlp_skipper.pth"
    )