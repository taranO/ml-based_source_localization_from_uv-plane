'''
Author: Olga TARAN, University of Geneva, 2023
'''

import argparse
from src.libs.utils import *
from src.libs.yaml_utils import *
from pathlib import Path

from src.trainer.base_run import BaseRun
from src.trainer.sampled_inverse_mag_phas import SampledInverseMagPhas

import warnings
warnings.simplefilter("ignore")

# ======================================================================================================================
parser = argparse.ArgumentParser(description='...')
parser.add_argument("--config_path", default="./configs/config.yaml", type=str, help="The config file path")
parser.add_argument("--type", default="train", type=str, choices=["train", "test"])
parser.add_argument('--seed', default=0,  type=int, help='seed for initializing training. ')

#parser.add_argument('--pref', default="noiseless_mag_phase_st1_seed_%d")
parser.add_argument('--pref', default="noisy_mag_phase_st1_seed_%d")

# training parameters
parser.add_argument('--epochs', default=1000, type=int, help='number of total epochs to run')
parser.add_argument('--start_epoch', default=1, type=int)

# validation & test parameters
parser.add_argument('--test_epoch', default=1000, type=int)

parser.add_argument('--is_debug', default=False, type=int)

# ======================================================================================================================
class SampledInverseRun(BaseRun):
    def __init__(self, args, config):
        super(SampledInverseRun, self).__init__(args, config)

        self.sampled_inverse = SampledInverseMagPhas(config=self.config, args=self.args)
        self.dataset = self.sampled_inverse.dataset

    def train(self):
        log.info("Dataset loading...")
        train_dataset_loader, nb_train = self.dataset.getLoader(data_subset="train")
        val_dataset_loader, nb_val = self.dataset.getLoader(data_subset="validation", batch_size=64)

        self.sampled_inverse.train(train_dataset_loader, nb_train, val_dataset_loader, nb_val)


    def test(self):
        Predicted = self.sampled_inverse.prediction(epoch=self.args.test_epoch, type="test")

        save_dir = makeDir(os.path.join(self.args.home, self.config.dataset["path"], "github_test_subset"))
        np.save(os.path.join(save_dir, "%s.npy" % self.args.pref), np.asarray(Predicted))

# ======================================================================================================================
if __name__ == "__main__":

    args = parser.parse_args()
    args.is_debug = args.is_debug if "is_debug" in args else False
    args.seed = args.seed if "seed" in args else 0
    args.pref = args.pref % (args.seed)
    args.home = str(Path.home())

    config = Config(yaml.load(open(args.config_path), Loader=yaml.FullLoader))

    Run = SampledInverseRun(args, config)
    if args.type == "train":
        Run.train()
    elif args.type == "test":
        Run.test()
    else:
        raise ValueError(f"Undefined type: {args.type}")
