import math
from sklearn.metrics import mean_squared_error
import tensorflow as tf

from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model

from src.libs.utils import *
from src.losses.distance_correlation import DistanceCorrelation
from src.losses.ssim import SSIM
from src.losses.jaccard_distance import JaccardDistance
from src.losses.dice_coefficient import DiceCoefficient
from src.losses.dice_bce import DiceBCE

# ======================================================================================================================

class BaseTrainer(Model):
    def __init__(self, config, args):
        super(BaseTrainer, self).__init__()

        self.args = args
        self.config = config

        self._is_debug = self.args.is_debug if "is_debug" in self.args else False
        self._is_write_summary = self.config["is_write_summary"]
        # --------------------------------------------------------------------------------------------------------------

        self.checkpoint_dir = os.path.join(self.config["checkpoints_dir"], self.args.pref)
        makeDir(self.checkpoint_dir)
        log.info("Checkpoint dir: %s" % self.checkpoint_dir)

        self.results_dir = os.path.join(self.config["results_dir"], self.args.pref)
        makeDir(self.results_dir)
        log.info("Results dir: %s" % self.results_dir)
        # --------------------------------------------------------------------------------------------------------------
        self.tensorboard_dir = os.path.join(self.config["summary_writer"], self.args.pref)
        self.writer = None
        if self.config["is_write_summary"]:
            makeDir(self.tensorboard_dir)
            log.info("Tensorboard dir: %s" % self.tensorboard_dir)
            
            self.tensorboard = tf.keras.callbacks.TensorBoard(self.tensorboard_dir, update_freq='epoch', write_graph=True,
                                                              write_grads=True)
            self.writer = tf.summary.create_file_writer(self.tensorboard_dir)

    def writeSummary(self, name, value, step):
        if self.writer is not None:
            with self.writer.as_default():
                tf.summary.scalar(name, value, step)

    def plotModel(self, model, name):
        plot_model(model,
                   to_file='%s/%s.png' % (self.results_dir, name),
                   show_shapes=True,
                   show_layer_names=True,
                   expand_nested=True)

    def saveSpeed(self, epoch):
        """Saving speed during the model training"""
        save_each = 1
        if epoch <= 10:
            save_each = 1
        elif epoch <= 100:
            save_each = 10
        elif epoch <= 1000:
            save_each = 100
        elif epoch <= 10000:
            save_each = 500

        return save_each

    def _step_decay(self, epoch, initial_lrate, current_lr=0.0):
        epochs_drop = 10.0
        if self.config["optimization"]["scheduler_type"] == "exp":
            decay_rate = 0.96
            lrate = initial_lrate * math.pow(decay_rate, epoch / epochs_drop)
        elif self.config["optimization"]["scheduler_type"] == "custom":
            if epoch == 100:
                lrate = initial_lrate - initial_lrate / 2
            else:
                lrate = current_lr
        else:
            alpha = 0
            cosine_decay = 0.5 * (1 + math.cos(math.pi * epoch / epochs_drop))
            decayed = (1 - alpha) * cosine_decay + alpha
            lrate = initial_lrate * decayed

        return lrate

    def _getLoss(self, loss):
        if loss == "binary_crossentropy":
            return tf.keras.losses.binary_crossentropy
        elif loss == "categorical_crossentropy":
            return tf.keras.losses.categorical_crossentropy
        elif loss == "sparse_categorical_crossentropy":
            return tf.keras.losses.sparse_categorical_crossentropy
        elif loss == "mse":
            return tf.keras.losses.mean_squared_error
        elif loss == "l1":
            return tf.keras.losses.mean_absolute_error
        elif loss == "dcorr":
            return DistanceCorrelation()
        elif loss == "ssim":
            return SSIM()
        elif loss == "jaccard_distance":
            return  JaccardDistance()
        elif loss == "dice_coefficient":
            return DiceCoefficient()
        elif loss == "dice_bce":
            return DiceBCE()
        else:
            raise ValueError('BaseTrainer._getLoss(.): undefined "loss"')

    def _getOptimizer(self, optimizer, lr):
        if optimizer == "adam":
            return tf.keras.optimizers.Adam(learning_rate=lr)
        elif optimizer == "rmsprop":
            return tf.keras.optimizers.RMSprop(learning_rate=lr)
        else:
            raise ValueError('\n\n BaseTrainer._getOptimizer(.): undefined "optimizer" \n\n')

    def _define_fig_size(self, start_epoch=1):
        N = 0
        for epoch in range(start_epoch, self.args.epochs + 1):
            save_each = self.saveSpeed(epoch)
            if epoch % save_each == 0 or epoch == self.args.epochs:
                N += 1

        dr = int(math.sqrt(N))
        dc = int(math.ceil(N / dr))

        return dr, dc

    def load(self, epoch, path):
        try:
            self.model.load_weights("%s/model_epoch_%d" % (path, epoch))
        except OSError:
            raise ValueError("Model does not exist: %s/model_epoch_%d" % (path, epoch))

    def predict(self, input):
        return self.model.predict(input)


    def prediction(self, epoch, n_batches=None, type="validation"):

        if os.path.isfile("%s/model_epoch_%d.index" % (self.checkpoint_dir, epoch)):
            print("%s/model_epoch_%d" % (self.checkpoint_dir, epoch))
            self.model.load_weights("%s/model_epoch_%d" % (self.checkpoint_dir, epoch))
        else:
            raise ValueError("Model is missing: %s/model_epoch_%d.index" % (self.checkpoint_dir, epoch))

        batch = 0
        dataLoader, nb = self.dataset.getLoader(data_subset=type, batch_size=1, is_shuffle=False)
        n_batches = nb if n_batches is None else n_batches

        Predicted = []
        log.info(f"Start prediction")
        for input, _ in dataLoader:
            predicted = self.model.predict(input)

            if len(predicted) > 1:
                predicted = predicted[0]
            Predicted.append(predicted.squeeze())

            batch += 1
            if batch >= n_batches:
                break
        log.info(f"End prediction")

        return Predicted

    def validate(self, val_dataset_loader, nb_val):
        dError = []
        batches = 0
        for input, output in val_dataset_loader:
            prediction = self.model.predict(input).squeeze()
            output = output.squeeze()

            m = output.shape[0]
            for ji in range(m):
                output[ji] = normalize(output[ji])
                prediction[ji] = normalize(prediction[ji])
                dError.append(mean_squared_error(output[ji], prediction[ji]))

            batches += 1
            if batches >= nb_val:
                break

        return np.mean(np.asarray(dError)), output, prediction
