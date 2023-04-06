import math
from scipy.ndimage import gaussian_filter
from keras import backend as K

from src.trainer.base_trainer import BaseTrainer
from src.data.dataset_loader import DataSetLoader
from src.libs.utils import *


# ======================================================================================================================
class SampledInverse(BaseTrainer):
    def __init__(self, **kwargs):
        super(SampledInverse, self).__init__(config=kwargs['config'], args=kwargs['args'])

        # --------------------------------------------------------------------------------------------------------------
        if 'not_init_db' not in kwargs['args']:
            self.dataset = DataSetLoader(self.config.dataset, self.args)

    def train(self, dataLoader, n_batches, val_dataset_loader=None, nb_val=None):
        """
        :param dataLoader:
        :param n_batches:
        :param val_dataset_loader:
        :param nb_val:

        :return:
        """
        iError = []
        dError = []
        Epochs = []

        if self.args.start_epoch > 1 and os.path.isfile(
                "%s/model_epoch_%d.index" % (self.checkpoint_dir, self.args.start_epoch)):
            self.model.load_weights("%s/model_epoch_%d" % (self.checkpoint_dir, self.args.start_epoch))

        for epoch in range(self.args.start_epoch, self.args.epochs + 1):
            save_each = self.saveSpeed(epoch)
            if self.config["optimization"]["lr_scheduler"]:
                current_lr = K.get_value(self.model.optimizer.lr)
                K.set_value(self.model.optimizer.lr, self._step_decay(epoch,
                                                                      self.config["optimization"]["lr"],
                                                                      current_lr=current_lr))
            Loss = []
            batches = 0
            for input, output in dataLoader:
                if self.config["stochastic"]:
                    noise = np.random.normal(self.config["stoch_mean"], self.config["stoch_std"], size=output.shape)
                    output += noise

                if self.config["blur_std"] > 0:
                    output = gaussian_filter(output, sigma=self.config["blur_std"])

                # --- main ----
                loss = self.model.train_on_batch(x=input, y=output)

                Loss.append(loss)
                batches += 1
                if batches >= n_batches:
                    break

            # --- log ----------------------------------------------------------
            lr = K.get_value(self.model.optimizer.lr)
            loss = np.mean(np.asarray(Loss))

            self.writeSummary('lr', lr, epoch)
            self.writeSummary(self._loss_name, loss, epoch)

            logstr = "epoch : %d" % epoch
            logstr += ",\t " + self._loss_name + " = %0.15f" % loss
            logstr += ",\t lr = %0.10f" % lr
            log.info(logstr)

            # ------------------------------------------------------------------------
            if epoch % save_each == 0 or epoch == self.args.epochs:
                self.model.save_weights("%s/model_epoch_%d" % (self.checkpoint_dir, epoch))

                derror, output, prediction = self.validate(val_dataset_loader, nb_val)
                self.writeSummary("validation_mse", derror, epoch)

                dError.append(derror)
                Epochs.append(epoch)

        log.info("Training has been finished.")
        log.info(f"Model checkpoint and metadata has been saved at {self.checkpoint_dir}.")

        log.info("\n\nValidation results:\n")
        n = len(Epochs)
        for i in range(n):
            log.info(f"epoch={Epochs[i]}, dmse={dError[i]}")
        i = np.argmin(dError)
        log.info(f"\nThe best: epoch={Epochs[i]}, dmse={dError[i]}")

