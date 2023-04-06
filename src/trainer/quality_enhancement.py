from sklearn.metrics import mean_squared_error
from scipy.ndimage import gaussian_filter

from keras import backend as K
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

from src.trainer.base_trainer import BaseTrainer
from src.data.dataset_loader import DataSetLoader
from src.models.auto_encoder import *
from src.libs.utils import *

# ======================================================================================================================
class QualityEnhancement(BaseTrainer):
    def __init__(self, pretrained_model, **kwargs):
        super(QualityEnhancement, self).__init__(config=kwargs['config'], args=kwargs['args'])

        # --------------------------------------------------------------------------------------------------------------
        self.dataset = DataSetLoader(self.config.dataset, self.args)

        self.pretrained_model = pretrained_model
        self.pretrained_model.model.trainable = False

        self.__init_ae()

        if self._is_write_summary:
            self.tensorboard.set_model(self.model)

    def __init_ae(self):
        input = Input(shape=eval(self.config["models"]["ae"]["input_size"]))
        self.AEModel = AutoEncoder(self.config["models"]["ae"], self.args)(input)

        if self._is_debug:
            self.AEModel.summary()
            self.plotModel(self.AEModel, "DenoisingAE")

        # --- main model -----------------------------------------------------------------------------------------------
        self.model = Model(inputs=input,
                           outputs=self.AEModel(input), name="Main")

        optimizer = self._getOptimizer(self.config["optimization"]["optimizer"], self.config["optimization"]["lr"])
        loss = self._getLoss(self.config["models"]["ae"]["loss"])
        self.__loss_name = self.config["models"]["ae"]["loss"]
        self.model.compile(loss=loss, optimizer=optimizer)  # , metrics=['accuracy']

    def train(self, dataLoader, n_batches, val_dataset_loader=None, nb_val=None):
        """

        :param pretrained_model:
        :param dataLoader:
        :param n_batches:
        :param val_dataset_loader:
        :param nb_val:
        :return:
        """

        # === Training =================================================================================================
        dError = []
        pError = []
        ioError = []
        Epochs = []

        if self.args.start_epoch > 1 and os.path.isfile("%s/model_epoch_%d.index" % (self.checkpoint_dir, self.args.start_epoch)):
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
                input2 = self.pretrained_model.predict(input)
                loss = self.model.train_on_batch(x=input2, y=output)

                Loss.append(loss)
                batches += 1
                if batches >= n_batches:
                    break

            # --- log ----------------------------------------------------------
            lr = K.get_value(self.model.optimizer.lr)
            loss = np.mean(np.asarray(Loss))

            self.writeSummary('lr', lr, epoch)
            self.writeSummary(self.__loss_name, loss, epoch)

            logstr = "epoch : %d" % epoch
            logstr += ",\t " + self.__loss_name + " = %0.15f" % loss
            logstr += ",\t lr = %0.10f" % lr
            log.info(logstr)

            # ------------------------------------------------------------------------
            if epoch % save_each == 0 or epoch == self.args.epochs:
                self.model.save_weights("%s/model_epoch_%d" % (self.checkpoint_dir, epoch))

                derror, perror, ioerror, output1, output, prediction = self.validate(val_dataset_loader, nb_val)
                self.writeSummary("validation_mse", derror, epoch)

                dError.append(derror)
                pError.append(perror)
                ioError.append(ioerror)
                Epochs.append(epoch)

        log.info("Training has been finished.")
        log.info(f"Model checkpoint and metadata has been saved at {self.checkpoint_dir}.")

        log.info("\n\nValidation results:\n")
        n = len(Epochs)
        for i in range(n):
            log.info(f"epoch={Epochs[i]}, pre-mse={pError[i]}, post-mse={dError[i]}, io-mse={ioError[i]}")
        i = np.argmin(dError)
        log.info(f"\nThe best: epoch={Epochs[i]}, pre-mse={pError[i]}, post-mse={dError[i]}, io-mse={ioError[i]}")


    def validate(self, val_dataset_loader, nb_val):
        dError = []
        pError = []
        ioError = []
        batches = 0
        for input, output in val_dataset_loader:
            input2 = self.pretrained_model.predict(input)
            prediction = self.model.predict(input2)

            input2 = input2.squeeze()
            output = output.squeeze()
            prediction = prediction.squeeze()
            m = output.shape[0]
            for ji in range(m):
                input2[ji] = normalize(input2[ji])
                output[ji] = normalize(output[ji])
                prediction[ji] = normalize(prediction[ji])

                dError.append(mean_squared_error(output[ji], prediction[ji]))
                pError.append(mean_squared_error(output[ji], input2[ji]))
                ioError.append(mean_squared_error(input2[ji], prediction[ji]))

            batches += 1
            if batches >= nb_val:
                break

        return np.mean(np.asarray(dError)), np.mean(np.asarray(pError)), np.mean(np.asarray(ioError)), input2, output, prediction

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
        for input, output in dataLoader:

            input2 = self.pretrained_model.predict(input)
            predicted = self.model.predict(input2)

            if len(predicted) > 1:
                predicted = predicted[0]
            Predicted.append(predicted.squeeze())

            batch += 1
            if batch >= n_batches:
                break

        log.info(f"End prediction")

        return Predicted
