from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from src.trainer.sampled_inverse import SampledInverse
from src.models.inverse_re_im import *

# ======================================================================================================================

class SampledInverseReIm(SampledInverse):
    def __init__(self, **kwargs):
        super(SampledInverseReIm, self).__init__(config=kwargs['config'], args=kwargs['args'])

        self.__init()

    def __init(self):
        size = eval(self.config["models"]["inverse"]["output_size"])[0]
        input1 = Input(shape=eval(self.config["models"]["inverse"]["input_size"]))

        self.InverseModel = InverseReIm(self.config["models"]["inverse"], self.args, size=size)(input1)

        if self._is_debug:
            self.InverseModel.summary()
            self.plotModel(self.InverseModel, self.InverseModel.name)

        # --- main model -----------------------------------------------------------------------------------------------
        self.model = Model(inputs=input1,
                           outputs=self.InverseModel(input1), name="Main")

        optimizer = self._getOptimizer(self.config["optimization"]["optimizer"], self.config["optimization"]["lr"])
        loss = self._getLoss(self.config["models"]["inverse"]["loss"])
        self._loss_name = self.config["models"]["inverse"]["loss"]
        self.model.compile(loss=loss, optimizer=optimizer)

        if self._is_write_summary:
            self.tensorboard.set_model(self.model)
