import numpy as np
import os
import logging as log
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from .dataset_base import DatasetBase

# ======================================================================================================================

class DataSetLoader(DatasetBase):

    def __init__(self, config, args):
        super(DataSetLoader, self).__init__()

        self._config = config
        self._is_debug = args.is_debug if "is_debug" in args else False
        self._data_size = self._config["data_size"]
        self._seed = args.seed if "seed" in args else -1
        self._batch_size = args.batch_size if "batch_size" in args else self._config["batch_size"]
        self._path = os.path.join(args.home, self._config["path"])

        self._initDataSets()

    def __inputNormalization(self, input):
        for i in range(len(self._config["z"])):
            if self._config["z"][i] != 0:
                input[:, i] *= self._config["z"][i]

            if "clip" in self._config:
                if len(self._config["clip"][i]) >= 2:
                    input[:, i] = np.clip(input[:, i], self._config["clip"][i][0], self._config["clip"][i][1])

        return input

    def _initDataSets(self):
        '''
        Read dataset from file and make a data_10k loader
        :return: ImageDataGenerator
        '''
        h, w = eval(self._config["data_size"])
        # --------------------------------------------------------------------------------------------------------------
        if self._is_debug:
            log.info("Input loading...")
        input = self._loadNpy(os.path.join(self._path, self._config["input"])).reshape((-1, 2, self._config["t"], self._config["s"]))
        if self._is_debug:
            print(input.shape)

        if self._is_debug:
            log.info("Output loading...")
        output = self._loadNpy(os.path.join(self._path, self._config["output"])).reshape((-1, h, w, 1))
        if self._is_debug:
            print(output.shape)

        if self._config["output_dtype"] == "uint8":
                output = output.astype(np.uint8)

        # --------------------------------------------------------------------------------------------------------------
        input = self.__inputNormalization(input)

        train_idx, test_idx, validation_idx = self._trainValTestSplit(input.shape[0],
                                                                      train_ratio=self._config["train_ratio"],
                                                                      test_ratio=self._config["test_ratio"])

        self.__train_input = input[train_idx]
        self.__train_output = output[train_idx] if len(output) else []

        self.__test_input = input[test_idx]
        self.__test_output = output[test_idx] if len(output) else []

        self.__validation_input = input[validation_idx]
        self.__validation_output = output[validation_idx] if len(output) else []


    def getLoader(self, data_subset="train", batch_size=None, is_shuffle=True):
        '''
        Create the data loader

        :param data_subset: str
        :return: ImageDataGenerator
        '''

        DataGenerator = ImageDataGenerator(samplewise_center=False,
                                           samplewise_std_normalization=False)
        if data_subset == "train":
            n_batches = self.__train_input.shape[0] // self._batch_size + 1
            n_batches = n_batches if n_batches < self.__train_input.shape[0] else self.__train_input.shape[0]

            DataLoader = DataGenerator.flow(x=self.__train_input,
                                            y=self.__train_output,
                                            batch_size=self._batch_size,
                                            shuffle=is_shuffle)
            samples_shape = self.__train_input.shape

        elif data_subset == "test":
            if batch_size is not None:
                n_batches = self.__test_input.shape[0] // batch_size + 1
            else:
                batch_size = 1
                n_batches = self.__test_input.shape[0]
            n_batches = n_batches if n_batches < self.__test_input.shape[0] else self.__test_input.shape[0]

            DataLoader = DataGenerator.flow(x=self.__test_input,
                                            y=self.__test_output,
                                            batch_size=batch_size,
                                            shuffle=False)
            samples_shape = self.__test_input.shape

        elif data_subset == "validation":
            if batch_size is not None:
                n_batches = self.__validation_input.shape[0] // batch_size + 1
            else:
                batch_size = 1
                n_batches = self.__validation_input.shape[0]
            n_batches = n_batches if n_batches < self.__validation_input.shape[0] else self.__validation_input.shape[0]

            DataLoader = DataGenerator.flow(x=self.__validation_input,
                                            y=self.__validation_output,
                                            batch_size=batch_size,
                                            shuffle=is_shuffle)
            samples_shape = self.__validation_input.shape
        else:
            raise ValueError('DataSetLoader.getLoader(.): undefined "data_subset"')

        log.info(f"Dataset subset: {data_subset}; samples: {samples_shape}; n_batches: {n_batches}")

        return DataLoader, n_batches
