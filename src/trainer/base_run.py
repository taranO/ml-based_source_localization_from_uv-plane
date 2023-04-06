from abc import ABC, abstractmethod
import yaml

from src.libs.utils import *

# ======================================================================================================================

class BaseRun(ABC):
    def __init__(self, args, config):

        # --- log configureation ----------------------------------------------------
        set_log_config(False)
        log.info("PID = %d\n" % os.getpid())

        self.args = args
        self.config = config
        self._is_debug = self.args.is_debug if "is_debug" in self.args else False

        # --------------------------------------------------------------------------------------------------------------
        if self._is_debug:
            print("\nArgs\n")
            log.info(yaml.dump(self.args, allow_unicode=True, default_flow_style=False))
            print("\nConfig\n")
            log.info(yaml.dump(self.config, allow_unicode=True, default_flow_style=False))
            log.info("\n")
        # --------------------------------------------------------------------------------------------------------------

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def test(self):
        pass


