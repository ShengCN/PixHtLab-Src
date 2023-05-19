from abc import ABC, abstractmethod
from collections import OrderedDict

class abs_model(ABC):
    """ Training Related Interface
    """
    @abstractmethod
    def setup_input(self, x):
        pass


    @abstractmethod
    def forward(self, x):
        pass


    @abstractmethod
    def supervise(self, input_x, y, is_training:bool)->float:
        pass


    @abstractmethod
    def get_visualize(self) -> OrderedDict:
        return {}


    """ Inference Related Interface
    """
    @abstractmethod
    def inference(self, x):
        pass


    @abstractmethod
    def batch_inference(self, x):
        pass


    """ Logging/Visualization Related Interface
    """
    @abstractmethod
    def get_logs(self):
        pass


    """ Getter & Setter
    """
    @abstractmethod
    def get_models(self) -> dict:
        """ GAN may have two models
        """
        pass


    @abstractmethod
    def get_optimizers(self) -> dict:
        """ GAN may have two optimizer
        """
        pass


    @abstractmethod
    def set_models(self, models) -> dict:
        """ GAN may have two models
        """
        pass


    @abstractmethod
    def set_optimizers(self, optimizers: dict):
        """ GAN may have two optimizer
        """
        pass
