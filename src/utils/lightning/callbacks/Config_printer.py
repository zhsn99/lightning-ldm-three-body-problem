from omegaconf import OmegaConf
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities import rank_zero_only
from os.path import join as pjoin
import time

class Config_printer(Callback):
    '''
    Prints and saves current full config at the start of fitting
    '''
    def __init__(self,config):
        super().__init__()
        self.config = config

    @rank_zero_only
    def setup(self, trainer, pl_module, stage):
        print(OmegaConf.to_yaml(self.config))
        OmegaConf.save(config=self.config, f=pjoin(self.config.instance_data_dir,f'last_full_config-{time.time()}.yaml'))
