import sys
import os

path = os.path.join(os.getcwd(), "external/")
sys.path.append(path)

import torch
from external.fast_reid.fastreid.config import get_cfg
from external.fast_reid import modeling
from external.fast_reid.fastreid.utils.checkpoint import Checkpointer


def setup_cfg(config_file, opts):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(opts)
    cfg.MODEL.BACKBONE.PRETRAIN = False

    cfg.freeze()

    return cfg


class FastReIDInferenceAdaptor(torch.nn.Module):
    def __init__(self, weights_path):
        super().__init__()
        config_file = "external/fast_reid/configs/Base-SBS.yml"
        self.cfg = setup_cfg(config_file, ['MODEL.WEIGHTS', weights_path])

        self.model = modeling.build_model(self.cfg)
        Checkpointer(self.model).load(weights_path)
        self.model = self.model.eval()

        self.pH, self.pW = self.cfg.INPUT.SIZE_TEST
        print(self.pH, self.pW)

    def forward(self, batch):
        with torch.no_grad():
            return self.model(batch)

# m = FastReIDInferenceAdaptor("external/weights/mot17_sbs_S50.pth")
# m.cuda()

