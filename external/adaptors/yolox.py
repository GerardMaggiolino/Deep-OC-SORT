import sys

import torch 
import torch.nn as nn

from yolox.models import YOLOPAFPN, YOLOX, YOLOXHead
from yolox.utils import postprocess


class PostModel(nn.Module): 
    def __init__(self, model): 
        super().__init__()
        self.exp = Exp()
        self.model = model

    def forward(self, batch): 
        """
        Returns Nx5, (x1, y1, x2, y2, conf)
        """
        raw = self.model(batch)
        pred = postprocess(raw, self.exp.num_classes, self.exp.test_conf, self.exp.nmsthre)[0]
        return torch.cat((pred[:, :4], (pred[:, 4] * pred[:, 5])[:, None]), dim=1)


def get_model(path): 
    model = Exp().get_model()
    ckpt = torch.load(path)
    model.load_state_dict(ckpt["model"])
    model = PostModel(model)
    model.cuda()
    model.eval()
    return model


class Exp:
    def __init__(self):
        # ---------------- model config ---------------- #
        self.num_classes = 80
        self.depth = 1.00
        self.width = 1.00
        # ---------------- dataloader config ---------------- #
        # set worker to 4 for shorter dataloader init time
        self.data_num_workers = 4
        self.input_size = (640, 640)
        self.random_size = (14, 26)
        self.train_ann = "instances_train2017.json"
        self.val_ann = "instances_val2017.json"
        # --------------- transform config ----------------- #
        self.degrees = 10.0
        self.translate = 0.1
        self.scale = (0.1, 2)
        self.mscale = (0.8, 1.6)
        self.shear = 2.0
        self.perspective = 0.0
        self.enable_mixup = True
        # --------------  training config --------------------- #
        self.warmup_epochs = 5
        self.max_epoch = 300
        self.warmup_lr = 0
        self.basic_lr_per_img = 0.01 / 64.0
        self.scheduler = "yoloxwarmcos"
        self.no_aug_epochs = 15
        self.min_lr_ratio = 0.05
        self.ema = True
        self.weight_decay = 5e-4
        self.momentum = 0.9
        self.print_interval = 10
        self.eval_interval = 10
        # -----------------  testing config ------------------ #
        self.test_conf = 0.001
        self.nmsthre = 0.65

        self.num_classes = 1
        self.depth = 1.33
        self.width = 1.25
        self.input_size = (800, 1440)
        self.test_size = (800, 1440)
        self.random_size = (18, 32)
        self.max_epoch = 80
        self.print_interval = 20
        self.eval_interval = 5
        self.test_conf = 0.1
        self.nmsthre = 0.7
        self.no_aug_epochs = 10
        self.basic_lr_per_img = 0.001 / 64.0
        self.warmup_epochs = 1

    def get_model(self):
        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if getattr(self, "model", None) is None:
            in_channels = [256, 512, 1024]
            backbone = YOLOPAFPN(self.depth, self.width, in_channels=in_channels)
            head = YOLOXHead(self.num_classes, self.width, in_channels=in_channels)
            self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        return self.model