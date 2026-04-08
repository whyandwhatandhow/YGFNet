# yola_trainer.py
from __future__ import annotations

from copy import copy
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.cfg import DEFAULT_CFG
from ultralytics.utils import RANK

from yola_model import YOLADetectionModel


def update_ii_weight(trainer):
    model = trainer.model
    ii_target = float(getattr(trainer, "ii_target", 5.0))
    ii_warmup = int(getattr(trainer, "ii_warmup", 10))

    if ii_warmup <= 0:
        model.ii_weight = ii_target
        return

    if trainer.epoch < ii_warmup:
        model.ii_weight = ii_target * (trainer.epoch + 1) / ii_warmup
    else:
        model.ii_weight = ii_target


class YOLATrainer(DetectionTrainer):
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        overrides = dict(overrides or {})

        # 先拿走自定义参数，避免进入 Ultralytics 的 cfg 校验
        self.ii_target = float(overrides.pop("ii", 5.0))
        self.ii_warmup = int(overrides.pop("ii_warmup", 10))

        super().__init__(cfg, overrides, _callbacks)

        self.add_callback("on_train_epoch_start", update_ii_weight)

    def get_model(self, cfg=None, weights=None, verbose=True):
        cfg = cfg or self.args.model
        model = YOLADetectionModel(
            cfg=cfg,
            nc=self.data["nc"],
            ch=self.data["channels"],
            verbose=verbose and RANK == -1,
            ii=self.ii_target,
            ii_warmup=self.ii_warmup,
        )
        if weights:
            model.load(weights, verbose=verbose)
        return model

    def get_validator(self):
        self.loss_names = ("box_loss", "cls_loss", "dfl_loss", "ii_loss")
        return DetectionValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks,
        )

    def label_loss_items(self, loss_items=None, prefix="train"):
        names = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is None:
            return names
        return {k: v for k, v in zip(names, loss_items)}