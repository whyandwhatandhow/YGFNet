# yola_model.py

# yola_model.py
from __future__ import annotations

import torch
import torch.nn as nn
from ultralytics.nn.tasks import DetectionModel


class YOLADetectionModel(DetectionModel):
    """
    在 Ultralytics YOLOv8 检测模型上叠加：
    1) IIBlock（通过 YAML 里的 YOLAConv 进入模型）
    2) II loss（SmoothL1Consistency）
    """

    def __init__(
        self,
        cfg="/home/dell/ymd/YGFNet/ultralytics/ultralytics/models/yolav8.yaml",
        ch=3,
        nc=None,
        verbose=True,
        ii=5.0,
        ii_warmup=10,
    ):
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

        # 目标 II loss 权重与 warmup
        self.ii_target = float(ii)
        self.ii_warmup = int(ii_warmup)
        self.ii_weight = 0.0

        self.ii_criterion = nn.SmoothL1Loss(reduction="mean")

    def loss(self, batch, preds=None):
        """
        返回：
            total_loss, loss_items
        其中 loss_items 最后一项是 II loss，便于日志里显示。
        """
        det_loss, det_items = super().loss(batch, preds)

        aux_loss = det_loss.new_tensor(0.0)

        # 取第一层（YOLAConv）缓存的辅助特征
        stem = self.model[0]
        aux = getattr(stem, "aux_feats", None)

        if aux is not None:
            feat_ii, feat_ii_gma = aux
            if feat_ii.shape == feat_ii_gma.shape:
                aux_loss = self.ii_criterion(feat_ii, feat_ii_gma)

        total_loss = det_loss + self.ii_weight * aux_loss

        if not torch.is_tensor(det_items):
            det_items = torch.as_tensor(det_items, device=total_loss.device)

        # 追加一项 ii_loss，方便训练日志显示
        loss_items = torch.cat([det_items, (self.ii_weight * aux_loss).detach().view(1)])
        return total_loss, loss_items