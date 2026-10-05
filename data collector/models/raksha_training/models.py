"""Explicit generic pretrained initializations and two-branch model setup."""


def build_r3d18(num_classes=2, weights_enum="R3D_18_Weights.KINETICS400_V1"):
    import torch.nn as nn
    from torchvision.models.video import r3d_18, R3D_18_Weights

    weight_table = {"R3D_18_Weights.KINETICS400_V1": R3D_18_Weights.KINETICS400_V1,
                    "KINETICS400_V1": R3D_18_Weights.KINETICS400_V1,
                    None: None}
    if weights_enum not in weight_table:
        raise ValueError(f"Unsupported R3D-18 weights enum: {weights_enum}")
    weights = weight_table[weights_enum]
    model = r3d_18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    transform = weights.transforms() if weights is not None else None
    mean = tuple(transform.mean) if transform else (0.43216, 0.394666, 0.37645)
    std = tuple(transform.std) if transform else (0.22803, 0.22145, 0.216989)
    info = {"architecture": "torchvision.models.video.r3d_18", "weights_enum": weights_enum,
            "weights_url": weights.url if weights else None,
            "weights_original_training_data": "Kinetics-400 generic action recognition" if weights else None,
            "transfer_description": "generic Kinetics-400 initialization followed by Indian task fine-tuning" if weights else "random initialization (smoke test only)",
            "normalization_mean": mean, "normalization_std": std,
            "input_layout": "B,C,T,H,W", "class_count": num_classes}
    return model, info


class FocalLoss:
    """Focal cross-entropy with gamma and optional class weights."""
    def __init__(self, gamma=2.0, weight=None):
        import torch
        self.gamma = float(gamma)
        self.weight = weight
        self._torch = torch

    def __call__(self, logits, target):
        torch = self._torch
        log_prob = torch.nn.functional.log_softmax(logits, dim=1)
        log_pt = log_prob.gather(1, target.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()
        loss = -((1 - pt) ** self.gamma) * log_pt
        if self.weight is not None:
            loss = loss * self.weight[target]
        return loss.mean()

