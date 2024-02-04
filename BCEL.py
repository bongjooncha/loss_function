#Binary Cross-Entropy Loss

import torch.nn.functional as F

def binary_cross_entropy_loss(y_true, y_pred):
    return F.binary_cross_entropy_with_logits(y_pred, y_true)

# 사용 예시:
# loss = binary_cross_entropy_loss(y_true, y_pred)
