"""
Prototype Generatation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def getPrototype(fts, mask, region=False):
    """
    Average the features to obtain the prototype

    Args:
        fts: input features, expect shape: B x Channel x X x Y x Z
        mask: binary mask, expect shape: B x class x X x Y x Z
        region: focus region, expect shape: B x X x Y x Z
    """
    num_classes = mask.shape[1]
    batch_size = mask.shape[0]
    if torch.is_tensor(region):
        features = [[getFeatures(fts[B,...], mask[B,C,...], region[B,...]) for B in range(batch_size)] for C in range(num_classes)]
    else:
        features = [[getFeatures(fts[B,...], mask[B,C,...]) for B in range(batch_size)] for C in range(num_classes)]
    prototypes = [torch.unsqueeze(torch.sum(torch.cat(class_fts),dim=0),0) / batch_size  for class_fts in features]
    return prototypes

def getFeatures(fts, mask, region=False):
    """
    Extract foreground and background features via masked average pooling

    Args:
        fts: input features, expect shape: C x X' x Y' x Z'
        mask: binary mask, expect shape: X x Y x Z
    """
    fts = torch.unsqueeze(fts, 0)
    if torch.is_tensor(region):
        mask = torch.unsqueeze(mask * region, 0)
        masked_fts = torch.sum(fts * mask[None, ...], dim=(2,3,4))
    else:
        mask = torch.unsqueeze(mask, 0)
        masked_fts = torch.sum(fts * mask[None, ...], dim=(2, 3, 4)) \
            / (mask[None, ...].sum(dim=(2, 3, 4)) + 1e-5) # 1 x C
    return masked_fts

def calDist(fts, prototype, scaler=1.):
    """
    Calculate the distance between features and prototypes

    Args:
        fts: input features
            expect shape: N x C x X x Y x Z
        prototype: prototype of one semantic class
            expect shape: 1 x C
    """
    dist = F.cosine_similarity(fts, prototype[..., None, None, None], dim=1) * scaler
    return dist