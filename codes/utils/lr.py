import numpy as np
            
def adjust_learning_rate(optimizer, epoch, lr, lr_rampdown_epochs=None, eta_min=None):

    # Cosine LR rampdown from https://arxiv.org/abs/1608.03983 (but one cycle only)
    if lr_rampdown_epochs:
        # assert lr_rampdown_epochs >= epochs
        lr = eta_min + (lr - eta_min) * cosine_rampdown(epoch, lr_rampdown_epochs)

    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = lr

    return lr

def cosine_rampdown(current, rampdown_length):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    assert 0 <= current <= rampdown_length
    return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))