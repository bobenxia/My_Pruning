import torch
import math

def sgd_optimizer(engine, cfg, model, no_l2_keywords, use_nesterov, keyword_to_lr_mult):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.base_lr
        weight_decay = cfg.weight_decay
        if "bias" in key or "bn" in key or "BN" in key:
            # lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.weight_decay_bias
            engine.echo('set weight_decay_bias={} for {}'.format(weight_decay, key))
        for kw in no_l2_keywords:
            if kw in key:
                weight_decay = 0
                engine.echo('NOTICE! weight decay = 0 for {} because {} in {}'.format(key, kw, key))
                break
        if 'bias' in key:
            apply_lr = 2 * lr
        else:
            apply_lr = lr
        if keyword_to_lr_mult is not None:
            for keyword, mult in keyword_to_lr_mult.items():
                if keyword in key:
                    apply_lr *= mult
                    engine.echo('multiply lr of {} by {}'.format(key, mult))
                    break
        params += [{"params": [value], "lr": apply_lr, "weight_decay": weight_decay}]
    # optimizer = torch.optim.Adam(params, lr)
    optimizer = torch.optim.SGD(params, lr, momentum=cfg.momentum, nesterov=use_nesterov)
    return optimizer

def get_optimizer(engine, cfg, model, no_l2_keywords, use_nesterov=False, keyword_to_lr_mult=None):
    return sgd_optimizer(engine, cfg, model, no_l2_keywords, use_nesterov=use_nesterov, keyword_to_lr_mult=keyword_to_lr_mult)


class CosineAnnealingExtendLR(torch.optim.lr_scheduler._LRScheduler):
    r"""Set the learning rate of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial lr and
    :math:`T_{cur}` is the number of epochs since the last restart in SGDR:

    .. math::
        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 +
        \cos(\frac{T_{cur}}{T_{max}}\pi))

    When last_epoch=-1, sets initial lr as lr.

    It has been proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_. Note that this only
    implements the cosine annealing part of SGDR, and not the restarts.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_max (int): Maximum number of iterations.
        eta_min (float): Minimum learning rate. Default: 0.
        last_epoch (int): The index of last epoch. Default: -1.

    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """

    def __init__(self, optimizer, T_cosine_max, eta_min=0, last_epoch=-1):
        self.eta_min = eta_min
        self.T_cosine_max = T_cosine_max
        super(CosineAnnealingExtendLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch <= self.T_cosine_max:
            return [self.eta_min + (base_lr - self.eta_min) *
                (1 + math.cos(math.pi * self.last_epoch / self.T_cosine_max)) / 2
                for base_lr in self.base_lrs]
        else:
            return [self.eta_min
                for _ in self.base_lrs]



if __name__ == "__main__":
    import torchvision
    
    model = torchvision.models.resnet50(pretrained=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.04, momentum=0.9, weight_decay=1e-4)

    lr_epoch_boundaries=[200, 400]
    scheduler = CosineAnnealingExtendLR(optimizer=optimizer, T_cosine_max=lr_epoch_boundaries[0])