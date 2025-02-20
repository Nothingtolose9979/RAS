import torch


def get_scheduler(optimizer, config):
    if config.type == "StepLR":
        step_size = config.kwargs.step_size
        step_gamma = config.kwargs.gamma
        steplr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = step_size, gamma = step_gamma)
        return steplr_scheduler
    elif config.type == "WarmupStepLR":
        step_size = config.kwargs.step_size
        step_gamma = config.kwargs.gamma
        steplr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = step_size, gamma = step_gamma)

        number_warmup_epochs = config.kwargs.number_warmup_epochs
        warmup_factor = config.kwargs.warmup_factor
        warmup_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=warmup_factor, total_iters=number_warmup_epochs)

        return torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, steplr_scheduler], milestones=[number_warmup_epochs])

    else:
        raise NotImplementedError
