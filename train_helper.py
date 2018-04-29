import torch

def save_ckpt(cfg, model, epoch, best_prec1, optim):
    checkpoint = {'model_param_best': model.state_dict()}
    torch.save(checkpoint, cfg.ckpt_model)

    if not isinstance(epoch, list):
        # save optim state
        optim_state = {
            'epoch': epoch,
            'best_prec1': best_prec1,
            'optim_state_best': optim.state_dict()
        }
        torch.save(optim_state, cfg.ckpt_optim)
    else:
        # save optim state
        optim_state = {
            'epoch': epoch[0],
            'epoch_pc': epoch[1],
            'epoch_pc_view': epoch[2],
            'best_prec1': best_prec1,
            'optim_state_best_pc': optim[0].state_dict(),
            'optim_state_best_all': optim[1].state_dict()
        }
        torch.save(optim_state, cfg.ckpt_optim)

