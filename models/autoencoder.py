import torch
from torch.nn import Module
from .diffusion import *

class AutoEncoder(Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.diffusion = DiffusionPoint(
            net = PointwiseNet(point_dim=4, context_dim=args.latent_dim, residual=args.residual),
            var_sched = VarianceSchedule(
                num_steps=args.num_steps,
                beta_1=args.beta_1,
                beta_T=args.beta_T,
                mode=args.sched_mode
            )
        )



    def decode(self, waves, num_points, flexibility=0.0, ret_traj=False):       
        return self.diffusion.sample(num_points, waves, flexibility=flexibility, ret_traj=ret_traj)

    def get_loss(self, x, waves):
        loss = self.diffusion.get_loss(x, waves)   
        return loss
