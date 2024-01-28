import torch
from torch.nn import Module

from .common import *
from .encoders import *
from .diffusion import *
from .flow import *
import sys


class FlowVAE(Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.flow = build_latent_flow(args)
        self.diffusion = DiffusionPoint(
            net = PointwiseNet(point_dim=3, context_dim=args.latent_dim, residual=args.residual),
            var_sched = VarianceSchedule(
                num_steps=args.num_steps,
                beta_1=args.beta_1,
                beta_T=args.beta_T,
                mode=args.sched_mode
            )
        )

    def get_m2ploss(self, x0_GT, micro, writer=None, it=None):
        
        """
        Args:
            x0_gt: (B, N, 3).
            micro:  Inpui microwave, (B, d).
        """
        loss_recons = self.diffusion.get_loss(x0_GT, micro)

        if writer is not None:
            
            writer.add_scalar('train/loss_recons', loss_recons, it)


        return loss_recons


    def sample(self, micro, num_points, flexibility, truncate_std=None):
        # batch_size, _ = w.size()
        # if truncate_std is not None:
        #     w = truncated_normal_(w, mean=0, std=1, trunc_std=truncate_std)
        # # Reverse: z <- w.
        # z = self.flow(w, reverse=True).view(batch_size, -1)
        samples = self.diffusion.sample(num_points, context=micro, flexibility=flexibility)
        return samples
