import math
import torch
import torch.nn as nn

from functools import reduce
from operator import mul

class VisualPromptTuning(nn.Module):
    def __init__(self, model, total_d_layer, num_tokens, device):
        super(VisualPromptTuning, self).__init__()

        self.total_d_layer = total_d_layer
        self.num_tokens = num_tokens
        self.patch_size = model.visual.patch_size
        self.width = model.visual.conv1.weight.shape[0]

        val = math.sqrt(6. / float(3 * reduce(mul, self.patch_size, 1) + self.width))

        self.learnable_visual_tokens = nn.Parameter(torch.zeros(
            self.total_d_layer, self.num_tokens, self.width, device=device
        ))

        nn.init.uniform_(self.learnable_visual_tokens.data, -val, val)


    def forward(self, x=None):
        return self.learnable_visual_tokens
