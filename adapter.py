import torch
import torch.nn as nn 
import torch.nn.functional as F

class LinearAdapter(nn.Module):
    def __init__(self, dim_in, dim_out, k):
        super(LinearAdapter, self).__init__()
        self.fc = nn.ModuleList([nn.Linear(dim_in, dim_out) for _ in range(k)])

    def forward(self, patch_tokens):
        for i in range(len(patch_tokens)):
            if len(patch_tokens[i].shape) == 3:
                patch_tokens[i] = self.fc[i](patch_tokens[i][:, 1:, :])
            else:
                B, C, H, W = patch_tokens[i].shape
                patch_tokens[i] = self.fc[i](patch_tokens[i].view(B, C, -1).permute(0,2,1).contiguous())

        return patch_tokens