import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq.modules import LayerNorm
class Switcher(nn.Module):
    def __init__(self, 
        input_dim, 
        output_dim, 
        dict_len, 
        layer_norm=False, 
        hidden_dim=1024, 
        num_ls=3,
    ):
        super().__init__()
        self.num_ls = num_ls
        self.dict_len = dict_len
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer_norm = layer_norm
        self.w1 = nn.ModuleList([nn.Linear(input_dim, hidden_dim) for _ in range(num_ls)])
        self.w2 = nn.ModuleList([nn.Linear(hidden_dim, output_dim) for _ in range(num_ls)])
        if layer_norm:
            self.layer_norm_layer = LayerNorm(input_dim)

    def forward(self, x, lang_ids):
        """
        x: [seq_len, bz, dim] Take the first index of feature as input
        dim1: input dim
        dim2: output dim
        """
        if self.layer_norm:
            x = self.layer_norm_layer(x)

        lang_ids = self.dict_len - 1 - lang_ids

        ## for wmt
        group = [( lang_ids <= 4 ).type(torch.long),
                ( torch.logical_and(lang_ids>4, lang_ids<7) ).type(torch.long),
                ( lang_ids >= 7 ).type(torch.long)]

        ## for opus subset
        # group = [( lang_ids <= 3 ).type(torch.long),
        #         ( torch.logical_and(lang_ids>3, lang_ids<6) ).type(torch.long),
        #         ( lang_ids >= 6 ).type(torch.long)]

        for ind in range(self.num_ls):
            selected_id = (group[ind]==1).nonzero().view(-1)
            if len(selected_id) > 0:
                ## in case no selected id in this iteration
                group_x = x[:, selected_id, :]
                group_x = F.gelu(self.w1[ind](group_x))
                group_x = self.w2[ind](group_x)
                x[:, selected_id, :] = group_x
        return x

class Mapper(nn.Module):
    def __init__(self, 
        dict_len,
        dim=1024,
        num_ls=8,
    ):
        super().__init__()
        self.num_ls = num_ls
        self.dim = dim
        self.dict_len = dict_len
        self.w = nn.ModuleList([nn.Linear(dim, dim) for _ in range(num_ls)])

    def forward(self, x, lang_ids):
        """
        x: [seq_len, bz, dim] Take the first index of feature as input
        """

        lang_ids = self.dict_len - 1 - lang_ids
        group = []
        for ind in range(1, self.num_ls+1):
            group.append(( lang_ids == ind ).type(torch.long))

        for ind in range(self.num_ls):
            selected_id = (group[ind]==1).nonzero().view(-1)
            if len(selected_id) > 0:
                ## in case no selected id in this iteration
                x[:, selected_id, :] = self.w[ind](x[:, selected_id, :])
        return x



### Lang-Specfic module
# class Switcher(nn.Module):
#     def __init__(self, 
#         input_dim, 
#         output_dim, 
#         dict_len, 
#         layer_norm=False, 
#         hidden_dim=1024, 
#         num_ls=8,
#     ):
#         super().__init__()
#         self.num_ls = num_ls
#         self.dict_len = dict_len
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         self.layer_norm = layer_norm
#         self.w1 = nn.ModuleList([nn.Linear(input_dim, hidden_dim) for _ in range(num_ls)])
#         self.w2 = nn.ModuleList([nn.Linear(hidden_dim, output_dim) for _ in range(num_ls)])
#         if layer_norm:
#             self.layer_norm_layer = LayerNorm(input_dim)

#     def forward(self, x, lang_ids):
#         """
#         x: [seq_len, bz, dim] Take the first index of feature as input
#         dim1: input dim
#         dim2: output dim
#         """
#         if self.layer_norm:
#             x = self.layer_norm_layer(x)

#         lang_ids = self.dict_len - 1 - lang_ids
#         group = []
#         for ind in range(1, self.num_ls+1):
#             group.append(( lang_ids == ind ).type(torch.long))

#         for ind in range(self.num_ls):
#             selected_id = (group[ind]==1).nonzero().view(-1)
#             if len(selected_id) > 0:
#                 ## in case no selected id in this iteration
#                 group_x = x[:, selected_id, :]
#                 group_x = F.gelu(self.w1[ind](group_x))
#                 group_x = self.w2[ind](group_x)
#                 x[:, selected_id, :] = group_x
#         return x