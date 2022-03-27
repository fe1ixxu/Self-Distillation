import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq.modules import LayerNorm
class Switcher(nn.Module):
    def __init__(
        self,
        base_model,
    ):
        """
        base_model has to be a Linear model
        """
        super().__init__()

        self.base_model = base_model
        self.expert_num = len(base_model)

    def forward(self, x):
        """
        x: [seq_len, bz, dim] Take the first index of feature as input
        dim1: input dim
        dim2: output dim
        """
        if self.training:
            if self.expert_num == 1:
                x = self.base_model[0](x)
            else:
                expert_id = torch.randint(low=0, high=self.expert_num, size=(1,)).item()
                x = self.base_model[expert_id](x)
        else:
            if self.expert_num == 1:
                x = self.base_model[0](x)
            else:
                results = []
                for expert_id in range(self.expert_num):
                    results.append(self.base_model[expert_id](x))
                results = torch.stack(results, dim=0)
                x = results.mean(dim=0)
                # mask = torch.randint(0, self.expert_num, size=(results.size(1),), device=results.device)
                # for i in range(self.expert_num):
                #     expert_mask = mask.eq(i)
                #     results[i] *= expert_mask.unsqueeze(-1).unsqueeze(-1)
                # x = results.sum(0)
        return x

class Mapper(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        num_lang,
        hidden_dim=256, 
    ):
        super().__init__()
        self.num_lang=num_lang
        self.output_dim = output_dim
        self.W1 = nn.ModuleList([nn.Linear(input_dim, hidden_dim) for _ in range(num_lang)])
        self.W2 = nn.ModuleList([nn.Linear(hidden_dim, output_dim) for _ in range(num_lang)])

    def forward(self, x, lang_ids):
        """
        x: [seq_len, bz, dim] Take the first index of feature as input
        """

        if len(lang_ids) == 1:
            x = self.W1[lang_ids](x)
            x = self.W2[lang_ids](F.relu(x))
            return x
        else:
            y = torch.zeros(x.shape[0], x.shape[1], self.output_dim).type(x.dtype).to(x.device)
            for ind in range(self.num_lang):
                selected_id = (lang_ids == ind).nonzero().view(-1)
                if len(selected_id) > 0:
                    slice_x = x[:, selected_id, :]
                    slice_x = self.W1[ind](slice_x)
                    slice_x = self.W2[ind](F.relu(slice_x))
                    y[:, selected_id, :] = slice_x
            return y


# class Switcher(nn.Module):
#     def __init__(self, 
#         input_dim, 
#         output_dim, 
#         dict_len, 
#         layer_norm=False, 
#         hidden_dim=256, 
#         num_ls=2,
#     ):
#         super().__init__()
#         self.num_ls = num_ls
#         self.dict_len = dict_len
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         self.layer_norm = layer_norm
#         self.w1 = nn.ModuleList([nn.Linear(input_dim, hidden_dim) for _ in range(num_ls)])
#         self.w2 = nn.ModuleList([nn.Linear(hidden_dim, output_dim) for _ in range(num_ls)])
#         # self.scores = nn.ModuleList([nn.Linear(output_dim, 1) for _ in range(num_ls)])
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

#         ## for iwslt
#         group = [( lang_ids <= 3 ).type(torch.long),
#                 ( lang_ids > 3 ).type(torch.long)]

#         ## for wmt
#         # group = [( lang_ids <= 4 ).type(torch.long),
#         #         ( torch.logical_and(lang_ids>4, lang_ids<7) ).type(torch.long),
#         #         ( lang_ids >= 7 ).type(torch.long)]

#         ## for opus subset
#         # group = [( lang_ids <= 3 ).type(torch.long),
#         #         ( torch.logical_and(lang_ids>3, lang_ids<6) ).type(torch.long),
#         #         ( lang_ids >= 6 ).type(torch.long)]
#         assert len(group) == self.num_ls

#         for ind in range(self.num_ls):
#             selected_id = (group[ind]==1).nonzero().view(-1)
#             if len(selected_id) > 0:
#                 ## in case no selected id in this iteration
#                 group_x = x[:, selected_id, :]
#                 group_x = F.gelu(self.w1[ind](group_x))
#                 group_x = self.w2[ind](group_x)
#                 # scores = F.sigmoid(self.scores[ind](torch.mean(group_x,dim=0)))  #[bz, 1]
#                 # group_x = group_x * scores
#                 x[:, selected_id, :] = group_x
#         return x
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
