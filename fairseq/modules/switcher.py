import torch
import torch.nn as nn
import torch.nn.functional as F

class Switcher(nn.Module):
    def __init__(self, input_dim, output_dim, dict_info):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dict_len, self.lang_len = dict_info
        self.w1 = nn.Linear(input_dim, input_dim)
        self.w2 = nn.Linear(input_dim, output_dim)
        self.lang_emb = nn.Embedding(self.lang_len, input_dim)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x, model, lang_ids, alpha=1):
        """
        x: [seq_len, bz, dim] Take the first index of feature as input
        lang_ids: [bz], sequence of language ids for each sentence
        dim1: input dim
        dim2: output dim
        """
        lang_ids = self.dict_len - lang_ids
        bz, scale = x[0].shape[0], 1
        v = self.lang_emb(lang_ids) #[bz, dim1]
        mask = self.w1(v).unsqueeze(1).permute(0,2,1) @ self.w2(v).unsqueeze(1) # [bz, dim1, dim2]\
        v = self.dropout(v)
        # mask = F.softmax(mask.view(bz, -1), dim=-1).view(mask.shape)*scale
        mask = F.sigmoid(alpha * mask)
        # if printout:
            # for i, m in enumerate(mask):
            #     print(f"mask {i}, {m}, max_value in m: {torch.topk(m.view(-1), 5)}")
        mask = mask * model.weight.permute(1,0) # [bz, dim1, dim2] 
        x = torch.einsum('abd,bdf->abf', x, mask)
        x += model.bias
        return x

# class Switcher(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super().__init__()
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         self.w1 = nn.Linear(input_dim, output_dim)
#         self.w2 = nn.Linear(input_dim, output_dim)

#     def forward(self, x, model, ind_start_without_pad=None, alpha=1):
#         """
#         x: [seq_len, bz, dim] Take the first index of feature as input
#         dim1: input dim
#         dim2: output dim
#         """

#         if ind_start_without_pad == None:
#             v1 = self.w1(x[0]) #[bz, dim2]
#             v2 = self.w2(x[0]) #[bz, dim2]
#         else:
#             v = []
#             for ind in range(x.shape[1]):
#                 v.append(x[ind_start_without_pad[ind] ,ind ,:])
#             v = torch.stack(v)
#             v1 = self.w1(v)
#             v2 = self.w2(v)

#         bz, scale = x[0].shape[0], 1
#         mask = v1.unsqueeze(1).permute(0,2,1) @ v2.unsqueeze(1) # [bz, dim2, dim2]
#         mask = F.sigmoid(alpha * mask)
#         # for i, m in enumerate(mask):
#         #     print(f"mask {i}, {m}, max_value in m: {torch.topk(m.view(-1), 5)}")
#         # exit(0)
#         mask = mask @ model.weight # [bz, dim2, dim1] 
#         x = torch.einsum('abd,bdf->abf', x, mask.permute(0,2,1))
#         x += model.bias
#         return x

# if ind_start_without_pad == None:
#     v1 = self.w1(x[0]) #[bz, dim1]
#     v2 = self.w2(x[0]) #[bz, dim2]
# else:
#     v = []
#     for ind in range(x.shape[1]):
#         v.append(x[ind_start_without_pad[ind] ,ind ,:])
#     v = torch.stack(v)
#     v1 = self.w1(v)
#     v2 = self.w2(v)
    # print(f"ind_start_without_pad {ind_start_without_pad}")

    #max(v1.shape[1], v2.shape[1])
### Batch Training, but it is easy to get GPU OOM


