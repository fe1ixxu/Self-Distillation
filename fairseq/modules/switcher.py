import torch
import torch.nn as nn
import torch.nn.functional as F
class Switcher(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.w1 = nn.Linear(input_dim, input_dim)
        self.w2 = nn.Linear(input_dim, output_dim)

    def forward(self, x, model, ind_start_without_pad=None, batch_training=True, printout=False):
        """
        x: [seq_len, bz, dim] Take the first index of feature as input
        dim1: input dim
        dim2: output dim
        """

        if ind_start_without_pad == None:
            v1 = self.w1(x[0]) #[bz, dim1]
            v2 = self.w2(x[0]) #[bz, dim2]
            # print("Not using ind!!!!!!!!")
        else:
            v = []
            for ind in range(x.shape[1]):
                v.append(x[ind_start_without_pad[ind] ,ind ,:])
            v = torch.stack(v)
            v1 = self.w1(v)
            v2 = self.w2(v)
            # print(f"ind_start_without_pad {ind_start_without_pad}")

        bz, scale = x[0].shape[0], 1 #max(v1.shape[1], v2.shape[1])
        if batch_training:
            ### Batch Training, but it is easy to get GPU OOM
            mask = v1.unsqueeze(1).permute(0,2,1) @ v2.unsqueeze(1) # [bz, dim1, dim2]
            # mask = F.softmax(mask.view(bz, -1), dim=-1).view(mask.shape)*scale
            mask = F.sigmoid(mask)
            # if printout:
            #     for i, m in enumerate(mask):
            #         print(f"mask {i}, {m}, max_value in m: {torch.topk(m.view(-1), 5)}")
            mask = mask * model.weight.permute(1,0) # [bz, dim1, dim2] 
            x = torch.einsum('abd,bdf->abf', x, mask)
            x += model.bias
        else:
            ### Sequential Training, but slow
            x = list(x.permute(1,0,2)) # [bz, seq, dim]
            for ind in range(bz):
                mask = v1[ind].unsqueeze(0).permute(1,0) @ v2[ind].unsqueeze(0) #[dim1, dim2]
                # mask = F.softmax(mask.view(-1), dim=-1).view(mask.shape)*scale
                mask = F.sigmoid(mask)
                mask = mask * model.weight.permute(1,0) #[dim1, dim2]
                x[ind] = x[ind] @ mask + model.bias
            x = torch.stack(x).permute(1,0,2) #[seq, bz, dim]
        return x