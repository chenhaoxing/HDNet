import torch
import torch.nn as nn
import torch.nn.functional as F

class LocalDynamics(nn.Module):
    def __init__(self, dims_in):

        super(LocalDynamics, self).__init__()
        self.linear = nn.Conv1d(dims_in*2, dims_in, kernel_size=1, stride=1, padding=0)
        self.softmax = nn.Softmax(dim = -1)
        self.k = 1
    def forward(self, x, mask):
        mask = F.interpolate(mask.detach(), size=x.size()[2:], mode='nearest')
        B, C, H, W = x.shape
        
        flattened_features_query = x * mask
        flattened_features_support = x * (1-mask)
        
        flattened_features = x.view(B, C, -1)
        masked_features = []
        for i in range(B):

            query_features = flattened_features_query[i].view(C, -1)
            support_features = flattened_features_support[i].view(C, -1)
            
            query = query_features.unsqueeze(0)
            support = support_features.unsqueeze(0)

            simi_matrix = torch.matmul(query.permute(0, 2, 1), support)
            
            weights_value, index = torch.topk(simi_matrix, dim=2, k=self.k)
            
            views = [support.shape[0]] + [1 if i != 2 else -1 for i in range(1, len(support.shape))]
            expanse = list(support.shape)
            expanse[0] = -1
            expanse[2] = -1
            index = index.view(views)
            index = index.expand(expanse)   
            weights_value = weights_value.view(views)
            weights_value = weights_value.expand(expanse)
            select_value = torch.gather(support, 2, index)
 
            select_value = select_value.view(1, C, self.k, -1)
            weights_value = weights_value.view(1, C, self.k, -1)
            weights_value = self.softmax(weights_value)
            fuse_tensor = weights_value * select_value
            fuse_tensor = torch.sum(fuse_tensor, -2)
            
            hybrid_feat = torch.cat((fuse_tensor, query), 1)
            hybrid_feat = self.linear(hybrid_feat)
            masked_features.append(hybrid_feat)

        refined_feat = torch.cat(masked_features, 0)
        refined_feat = refined_feat.view(B, C, H, W)
        
        return refined_feat * mask + x * (1-mask)

        
        
        
