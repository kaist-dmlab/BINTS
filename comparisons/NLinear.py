import torch
import torch.nn as nn

class Config:
    def __init__(self, config_dict):
        self._config_dict = config_dict

    def __getattr__(self, item):
        return self._config_dict[item]

class Model(nn.Module):
    """
    Normalization-Linear
    """
    def __init__(self, opts):
        super(Model, self).__init__()
        configs = Config(opts)       
        self.seq_len = configs.seq_day * configs.cycle
        self.pred_len = configs.pred_day * configs.cycle
        self.enc_in = (configs.node_feature_dim * configs.num_node) + ( configs.num_node * configs.num_node )
        
        # Use this line if you want to visualize the weights
        # self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        self.channels = self.enc_in // 2
        self.individual = True
        if self.individual:
            self.Linear = nn.ModuleList()
            for i in range(self.channels):
                self.Linear.append(nn.Linear(self.seq_len,self.pred_len))
        else:
            self.Linear = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seq_last = x[:,-1:,:].detach()
        x = x - seq_last
        if self.individual:
            output = torch.zeros([x.size(0),self.pred_len,x.size(2)],dtype=x.dtype).to(x.device)
            for i in range(self.channels):
                output[:,:,i] = self.Linear[i](x[:,:,i])
            x = output
        else:
            x = self.Linear(x.permute(0,2,1)).permute(0,2,1)
        x = x + seq_last
        return x # [Batch, Output length, Channel]