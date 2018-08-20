import torch.nn as nn
import torch
import numpy as np
import pickle as pkl



class KG_embedding(nn.Module):
    def __init__(self, options):
        super(KG_embedding, self).__init__()
        
        self.B  = nn.Linear(options.ut_hid_size, options.kb_embed_size)
        #self.softmax_pred =  nn.Softmax(dim= 1)
      
    def forward(self, q, vals):
        q = self.B(q)
        #q = torch.transpose(q,1, 0 )
        #pred = self.softmax_pred(torch.sum(vals * q, dim= 2))
        pred = torch.sum(vals * q, dim=2)
        return pred
        
   
        