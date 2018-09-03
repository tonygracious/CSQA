
import torch.nn as nn
import torch
import numpy as np
import pickle as pkl


class KVMemoryReader(nn.Module):
    def __init__(self, options):
        super(KVMemoryReader, self).__init__()
        #self.embed_entity = nn.Embedding(options.kb_size, options.kb_embed_size)
        #self.embed_relations = nn.Embedding(options.kb_relations_size, options.kb_embed_size)
        #self.embed_entity.weight.requires_grad = False
        #self.embed_relations.weight.requires_grad = False
        #self.dim_red =  nn.Linear( options.kb_embed_size * 2, options.kb_embed_size)
        #self.A = nn.Linear(options.kb_embed_size,  options.dec_hid_size )
        #self.R = nn.ModuleList( [nn.Linear(options.dec_hid_size, options.dec_hid_size ), \
        #          nn.Linear(options.dec_hid_size, options.dec_hid_size )] )
        self.C = nn.Linear(options.ut_hid_size, options.kb_embed_size * 2)
        self.R = nn.Linear(options.kb_embed_size, options.ut_hid_size)
        self.softmax = nn.Softmax(dim= 1)
        #self.B = nn.Linear(options.dec_hid_size, options.kb_embed_size)
        #self.softmax_pred =  nn.Softmax(dim= 0)
        self.n_hops = 2
        self.drop = nn.Dropout(options.drp)
        
    def forward(self, q, ent, rel, value, mem_weights):
        key = torch.cat((ent, rel), dim = 2)
        q_i = q
        for i in range(self.n_hops):
            q_temp = self.C(q_i)
            att = self.softmax( torch.sum(q_temp * self.drop(key), dim = 2) ) * mem_weights
            normalizer = att.sum(dim= 1).unsqueeze(1) + 0.0000000001
            att = att / normalizer
            ot = torch.sum(att.unsqueeze(2) * value, dim = 1).unsqueeze(1)
            q_i = q_i + self.R(ot)
            #proj = self.softmax(q_i * self.A(self.dim_red(key)) ) * self.A(value)
            #proj = proj.sum(dim=1).unsqueeze(1)
            #q_i = self.R[i](proj + q)
            
        return(q_i)
    
   # def prediction(self, q ):
   #     q = q.squeeze(dim = 1)
   #     q = self.B(q)
   #     q = torch.transpose(q,1, 0 )
   #     pred = torch.transpose( self.softmax_pred( torch.mm(self.embed_entity.weight, q) ), 1, 0)
   #     return pred

   # def load_embed(self):
   #      
   #     self.embed_entity.weight = np.load('Preprocessing_files/kg_embeddings')
   #     self.embed_relations.weight = pkl.load(open('Preprocessing_files/rel.pkl', 'rb'))[0]
        