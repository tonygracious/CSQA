import torch
import torch.nn as nn
import numpy as np

import os




# In[3]:


from Utterance_encoder import UtteranceEncoder
from Decoder import Decoder
import pickle as pkl
from KVMemoryReader import KVMemoryReader
from KG_embedding import KG_embedding
from InterUtterance_encoder import InterUtteranceEncoder





class model(nn.Module):
    def __init__(self, options):
        super(model, self).__init__()
        self.enc = UtteranceEncoder(options)
        self.kvmlookup = KVMemoryReader(options)
        self.inter_utter_encoder = InterUtteranceEncoder(options.ut_hid_size, options.ut_hid_size, options)
        self.dec = Decoder(options, options.response_vocab_size)
        self.kg_predict = KG_embedding(options)

    def forward(self, u1_text, u1_kb, u2_text, u2_kb,  u3,\
                cand_ent_embed, cand_rel_embed, cand_val_embed):
        
        #u1_text, u1_kb, u2_text, u2_kb, target_kb, target_utterance,\
        #                cand_ent_embed, cand_rel_embed, cand_val = batch
        #u1, u1_lenghts, u2, u2_lenghts, u3, u3_lenghts = batch[0], batch[1], batch[2], batch[3], batch[4], batch[5]
        #u1_text = np.flip(u1_text, axis = 1).copy()
        #u1_kb = np.flip(u1_kb, axis = 1).copy()
            
        #u2_text = np.flip(u2_text, axis = 1).copy()
        #u2_kb = np.flip(u2_kb, axis = 1).copy()
        
        
        u1_context = self.enc(u1_text, u1_kb)                        
        u2_context = self.enc( u2_text, u2_kb)
                                         
        u = torch.cat((u1_context, u2_context), dim= 1)
        q = self.inter_utter_encoder(u)

        #cand_ent_embed  = torch.FloatTensor(ent_embed[cand_ent]).cuda()
        #cand_rel_embed  = torch.FloatTensor(rel_embed[cand_rel]).cuda()
        #cand_val_embed  = torch.FloatTensor(ent_embed[cand_val]).cuda()
        cand_pred = cand_val_embed
        
        kb_context = self.kvmlookup(q,  cand_ent_embed, cand_rel_embed, cand_val_embed)
        
        
        context = torch.cat((q, kb_context), dim= 2)
        
        #u3 = torch.LongTensor(target_utterance).cuda()

        pred_text, _ = self.dec((context, u3 ) )
            
        pred_kg = self.kg_predict(kb_context, cand_pred)
            
        return(pred_text, pred_kg)
            
            
          
