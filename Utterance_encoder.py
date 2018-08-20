import torch.nn as nn, torch, copy, tqdm, math
from torch.autograd import Variable
import torch.nn.functional as F
import pickle as pkl
import numpy as np
#use_cuda = torch.cuda.is_available()




# encode each sentence utterance into a single vector
class UtteranceEncoder(nn.Module):
    def __init__(self, options):
        super(UtteranceEncoder, self).__init__()
        self.use_embed = options.use_embed
        self.hid_size = options.hid_size
        self.num_lyr = options.num_lyr
        self.drop = nn.Dropout(options.drp)
        self.direction = 2 if options.bidi else 1
        # by default they requires grad is true
        self.embed_vocab = nn.Embedding(options.vocab_size, options.emb_size, padding_idx=None, sparse=False)
        #self.embed_kb = nn.Embedding(kb_size, kb_emb_size, padding_idx =  1 , sparse= False)
        #self.embed_kb.weight.requires_grad = False
        if self.use_embed:
            #pretrained_weight_vocab, pre_trained_weight_kb = self.load_embeddings(vocab_size, voc_emb_size)
            pretrained_weight_vocab= self.load_embeddings()
            self.embed_vocab.weight.data.copy_(torch.from_numpy(pretrained_weight_vocab))
            
            #self.embed_kb.weight.data.copy_(torch.from_numpy(pre_trained_weight_kb))
            
        self.rnn = nn.GRU(input_size=options.emb_size + options.kb_embed_size , hidden_size= options.hid_size,
                          num_layers=self.num_lyr, bidirectional=options.bidi, batch_first=True, dropout=options.drp)

    
    def forward(self, inp_voc, inp_kb):
                
        #x, x_lengths = inp_voc[0], inp_voc[1]
        #y, _         = inp_kb[0], inp_kb[1]
        x        = inp_voc
        y        = inp_kb
        bt_siz, seq_len = x.size(0), x.size(1)
        h_0 = Variable(torch.zeros(self.direction * self.num_lyr, bt_siz, self.hid_size)).cuda()
        #if use_cuda:
        #    h_0 = h_0.cuda()
        token_emb = self.embed_vocab(x)
        #token_kb  = self.embed_kb(y)
        token_kb  = y
        emb       = torch.cat((token_emb, token_kb), dim = 2)
        emb = self.drop(emb)
        #emb = torch.nn.utils.rnn.pack_padded_sequence( emb, x_lengths, batch_first=True)
        gru_out, gru_hid = self.rnn( emb, h_0)
        # assuming dimension 0, 1 is for layer 1 and 2, 3 for layer 2
        if self.direction == 2:
            gru_hids = []
            for i in range(self.num_lyr):
                x_hid_temp, _ = torch.max(gru_hid[2*i:2*i + 2, :, :], 0, keepdim=True)
                gru_hids.append(x_hid_temp)
            gru_hid = torch.cat(gru_hids, 0)
        # gru_out, _ = torch.nn.utils.rnn.pad_packed_sequence(gru_out, batch_first=True)
        # using gru_out and returning gru_out[:, -1, :].unsqueeze(1) is wrong coz its all 0s careful! it doesn't adjust for variable timesteps

        gru_hid = gru_hid[self.num_lyr-1, :, :].unsqueeze(0)
        # take the last layer of the encoder GRU
        gru_hid = gru_hid.transpose(0, 1)

        return gru_hid

        
        
    
    def load_embeddings(self):
        
        embedding_matrix = pkl.load(open('Preprocessing_files/embedding_words2.pkl', 'rb'))[0]
        
        #kg_entity_embedding = np.load('Preprocessing_files/kg_embeddings')
        
        return embedding_matrix
        #return embedding_matrix, kg_entity_embedding
