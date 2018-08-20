import torch.nn as nn, torch, copy, tqdm, math
from torch.autograd import Variable
import torch.nn.functional as F
import pickle as pkl
import numpy as np
use_cuda = torch.cuda.is_available()

from util import *



# decode the hidden state
class Decoder(nn.Module):
    def __init__(self, options, res_vocab_size):
        super(Decoder, self).__init__()
        self.emb_size = options.emb_size
        self.hid_size = options.dec_hid_size
        self.num_lyr = 1
        self.teacher_forcing = options.teacher
        self.train_lm = options.lm

        self.drop = nn.Dropout(options.drp)
        self.tanh = nn.Tanh()
        self.shared_weight = options.shrd_dec_emb

        self.embed_in = nn.Embedding(res_vocab_size, self.emb_size, padding_idx=None, sparse=False)
        
        if options.use_embed:
            pretrained_weight_vocab = self.load_embeddings()
            self.embed_in.weight.data.copy_(torch.from_numpy(pretrained_weight_vocab))
            
        if not self.shared_weight:
            self.embed_out = nn.Linear(self.emb_size, res_vocab_size, bias=False)

        self.rnn = nn.GRU(hidden_size=self.hid_size,input_size=self.emb_size,num_layers=self.num_lyr,batch_first=True, dropout=options.drp)

        if options.seq2seq:
            self.ses_to_dec = nn.Linear(2 * options.ut_hid_size, self.hid_size)
            self.dec_inf = nn.Linear(self.hid_size, self.emb_size*2, False)
            self.ses_inf = nn.Linear(2 * options.ut_hid_size, self.emb_size*2, False)
        else:
            self.ses_to_dec = nn.Linear(options.ses_hid_size, self.hid_size)
            self.dec_inf = nn.Linear(self.hid_size, self.emb_size*2, False)
            self.ses_inf = nn.Linear(options.ses_hid_size, self.emb_size*2, False)
        self.emb_inf = nn.Linear(self.emb_size, self.emb_size*2, True)
        self.tc_ratio = 1.0

        if options.lm:
            self.lm = nn.GRU(input_size=self.emb_size, hidden_size=self.hid_size, num_layers=self.num_lyr, batch_first=True, dropout=options.drp, bidirectional=False)
            self.lin3 = nn.Linear(self.hid_size, self.emb_size, False)

    def do_decode_tc(self, context_encoding, target):
        #print(target.size(), target_lengths)
        target_emb = self.embed_in(target)
        target_emb = self.drop(target_emb)
        # below will be used later as a crude approximation of an LM
        emb_inf_vec = self.emb_inf(target_emb)

        #target_emb = torch.nn.utils.rnn.pack_padded_sequence(target_emb, target_lengths, batch_first=True)

        #print(context_encoding.size())
        init_hidn = self.tanh(self.ses_to_dec(context_encoding))
        #print(init_hidn.size())
        init_hidn = init_hidn.view(self.num_lyr, target.size(0), self.hid_size)

        hid_o, hid_n = self.rnn(target_emb, init_hidn)
        #hid_o, _ = torch.nn.utils.rnn.pad_packed_sequence(hid_o, batch_first=True)
        # linear layers not compatible with PackedSequence need to unpack, will be 0s at padded timesteps!

        dec_hid_vec = self.dec_inf(hid_o)
        ses_inf_vec = self.ses_inf(context_encoding)
        #print(dec_hid_vec.size(), ses_inf_vec.size(), emb_inf_vec.size())
        total_hid_o = dec_hid_vec + ses_inf_vec + emb_inf_vec

        hid_o_mx = max_out(total_hid_o)
        hid_o_mx = F.linear(hid_o_mx, self.embed_in.weight) if self.shared_weight else self.embed_out(hid_o_mx)

        if self.train_lm:
            siz = target.size(0)

            lm_hid0 = Variable(torch.zeros(self.num_lyr, siz, self.hid_size))
            if use_cuda:
                lm_hid0 = lm_hid0.cuda()

            lm_o, lm_hid = self.lm(target_emb, lm_hid0)
            lm_o, _ = torch.nn.utils.rnn.pad_packed_sequence(lm_o, batch_first=True)
            lm_o = self.lin3(lm_o)
            lm_o = F.linear(lm_o, self.embed_in.weight) if self.shared_weight else self.embed_out(lm_o)
            return hid_o_mx, lm_o
        else:
            return hid_o_mx, None


    def do_decode(self, siz, seq_len, context_encoding, target):
        ses_inf_vec = self.ses_inf(context_encoding)
        context_encoding = self.tanh(self.ses_to_dec(context_encoding))
        hid_n, preds, lm_preds = context_encoding, [], []

        hid_n = hid_n.view(self.num_lyr, siz, self.hid_size)
        inp_tok = Variable( torch.zeros(siz, 1).long())
        lm_hid = Variable(torch.zeros(self.num_lyr, siz, self.hid_size))
        if use_cuda:
            lm_hid = lm_hid.cuda()
            inp_tok = inp_tok.cuda()


        for i in range(seq_len):
            # initially tc_ratio is 1 but then slowly decays to 0 (to match inference time)
            if torch.randn(1)[0] < self.tc_ratio:
                inp_tok = target[:, i].unsqueeze(1)

            inp_tok_embedding = self.embed_in(inp_tok)
            emb_inf_vec = self.emb_inf(inp_tok_embedding)

            inp_tok_embedding = self.drop(inp_tok_embedding)

            hid_o, hid_n = self.rnn(inp_tok_embedding, hid_n)
            dec_hid_vec = self.dec_inf(hid_o)

            total_hid_o = dec_hid_vec + ses_inf_vec + emb_inf_vec
            hid_o_mx = max_out(total_hid_o)

            hid_o_mx = F.linear(hid_o_mx, self.embed_in.weight) if self.shared_weight else self.embed_out(hid_o_mx)
            preds.append(hid_o_mx)

            if self.train_lm:
                lm_o, lm_hid = self.lm(inp_tok_embedding, lm_hid)
                lm_o = self.lin3(lm_o)
                lm_o = F.linear(lm_o, self.embed_in.weight) if self.shared_weight else self.embed_out(lm_o)
                lm_preds.append(lm_o)

            #op = hid_o_mx[:, :, 1:]
            op = hid_o_mx
            op = F.log_softmax(op, 2, 5)
            max_val, inp_tok = torch.max(op, dim=2)
            # now inp_tok will be val between 0 and 10002 ignoring padding_idx
            # here we do greedy decoding
            # so we can ignore the last symbol which is a padding token
            # technically we don't need a softmax here as we just want to choose the max token, max score will result in max softmax.Duh!

        dec_o = torch.cat(preds, 1)
        dec_lmo = torch.cat(lm_preds, 1) if self.train_lm else None
        return dec_o, dec_lmo

    def forward(self, input):
        if len(input) == 1:
            context_encoding = input
            x, x_lengths = None, None
            beam = 5
        elif len(input) == 2:
            context_encoding, x, = input
            beam = 5
        else:
            context_encoding, x,  beam = input

        if use_cuda:
            x = x.cuda()
        siz, seq_len = x.size(0), x.size(1)

        if self.teacher_forcing:
            dec_o, dec_lm = self.do_decode_tc(context_encoding, x)
        else:
            dec_o, dec_lm = self.do_decode(siz, seq_len, context_encoding, x)

        return dec_o, dec_lm

    def set_teacher_forcing(self, val):
        self.teacher_forcing = val

    def get_teacher_forcing(self):
        return self.teacher_forcing

    def set_tc_ratio(self, new_val):
        self.tc_ratio = new_val

    def get_tc_ratio(self):
        return self.tc_ratio
    
    def load_embeddings(self):
        embedding_matrix = pkl.load(open('Preprocessing_files/embedding_words2.pkl', 'rb'))[1]
        return embedding_matrix

