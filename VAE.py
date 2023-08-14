import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
import torch
import torch.nn as nn
from torch.autograd import Variable

import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem

import utils
import encoding_utils as eutils

config = utils.get_config(print_dict = False)

###### 1. plan for initialization
device = torch.device('cuda')

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.GRU):
        
        nn.init.orthogonal_(m.all_weights[0][0])
        nn.init.orthogonal_(m.all_weights[0][1])
        nn.init.zeros_(m.all_weights[0][2])
        nn.init.zeros_(m.all_weights[0][3])
        
    elif isinstance(m, nn.Embedding):
        nn.init.constant_(m.weight, 1)

def load_pretrained(model, pretrained = True, model_path = config["model_path"]):
    
    if pretrained:
        pretrained_dict = torch.load(model_path)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    else:
        pass
        
###### 2. sequence encoder/decoder (RNN)
class Encoder(nn.Module):
    def __init__(self, vocab_dim, emb_dim, enc_hidden_dim, emb_dropout, pad_idx):
        super().__init__()
        
        self.enc_hidden_dim = enc_hidden_dim
        self.embedding = nn.Embedding(vocab_dim, emb_dim, padding_idx = pad_idx)
        self.rnn = nn.GRU(emb_dim, enc_hidden_dim, bidirectional = True, batch_first = True)
        self.fc = nn.Linear(enc_hidden_dim * 2, enc_hidden_dim)

        self.emb_dropout = nn.Dropout(emb_dropout)
        
    def forward(self, input_seq, input_len): 
        
        input_emb = self.emb_dropout(self.embedding(input_seq))
        packed_input = nn.utils.rnn.pack_padded_sequence(input_emb, input_len.data.tolist(), batch_first = True, enforce_sorted = False)
        _, enc_h = self.rnn(packed_input)
        
        h = self.fc(torch.cat((enc_h[-2,:,:], enc_h[-1,:,:]), dim = 1))
        
        return h

class Decoder(nn.Module):
    def __init__(self, vocab_dim, emb, emb_dim, dec_hidden_dim, latent_dim, emb_dropout, pad_idx):
        super().__init__()
        
        self.pad_idx = pad_idx
        
        self.embedding = emb
        self.rnn = nn.GRU(emb_dim + latent_dim, dec_hidden_dim, batch_first = True)  # GRU
        self.emb_dropout = nn.Dropout(emb_dropout)
        self.z_to_h = nn.Linear(latent_dim, dec_hidden_dim)
        self.output_to_vocab = nn.Linear(dec_hidden_dim, vocab_dim)
        
    def forward(self, dec_input_seq, input_len, z):
        
        input_emb = self.emb_dropout(self.embedding(dec_input_seq))
        z_0 = z.unsqueeze(1).repeat(1, input_emb.size(1), 1)
        x_input = torch.cat([input_emb, z_0], dim = -1)
        packed_input = nn.utils.rnn.pack_padded_sequence(x_input, input_len.data.tolist(), batch_first=True, enforce_sorted = False)
        h_0 = self.z_to_h(z)
        h_0 = h_0.unsqueeze(0).repeat(1, 1, 1)
        output, _ = self.rnn(packed_input, h_0)
        padded_output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        y = self.output_to_vocab(padded_output)
        
        recon_loss = nn.functional.cross_entropy(y[:, :-1].contiguous().view(-1, y.size(-1)), dec_input_seq[:, 1:torch.max(input_len).item()].contiguous().view(-1), ignore_index=self.pad_idx)

        return h_0, recon_loss

###### 3. categotry encoder/decoder (MLP)

class CategoryEncoder(nn.Module):
    def __init__(self, bb1_vocab_dim, reaction_vocab_dim, emb_dim, enc_hidden_dim, cenc_dropout):
        super().__init__()
        
        # embeddings
        self.bb1_embed = nn.Embedding(bb1_vocab_dim, emb_dim)
        self.reaction_embed = nn.Embedding(reaction_vocab_dim, emb_dim)
        
        def mlp_block(in_dim, out_dim, cenc_dropout, batch_norm = True):
            
            block = [nn.Linear(in_dim, out_dim), nn.LeakyReLU()]
            if batch_norm:
                block.append(nn.BatchNorm1d(out_dim, momentum = 0.1))
            block.append(nn.Dropout(cenc_dropout, inplace = False))
            return block
        
        self.enc_blocks = nn.Sequential(
            *mlp_block(emb_dim, emb_dim * 2, cenc_dropout, batch_norm = True),
            *mlp_block(emb_dim * 2, enc_hidden_dim, cenc_dropout, batch_norm = True),  
        )
        
    def forward(self, input_features):
        
        bb1_emb = self.bb1_embed(input_features[:,0])
        reaction_emb = self.reaction_embed(input_features[:,1])
        
        enc_h = torch.stack((bb1_emb, reaction_emb))
        enc_h = torch.sum(enc_h, dim = 0)
        h = self.enc_blocks(enc_h)
        
        return h


class CategoryDecoder(nn.Module):
    def __init__(self, bb1_vocab_dim, reaction_vocab_dim, emb_dim, dec_hidden_dim, latent_dim, cdec_dropout):
        super().__init__()
        
        def mlp_block(in_dim, out_dim, cdec_dropout, batch_norm = True):
            
            block = [nn.Linear(in_dim, out_dim), nn.LeakyReLU()]
            if batch_norm:
                block.append(nn.BatchNorm1d(out_dim, momentum = 0.1))
            block.append(nn.Dropout(cdec_dropout, inplace = False))
            
            return block
        
        self.dec_blocks = nn.Sequential(
            *mlp_block(dec_hidden_dim, dec_hidden_dim * 2, cdec_dropout, batch_norm = True),
            *mlp_block(dec_hidden_dim * 2, dec_hidden_dim, cdec_dropout, batch_norm = True),
        )
        
        self.z_to_h = nn.Linear(latent_dim, dec_hidden_dim)
        
        self.output_to_bb1_vocab = nn.Linear(dec_hidden_dim, bb1_vocab_dim)
        self.output_to_reaction_vocab = nn.Linear(dec_hidden_dim, reaction_vocab_dim)
        
    def forward(self, dec_input_features, z):
        
        dec_h = self.z_to_h(z)
        output = self.dec_blocks(dec_h)
        
        bb1_out = self.output_to_bb1_vocab(output)
        reaction_out = self.output_to_reaction_vocab(output)
        
        bb1_target = dec_input_features[:,0]
        reaction_target = dec_input_features[:,1]
        
        recon_loss = nn.functional.cross_entropy(bb1_out, bb1_target) + nn.functional.cross_entropy(reaction_out, reaction_target)
        
        return recon_loss

###### 4. property predictor (MLP)
def evaluate_accuracy(pred, y):
    
    batch_size = y.size()[0]

    predicted = pred.ge(0.5).view(-1)  
    accuracy = (y == predicted).sum().float() / batch_size

    return accuracy

class PropertyPredictor(nn.Module):
    def __init__(self, latent_dim, target_dim, prop_dropout):
        super().__init__()
        
        def mlp_block(in_dim, out_dim, dropout_rate, batch_norm = True):
            
            block = [nn.Linear(in_dim, out_dim), nn.LeakyReLU()]
            if batch_norm:
                block.append(nn.BatchNorm1d(out_dim, momentum = 0.1))
            block.append(nn.Dropout(dropout_rate, inplace = False))
            return block
        
        self.pred_blocks = nn.Sequential(
            *mlp_block(latent_dim, latent_dim * 2, prop_dropout, batch_norm = True),
            *mlp_block(latent_dim * 2, latent_dim, prop_dropout, batch_norm = True),
        )
        
        self.last = nn.Linear(latent_dim, target_dim)
        
    def forward(self, z, prop_target, prop_idt):
        
        output = self.pred_blocks(z)
        
        result = torch.squeeze(self.last(output))
        result_sig = nn.functional.sigmoid(result)
        prop_loss =  nn.functional.binary_cross_entropy(result_sig * prop_idt, prop_target * prop_idt, reduction = 'sum') / (torch.sum(prop_idt) + 1e-8)
        idt_idx = torch.nonzero(prop_idt).view(-1)
        accuracy = evaluate_accuracy(result_sig[idt_idx], prop_target[idt_idx])
        
        return result_sig, prop_loss, accuracy

###### 5. VAE

class VAE(nn.Module):
    def __init__(self, encoder, decoder, category_encoder, category_decoder, prop_predictor, latent_dim, sos_idx, eos_idx, pad_idx): 
        super(VAE, self).__init__()
        
        self.pad_idx = pad_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        
        self.encoder = encoder
        self.decoder = decoder
        self.c_encoder = category_encoder
        self.c_decoder = category_decoder
        
        self.prop_predictor = prop_predictor
        
        self.h_to_mu = nn.Linear(self.encoder.enc_hidden_dim, latent_dim)
        self.h_to_logvar = nn.Linear(self.encoder.enc_hidden_dim, latent_dim)

        
    def reparameterize(self, mu, log_var): # reparameterization trick
        std = torch.exp(log_var * 0.5)
        eps = torch.randn_like(std)
        
        return mu + eps * std 
    
    def forward(self, input_seq, input_features, input_len, prop_target, prop_idt): 
        
        h_s = self.encoder(input_seq, input_len)
        h_c = self.c_encoder(input_features)
        h = h_s + h_c
        
        mu = self.h_to_mu(h)
        logvar = self.h_to_logvar(h)
        z = self.reparameterize(mu, logvar)
        
        h_0, recon_loss = self.decoder(input_seq, input_len, z)
        c_recon_loss = self.c_decoder(input_features, z)
        
        prop_result, prop_loss, accuracy = self.prop_predictor(mu, prop_target, prop_idt)
        
        return mu, logvar, recon_loss, c_recon_loss, prop_loss, accuracy
    
    
    def inference(self, inf_batch_size, max_len, temp, z = None):
        
        
        with torch.no_grad():
            
            if z is None:
                z = torch.randn(inf_batch_size, self.h_to_mu.out_features).to(device)
            else:
                z = z
            
            if inf_batch_size == 1:
                z_0 = z.view(1, 1, -1)
                
            else:
                z_0 = z.unsqueeze(1)
            
            h_0 = self.decoder.z_to_h(z)
            h_0 = h_0.unsqueeze(0)
            
            w = torch.tensor(self.sos_idx).repeat(inf_batch_size).to(device)
            x = torch.tensor(self.pad_idx).repeat(inf_batch_size, max_len).to(device)
            
            x[:, 0] = self.sos_idx
            
            eos_p = torch.tensor(max_len).repeat(inf_batch_size).to(device)
            eos_m = torch.zeros(inf_batch_size, dtype=torch.uint8).to(device)
            
            # sequence part
            for i in range(1, max_len):
                
                input_emb = self.decoder.embedding(w).unsqueeze(1)
                
                x_input = torch.cat([input_emb, z_0], dim = -1)

                o, h_0 = self.decoder.rnn(x_input, h_0)
                y = self.decoder.output_to_vocab(o.squeeze(1))
                y = nn.functional.softmax(y / temp, dim=-1)

                w = torch.multinomial(y, 1)[:, 0]
                x[~eos_m, i] = w[~eos_m]
                eos_mi = ~eos_m & (w == self.eos_idx)
                eos_p[eos_mi] = i + 1
                eos_m = eos_m | eos_mi

            new_x = []
            for i in range(x.size(0)):
                new_x.append(x[i, :eos_p[i]])
                
            new_x_cpu = [x.cpu().numpy().tolist() for x in new_x]
            
            # categorical part
            h_c = self.c_decoder.z_to_h(z)
            o_c = self.c_decoder.dec_blocks(h_c)
            
            bb1_out = self.c_decoder.output_to_bb1_vocab(o_c)
            reaction_out = self.c_decoder.output_to_reaction_vocab(o_c)
        
            bb1_id = torch.argmax(nn.functional.softmax(bb1_out, dim=-1), dim=1)
            reaction_id = torch.argmax(nn.functional.softmax(reaction_out, dim=-1), dim=1)
            
            bb1_cpu = bb1_id.detach().cpu().numpy().tolist()
            reaction_cpu = reaction_id.detach().cpu().numpy().tolist()
            
            # prediction
            pred = nn.functional.sigmoid(torch.squeeze(self.prop_predictor.last(self.prop_predictor.pred_blocks(z))))
            pred_category = pred.ge(0.5).view(-1)
            pred_cpu = pred_category.detach().cpu().numpy().tolist()
            pred_prob_cpu = pred.detach().cpu().numpy().tolist()
            
            return new_x_cpu, bb1_cpu, reaction_cpu, pred_cpu, pred_prob_cpu
        
    def input_to_latent(self, input_data):
        
        if input_data.ndim == 1:
            input_data = input_data[np.newaxis, ]
        
        bb2 = torch.from_numpy(input_data[:,:config["max_len"]]).type(torch.LongTensor)
        len = torch.from_numpy(np.array(input_data[:, config["max_len"]])).type(torch.LongTensor)
        features = torch.from_numpy(input_data[:, config["max_len"]+1:config["max_len"]+3]).type(torch.LongTensor)
        
        h_s = self.encoder(bb2.cuda(), len.cuda())
        h_c = self.c_encoder(features.cuda())
            
        h = h_s + h_c
            
        mu = self.h_to_mu(h)
        logvar = self.h_to_logvar(h)
        z = self.reparameterize(mu, logvar)
        
        return mu, z
    
    
    def latent_to_prob(self, z):
        
        pred = nn.functional.sigmoid(torch.squeeze(self.prop_predictor.last(self.prop_predictor.pred_blocks(z))))
        
        return pred
        
    def save_VAE(self, path = None):
        
        if path is None:
            path = os.path.join(os.getcwd(), 'model')
        else:
            path = os.path.join(os.getcwd(), path)
        
        if os.path.exists(path) == True:
            pass
            print('Path already existed.')
        else:
            os.mkdir(path)
            
        print('Path created.')
        
        torch.save(self.state_dict(), path + '/trained_model.pth')


###### annealing function

def linear_anneal(start = 0, stop = 1, n_epoch = 100, offset = 0, offset_0 = 0):
    
    L = np.ones(n_epoch)
    step = (stop - start)/ (n_epoch - offset - offset_0)
    
    for i in range(n_epoch):
        
        v, i = start, 0
        while v <= stop:
            
            if i < offset_0:
                L[i] = 0
            else:
                L[i] = min(v, 1)
                v += step
            i += 1
    return L

# https://github.com/haofuml/cyclical_annealing/blob/master/plot/plot_schedules.ipynb
def kl_anneal(start = 0, stop = 1, n_epoch = 100, n_cycle = 5, ratio = 0.7):
    
    L = stop * np.ones(n_epoch)
    period = n_epoch / n_cycle
    step = (stop - start)/(period * ratio) # linear schedule

    for c in range(n_cycle):

        v , i = start, 0
        while v <= stop and (int(i+c * period) < n_epoch):
            L[int(i+c * period)] = v
            v += step
            i += 1
    return L

def kl_loss_func(mu, logvar, start, stop, n_epoch, n_cycle, ratio, current_epoch):

    KL_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KL_loss = KL_loss.to(device)
    KL_weight = kl_anneal(start, stop, n_epoch, n_cycle, ratio)[current_epoch]

    return KL_loss, KL_weight

def load_VAE(pretrained = False):
    
    bb1_vocab_dim = config["bb1_vocab_dim"]
    reaction_vocab_dim = config["reaction_vocab_dim"]

    enc_dropout = config["enc_dropout"]
    dec_dropout = config["dec_dropout"]
    c_enc_dropout = config["c_enc_dropout"]
    c_dec_dropout = config["c_dec_dropout"]
    prop_dropout = config["prop_dropout"]

    vocab_dim = config["vocab_dim"]
    enc_emb_dim = config["enc_emb_dim"]
    latent_dim = config["latent_dim"]
    enc_hidden_dim = config["enc_hidden_dim"]
    dec_hidden_dim = config["dec_hidden_dim"]

    target_dim = config["target_dim"]

    pad_idx = config["pad_idx"]
    sos_idx = config["sos_idx"]
    eos_idx = config["eos_idx"]

    device = torch.device('cuda')

    # VAE
    enc = Encoder(vocab_dim, enc_emb_dim, enc_hidden_dim, enc_dropout, pad_idx)
    dec = Decoder(vocab_dim, enc.embedding, enc_emb_dim, dec_hidden_dim, latent_dim, dec_dropout, pad_idx)
    c_enc = CategoryEncoder(bb1_vocab_dim, reaction_vocab_dim, enc_emb_dim, enc_hidden_dim, c_enc_dropout)
    c_dec =  CategoryDecoder(bb1_vocab_dim, reaction_vocab_dim, enc_emb_dim, dec_hidden_dim, latent_dim, c_dec_dropout)
    prop_pred = PropertyPredictor(latent_dim, target_dim, prop_dropout)
    model = VAE(enc, dec, c_enc, c_dec, prop_pred, latent_dim, sos_idx, eos_idx, pad_idx).to(device)
    model.apply(weight_init)

    load_pretrained(model, pretrained = pretrained)
    
    return model

##################################################################################
cuda = True
device = torch.device('cuda')
clip_grad = config["clip_grad"]
FloatTensor = torch.cuda.FloatTensor
LongTensor = torch.cuda.LongTensor

n_epochs = config["n_epochs"]
sample_inteval = config["sample_interval"]
datapoint_interval = config["datapoint_interval"]
batch_size = config["batch_size"]

# model training process
def train(model = None, optimizer = None, train_dataloader = None, test_dataloader = None, figures = False):
    model.train()
    
    if figures:
        batches_list = []
        recon_losses = []
        c_recon_losses = []
        KL_losses = []
        prop_losses = []

        KL_weights = []
        recon_loss_weights = []

        test_recon_losses = []
        test_c_recon_losses = []
        test_prop_losses = []


    for epoch in range(n_epochs):
        
        model.train() # 这一行代码等价于 model.train(True)
        for i, dat in enumerate(train_dataloader):  # 要记得把 feature也 encoding 进去啊！！！！还有 y
            model.train()
            
            mol, mol_len, features, target, idt = dat
            mol = Variable(mol.type(LongTensor))
            mol_len = Variable(mol_len.type(LongTensor))
            features = Variable(features.type(LongTensor))
            target = Variable(target.type(FloatTensor))
            idt = Variable(idt.type(LongTensor))
            
            mu, logvar, recon_loss, c_recon_loss, prop_loss, train_accuracy = model(mol, features, mol_len, target, idt)
            recon_loss_weight = linear_anneal(start = config["recon_loss_start"], stop = config["recon_loss_stop"], n_epoch = n_epochs, offset = config["recon_loss_offset"], offset_0 = config["recon_loss_offset_0"])[epoch]
            KL_loss, KL_weight = kl_loss_func(mu, logvar, start = config["cyclic_loss_start"], stop = config["cyclic_loss_stop"], n_epoch = n_epochs, n_cycle = config["cyclic_loss_n_cycle"], ratio = config["cyclic_loss_ratio"], current_epoch = epoch)
            
            prop_loss_weight = linear_anneal(start = config["prop_loss_start"], stop = config["prop_loss_stop"], n_epoch = n_epochs, offset = config["prop_loss_offset"], offset_0 = config["prop_loss_offset_0"])[epoch] 
            
            KL_loss = KL_loss / batch_size
            c_recon_weighted_loss = c_recon_loss * recon_loss_weight
            KL_weighted_loss = KL_loss * KL_weight
            loss = recon_loss + c_recon_weighted_loss + prop_loss_weight * prop_loss + KL_weighted_loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
            
            batches_done = epoch * len(train_dataloader) + i
            
            
            if  batches_done % datapoint_interval == 0: 
                print("[TRAIN][Epoch %d/%d] [Batch %d/%d] [Recon Loss: %.3f][C Recon Loss: %.3f][Prop Loss: %.3f][Train Acc: %.3f][KL Loss: %.3f]"
                    % (epoch + 1, n_epochs, i + 1, len(train_dataloader), 
                    recon_loss.item(), c_recon_loss.item(), prop_loss.item(),
                    train_accuracy.item(), KL_loss.item()))
                
                if figures:
                    
                    batches_list.append(batches_done)
                    recon_losses.append(recon_loss.item())
                    c_recon_losses.append(c_recon_loss.item())
                    KL_losses.append(KL_loss.item())
                    prop_losses.append(prop_loss.item())
                    
                    KL_weights.append(KL_weight)
                    recon_loss_weights.append(recon_loss_weight)
            
            with torch.no_grad():
                
                model.eval()
                if  batches_done % (datapoint_interval) == 0:
                
                    for j, test_dat in enumerate(test_dataloader):
                    
                        test_mol, test_mol_len, test_features, test_target, test_idt = test_dat
                        test_mol = Variable(test_mol.type(LongTensor))
                        test_mol_len = Variable(test_mol_len.type(LongTensor))
                        test_features = Variable(test_features.type(LongTensor))
                        test_target = Variable(test_target.type(FloatTensor))
                        test_idt = Variable(test_idt.type(LongTensor))
                    
                        
                        test_mu, test_logvar, test_recon_loss, test_c_recon_loss, test_prop_loss, test_accuracy = model(test_mol, test_features, test_mol_len, test_target, test_idt)
                        
                        if figures:
                            test_recon_losses.append(test_recon_loss.item())
                            test_c_recon_losses.append(test_c_recon_loss.item())
                            test_prop_losses.append(test_prop_loss.item())
                        
                    
                    print("[TEST][Epoch %d/%d] [Batch %d/%d] [Recon Loss: %.3f][C Recon Loss: %.3f] [Prop Loss: %.3f][test Acc: %.3f]"
                        % (epoch + 1, n_epochs, i + 1, len(train_dataloader), 
                        test_recon_loss.item(), test_c_recon_loss.item(), test_prop_loss.item(),
                        test_accuracy.item()))

    if figures:
        epoches_list = list(range(1, 101))
        constant_recon_schedule = [1] * n_epochs
        recon_loss_linear_schedule = VAE.linear_anneal(start = config["recon_loss_start"], stop = config["recon_loss_stop"], n_epoch = n_epochs, offset = config["recon_loss_offset"], offset_0 = config["recon_loss_offset_0"])
        prop_loss_linear_schedule  = VAE.linear_anneal(start = config["prop_loss_start"], stop = config["prop_loss_stop"], n_epoch = n_epochs, offset = config["prop_loss_offset"], offset_0 = config["prop_loss_offset_0"])
        cyclic_KL_schedule = VAE.kl_anneal(start = config["cyclic_loss_start"], stop = config["cyclic_loss_stop"], n_epoch = n_epochs, n_cycle = config["cyclic_loss_n_cycle"], ratio = config["cyclic_loss_ratio"])
        utils.plot_training(batches_list, test_recon_losses, test_c_recon_losses, KL_losses, test_prop_losses)
        utils.plot_schedular(epoches_list, constant_recon_schedule, recon_loss_linear_schedule, prop_loss_linear_schedule, cyclic_KL_schedule)
    
    model.save_VAE(path = None)

from rdkit.Chem import Draw
def plot_bbs(bb_list, max_num):

    bbs = [Chem.MolFromSmiles(smi) for smi in bb_list[: max_num]]   
    Draw.MolsToGridImage(bbs,molsPerRow = 5,subImgSize=(200,200))


def gen_results_df(gen_mols, gen_bb1, gen_reaction, pred, index_to_smile, ordinal_encoder):
    
    # idx to smiles
    gen_smiles = eutils.idx_to_smiles(gen_mols, index_to_smile)
    gen_smiles_de = [eutils.remove_sos_eos(smile, mode = 'smile') for smile in gen_smiles]
    
    # idx to bb1 and reactions
    ordinal_inverse = np.array([gen_bb1,gen_reaction]).T
    bb1_and_reaction = ordinal_encoder.inverse_transform(ordinal_inverse)
    if pred is not None:
        gen_results = pd.DataFrame({"bb1_sk":bb1_and_reaction[:,0], "bb2_sk": gen_smiles_de, "reaction type": bb1_and_reaction[:,1], "collapsed": pred})
    else:
        gen_results = pd.DataFrame({"bb1_sk":bb1_and_reaction[:,0], "bb2_sk": gen_smiles_de, "reaction type": bb1_and_reaction[:,1]})
        
    return gen_results

def random_sampling(model = None, batch_size = None, index_to_smile = None, ordinal_encoder = None):
    
    model.eval()
    gen_mols, gen_bb1, gen_reaction, pred, _ = model.inference(inf_batch_size =  batch_size, max_len = config["max_len"], temp = 1)
    
    gen_results = gen_results_df(gen_mols, gen_bb1, gen_reaction, pred, index_to_smile, ordinal_encoder)
    
    return gen_results

def latent_to_mol(model = None, z_input = None, batch_size = None, index_to_smile = None, ordinal_encoder = None):
    
    model.eval()
    gen_mols, gen_bb1, gen_reaction, pred, _ = model.inference(inf_batch_size =  batch_size, max_len = config["max_len"], temp = 1, z = z_input)
    
    gen_results = gen_results_df(gen_mols, gen_bb1, gen_reaction, pred, index_to_smile, ordinal_encoder)
    
    return gen_results

##############  interpolation
# https://github.com/timbmg/Sentence-VAE
def lerp(start, end, steps):

    interpolation = np.zeros((start.shape[0], steps + 2))

    for i, (s, e) in enumerate(zip(start, end)):
        interpolation[i] = np.linspace(s, e, steps+2)
    return interpolation.T

# https://github.com/cvlab-epfl/adv_param_pose_prior/blob/main/lib/functional/interpolation.py
from sklearn.preprocessing import normalize
import math

def slerp(start, end, steps, slerp_type = 'in'):
    
    start_normed = normalize(start[np.newaxis,:])
    end_normed = normalize(end[np.newaxis,:])
    
    cos_theta = (start_normed * end_normed).sum(axis = 1)
    theta = np.repeat(np.arccos(cos_theta), steps, axis = 0).reshape(1, -1, 1)
    interp_steps = np.linspace(0, 1, steps).reshape(1,-1,1)

    if slerp_type == 'in':
        phi = theta * interp_steps 
        
    elif slerp_type == 'circle':
        phi = 2 * math.pi * interp_steps
    
    sin_theta = np.sin(theta)
    alpha = np.sin(theta - phi) / sin_theta
    beta = np.sin(phi) / sin_theta
    z = alpha * start.reshape(1, 1, -1) + beta * end.reshape(1, 1, -1)
    z_0 = z.squeeze()

    return z_0

def interpolation(z_1, z_2, steps, model, mode, index_to_smile, ordinal_encoder):
    
    if mode == 'linear':
        z_0 = torch.from_numpy(lerp(start=z_1, end=z_2, steps = steps)).float().cuda()
    
    elif mode == 'slerp-in':
        z_0 = torch.from_numpy(slerp(start=z_1, end=z_2, steps = steps, slerp_type = 'in')).float().cuda()
        
    elif mode == 'slerp-circle':
        z_0 = torch.from_numpy(slerp(start=z_1, end=z_2, steps = steps, slerp_type = 'circle')).float().cuda()
        
    gen_mols, gen_bb1, gen_reaction, pred, _ = model.inference(inf_batch_size = z_0.size()[0], max_len = config["max_len"], temp = 1, z = z_0)
    gen_results = gen_results_df(gen_mols, gen_bb1, gen_reaction, pred, index_to_smile, ordinal_encoder)
    
    return gen_results, z_0

def reconstruct_molecules_batch(input_data, model, index_to_smile, ordinal_encoder):
    
    mu, z = model.input_to_latent(input_data)
    recon_mols, recon_bb1, recon_reaction, recon_pred, _ = model.inference(inf_batch_size = input_data.shape[0], max_len = config["max_len"], temp = 1, z = mu)
    
    recon_smiles = eutils.idx_to_smiles(recon_mols, index_to_smile)
    recon_smiles = [eutils.remove_sos_eos(smile, mode = 'smile') for smile in recon_smiles]
    return recon_smiles, recon_bb1, recon_reaction

def reconstruction_accuracy(input_data, recon_data, index_to_smile):
    
    original_smile = np.array(input_data[:, :config["max_len"]]) 
    original_smiles = eutils.idx_to_smiles(original_smile, index_to_smile)
    original_smiles = [eutils.remove_sos_eos(smile, mode = 'smile') for smile in original_smiles]
    original_smiles = [eutils.remove_padding(smile, mode = 'smile') for smile in original_smiles]
    original_bb1 = input_data[:,config["max_len"] + 1]
    original_reaction = input_data[:,config["max_len"] + 2]
    
    recon_smile = recon_data[0]
    recon_bb1 = recon_data[1]
    recon_reaction = recon_data[2]
    
    #I. BB2 match
    ori_cano = [Chem.MolToSmiles(Chem.MolFromSmiles(smi)) for smi in original_smiles]
    
    index_cano_valid = []
    index_same_bb2 = []
    rec_mol = [Chem.MolFromSmiles(smi) for smi in recon_smile]
    for i in range(len(rec_mol)):
        if rec_mol[i] is not None:
            index_cano_valid.append(i)
            
    for i in index_cano_valid:
        if Chem.MolToSmiles(rec_mol[i]) == ori_cano[i]:
            index_same_bb2.append(i)
    
    
    index_same_bb1 = []
    index_same_react = []
    #II. BB1 match
    for i in range(len(original_bb1)):
        if recon_bb1[i] == original_bb1[i]:
            index_same_bb1.append(i)
    
    for i in range(len(original_reaction)):
        if recon_reaction[i] == original_reaction[i]:
            index_same_react.append(i)
    
    index_same = list(set(index_same_bb1) & set(index_same_bb2) & set(index_same_react))
    
    return len(index_same)/len(original_smile)

def reconstruct_around_single_molecule_repeat(input_data, index, batch_size, model, index_to_smile, ordinal_encoder):
    
    input_data = input_data[index][np.newaxis, ]
    input_boost = np.repeat(input_data, batch_size, axis=0)
    
    mu, z = model.input_to_latent(input_boost)
    recon_mols, recon_bb1, recon_reaction, recon_pred, _ = model.inference(inf_batch_size = input_boost.shape[0], max_len = config["max_len"], temp = 1, z = z)
    recon_smiles = eutils.idx_to_smiles(recon_mols, index_to_smile)
    recon_smiles = [eutils.remove_sos_eos(smile, mode = 'smile') for smile in recon_smiles]
    
    gen_results = gen_results_df(recon_mols, recon_bb1, recon_reaction, None, index_to_smile, ordinal_encoder)
    
    return gen_results, mu, z

from collections import Counter
from collections import OrderedDict
import matplotlib.pyplot as plt

def single_reconstruction_analysis_tool(recon_multiple, recon_mu, recon_z):
    
    recon_multiple['dist'] = [np.linalg.norm(recon_z.detach().cpu().numpy()[i] - recon_mu.detach().cpu().numpy()[i]) for i in range(recon_z.size()[0])]

    mols = []
    for i in range(len(recon_multiple['bb2_sk'])):
        mol = recon_multiple['bb2_sk'][i] + '_' + recon_multiple['bb1_sk'][i] + '_' + recon_multiple['reaction type'][i]
        mols.append(mol)
    recon_multiple['mol'] = mols

    count = Counter(mols)
    count = OrderedDict(sorted(count.items()))
    mols, frequency = zip(*count.items())
    mols = list(mols)
    frequency = list(frequency)
    mols_freq = zip(mols, frequency)
    sorted_mols_freq = sorted(mols_freq, key=lambda x:x[1], reverse=True)
    result = zip(*sorted_mols_freq)
    sorted_mols, sorted_frequency = [list(x) for x in result]
    sorted_mols_select = sorted_mols[:5]
    sorted_frequency_select = sorted_frequency[:5]
    
    ave_dist = []
    for i in range(len(sorted_mols_select)):
        index = recon_multiple[recon_multiple["mol"] == sorted_mols_select[i]].index
        average = torch.norm(recon_mu[0].detach().cpu() - torch.mean(recon_z.detach().cpu()[index], dim=0)).numpy()
        ave_dist.append(average)
        
    plt.figure(figsize=(20,10))
    plt.bar(sorted_mols_select, sorted_frequency_select)
    plt.xlabel('Molecules', fontproperties = 'Times New Roman', fontsize = 28, labelpad = 160)
    plt.ylabel('Frequency', fontproperties = 'Times New Roman', fontsize = 28)

    plt.tick_params(axis = 'both', which = 'major', labelsize = 10, labelbottom=False)
    if not os.path.exists('./figures'):
        os.makedirs('./figures')
    plt.savefig('./figures/single_recon.png')  
    plt.show()
    plt.clf()
    
    return ave_dist[1:]
    
def extract_high_prob(input_data, model, threshold = 0.9):
    
    
    model.eval()
    iter = input_data.shape[0] // 1000
    for i in range(1, iter):
        
        z, _ = model.input_to_latent(input_data[1000*(i-1): 1000*i])
        z_batch = z.detach().cpu().numpy()
        
        if i == 1:
            z_input = z_batch
        else:
            z_input = np.vstack((z_input, z_batch))
            
        pred = model.latent_to_prob(z)
        pred_batch = pred.detach().cpu().numpy()
        
        if i == 1:
            pred_input = pred_batch
        else:
            pred_input = np.hstack((pred_input, pred_batch))
            
        torch.cuda.empty_cache()
    
    z, _ = model.input_to_latent(input_data[1000*(iter - 1):])
    z_batch = z.detach().cpu().numpy()
    z_input = np.vstack((z_input,z_batch))
    
    pred = model.latent_to_prob(z)
    pred_batch = pred.detach().cpu().numpy()
    pred_input = np.hstack((pred_input, pred_batch))
    
    input_target = input_data[:, config["max_len"] + 3]
    
    df = pd.DataFrame({"True label collapse": input_target.flatten(), "predicted probability collapse": pred_input.flatten()})
    screen_non_collapse = df[(df["predicted probability collapse"] <= (1-threshold)) & (df["True label collapse"] == 0)]
    screen_collapse =  df[(df["predicted probability collapse"] >= (threshold)) & (df["True label collapse"] == 1)]
    
    return screen_non_collapse.index, screen_collapse.index 


class VAE_only(nn.Module):
    def __init__(self, encoder, decoder, category_encoder, category_decoder, latent_dim, sos_idx, eos_idx, pad_idx): 
        super(VAE_only, self).__init__()
        
        self.pad_idx = pad_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        
        self.encoder = encoder
        self.decoder = decoder
        self.c_encoder = category_encoder
        self.c_decoder = category_decoder
        
        self.h_to_mu = nn.Linear(self.encoder.enc_hidden_dim, latent_dim)
        self.h_to_logvar = nn.Linear(self.encoder.enc_hidden_dim, latent_dim)
        
    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var * 0.5)
        eps = torch.randn_like(std)
        
        return mu + eps * std 
    
    def forward(self, input_seq, input_features, input_len): 
        
        h_s = self.encoder(input_seq, input_len)
        h_c = self.c_encoder(input_features)
        h = h_s + h_c 
        
        mu = self.h_to_mu(h)
        logvar = self.h_to_logvar(h)
        z = self.reparameterize(mu, logvar)
        
        h_0, recon_loss = self.decoder(input_seq, input_len, z)
        c_recon_loss = self.c_decoder(input_features, z)
        
        return mu, logvar, recon_loss, c_recon_loss
    
    
    def inference(self, inf_batch_size, max_len, temp, z = None):
        
        
        with torch.no_grad():
            
            if z is None:
                z = torch.randn(inf_batch_size, self.h_to_mu.out_features).to(device)
            else:
                z = z
            
            if inf_batch_size == 1:
                z_0 = z.view(1, 1, -1)
                
            else:
                z_0 = z.unsqueeze(1)
            
            h_0 = self.decoder.z_to_h(z)
            h_0 = h_0.unsqueeze(0)
            
            w = torch.tensor(self.sos_idx).repeat(inf_batch_size).to(device)
            x = torch.tensor(self.pad_idx).repeat(inf_batch_size, max_len).to(device)
            
            x[:, 0] = self.sos_idx
            
            eos_p = torch.tensor(max_len).repeat(inf_batch_size).to(device)
            eos_m = torch.zeros(inf_batch_size, dtype=torch.uint8).to(device)
            
            for i in range(1, max_len):
                
                input_emb = self.decoder.embedding(w).unsqueeze(1)
                
                x_input = torch.cat([input_emb, z_0], dim = -1)

                o, h_0 = self.decoder.rnn(x_input, h_0)
                y = self.decoder.output_to_vocab(o.squeeze(1))
                y = nn.functional.softmax(y / temp, dim=-1)

                w = torch.multinomial(y, 1)[:, 0]
                x[~eos_m, i] = w[~eos_m]
                eos_mi = ~eos_m & (w == self.eos_idx)
                eos_p[eos_mi] = i + 1
                eos_m = eos_m | eos_mi

            new_x = []
            for i in range(x.size(0)):
                new_x.append(x[i, :eos_p[i]])
                
            new_x_cpu = [x.cpu().numpy().tolist() for x in new_x]
            
            h_c = self.c_decoder.z_to_h(z)
            o_c = self.c_decoder.dec_blocks(h_c)
            
            bb1_out = self.c_decoder.output_to_bb1_vocab(o_c)
            reaction_out = self.c_decoder.output_to_reaction_vocab(o_c)
        
            bb1_id = torch.argmax(nn.functional.softmax(bb1_out, dim=-1), dim=1)
            reaction_id = torch.argmax(nn.functional.softmax(reaction_out, dim=-1), dim=1)
            
            bb1_cpu = bb1_id.detach().cpu().numpy().tolist()
            reaction_cpu = reaction_id.detach().cpu().numpy().tolist()
            
            return new_x_cpu, bb1_cpu, reaction_cpu
        
    def input_to_latent(self, input_data):
        
        
        if input_data.ndim == 1:
            input_data = input_data[np.newaxis, ]
        
        bb2 = torch.from_numpy(input_data[:,:config["max_len"]]).type(torch.LongTensor)
        len = torch.from_numpy(np.array(input_data[:, config["max_len"]])).type(torch.LongTensor)
        features = torch.from_numpy(input_data[:, config["max_len"]+1:config["max_len"]+3]).type(torch.LongTensor)
        
        h_s = self.encoder(bb2.cuda(), len.cuda())
        h_c = self.c_encoder(features.cuda())
            
        h = h_s + h_c
            
        mu = self.h_to_mu(h)
        logvar = self.h_to_logvar(h)
        z = self.reparameterize(mu, logvar)
        
        return mu, z
    
    
def load_VAE_only(pretrained = False):
    
    bb1_vocab_dim = config["bb1_vocab_dim"]
    reaction_vocab_dim = config["reaction_vocab_dim"]

    enc_dropout = config["enc_dropout"]
    dec_dropout = config["dec_dropout"]
    c_enc_dropout = config["c_enc_dropout"]
    c_dec_dropout = config["c_dec_dropout"]

    vocab_dim = config["vocab_dim"]
    enc_emb_dim = config["enc_emb_dim"]
    latent_dim = config["latent_dim"]
    enc_hidden_dim = config["enc_hidden_dim"]
    dec_hidden_dim = config["dec_hidden_dim"]

    target_dim = config["target_dim"]

    pad_idx = config["pad_idx"]
    sos_idx = config["sos_idx"]
    eos_idx = config["eos_idx"]

    device = torch.device('cuda')

    enc = Encoder(vocab_dim, enc_emb_dim, enc_hidden_dim, enc_dropout, pad_idx)
    dec = Decoder(vocab_dim, enc.embedding, enc_emb_dim, dec_hidden_dim, latent_dim, dec_dropout, pad_idx)
    c_enc = CategoryEncoder(bb1_vocab_dim, reaction_vocab_dim, enc_emb_dim, enc_hidden_dim, c_enc_dropout)
    c_dec =  CategoryDecoder(bb1_vocab_dim, reaction_vocab_dim, enc_emb_dim, dec_hidden_dim, latent_dim, c_dec_dropout)
    model = VAE_only(enc, dec, c_enc, c_dec, latent_dim, sos_idx, eos_idx, pad_idx).to(device)
    model.apply(weight_init)

    load_pretrained(model, pretrained = pretrained, model_path = "./model/VAE_Model_No_Predictor.pth")
    
    return model

