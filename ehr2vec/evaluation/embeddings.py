from os.path import join, split

import numpy as np
import pandas as pd
import torch
from dataloader.collate_fn import dynamic_padding
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from trainer.trainer import EHRTrainer
# TODO: fix accumulation methods!

class Forwarder(EHRTrainer):
    def __init__(self, model, dataset, batch_size=50, acc_method=torch.mean, test=False, layers='last'):
        self.model = model
        self.model.eval()
        self.dataset = dataset
        self.batch_size = batch_size
        self.acc_method = acc_method
        self.test = test
        self.layers = layers
        self.validate_parameters()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, collate_fn=dynamic_padding)
    
    def forward_patients(self)->dict:
        hidden_all = []
        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.dataloader, desc='Batch Forward')):
                mask = batch['attention_mask']
                self.to_device(batch)
                output = self.forward(self.model, batch)
                hidden = output.hidden_states[-1].detach()
                hidden_vec = self.get_hidden_vec(hidden, mask, output)
                hidden_all.append(hidden_vec)
                if self.test:
                    if i>1:
                        break
        return torch.cat(hidden_all)
    
    @staticmethod
    def get_hidden_vec(self, hidden, mask , output):
        if self.acc_method=='mean':
            hidden_vec = torch.mean(mask.unsqueeze(-1) * hidden, dim=1)
        elif self.acc_method=='weighted_sum':
            hidden_vec = self.attention_weighted_sum(output, hidden, mask, self.layers)
        else:
            hidden_vec = output.hidden_states[-1][0]
        
        return hidden_vec
    
    def validate_parameters(self):
        valid_methods = ['mean', 'weighted_sum', 'CLS']
        valid_layers = ['last', 'all']
        if self.acc_method not in valid_methods:
            raise ValueError(f"Method {self.acc_method} not implemented yet.")
        if self.layers not in valid_layers:
            raise ValueError(f"Layers {self.layers} not implemented yet.")

    @staticmethod
    def attention_weighted_sum(outputs, hidden, mask, layers='all'):
        """Compute embedding using attention weights"""
        attention = outputs['attentions'] # tuple num layers (batch_size, num_heads, sequence_length, sequence_length)
        if layers=='all':
            attention = torch.stack(attention).mean(dim=0).mean(dim=1) # average over all layers and heads
        elif layers=='last':
            attention = attention[-1].mean(dim=1) # average over all layers and heads
        else:
            raise ValueError(f"Layers {layers} not implemented yet.")
        weights = torch.mean(attention, dim=1) # (batch_size, sequence_length, sequence_length)
        weights = weights / torch.sum(weights, dim=1, keepdim=True) # normalize, potentially uise softmax
        hidden_vec = torch.sum(mask.unsqueeze(-1) * hidden * weights.unsqueeze(-1), dim=1)
        return hidden_vec

    def produce_patient_trajectory_embeddings(self, start_code_id=4)->dict:
        """Produce embedding for each patient. Showing an increasing number of codes to the model,
        starting with start_code_id. 
        Returns:
            pat_emb: tensor of shape (num_pat*K, hidden_dim), where K is the sum of the number of steps per patient
            pat_num
            step_num: e.g. 2: show first 2 codes only 
        """
        pat_counts_all, seqs_len_all, hidden_all, masks = [], [], [], []
        data = {k:[] for k in self.dataset.features.keys() if k in ['concept', 'age', 'abspos']}
        
        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.dataloader, desc='Batch Forward')):
                pat_counts = [j for j in range(i*self.batch_size, (i+1)*self.batch_size)]
                if i==len(self.dataloader):
                    pat_counts = [j for j in range(i*self.batch_size, i*self.batch_size+len(batch['concept']))]
                for length in tqdm(range(start_code_id, batch['concept'].shape[1]), desc='Truncation'): 
                    trunc_batch = self.censor(batch, length)
                    self.to_device(trunc_batch)
                    output = self.forward(self.model, trunc_batch)
                    mask = trunc_batch['attention_mask'][:,-1].detach().numpy().flatten().astype(bool)
                    masks.append(mask)
                    
                    seqs_len_all = seqs_len_all + [length] * self.batch_size
                    pat_counts_all = pat_counts_all + pat_counts
                    
                    for feat in data: # concept, age, abspos
                        feat_arr = (trunc_batch[feat][:,-1]).detach().numpy().flatten()[mask]
                        data[feat].append(feat_arr)
                    
                    if self.acc_method=='CLS':
                        hidden_vec = output.hidden_states[-1][0]
                    else:
                        hidden_vec = self.acc_method(output.hidden_states[-1], dim=1)
                    
                    hidden_all.append(hidden_vec.detach().numpy()[mask])
            
                if self.stop_iter:
                    if i>=(self.stop_iter-1):
                        break
    
        data = {k:np.concatenate(v) for k,v in data.items()}
        masks = np.concatenate(masks)
        data['seq_len'] = np.array(seqs_len_all)[masks]
        data['pat_count'] = np.array(pat_counts_all)[masks]
        data['concept_enc'] = np.concatenate(hidden_all)
        return data

    def produce_concept_embeddings(self, model, dataset: Dataset, batch_size:int=50, stop_iter:int=None)->dict:
        data = {k:[] for k in dataset.features.keys() if k in ['concept', 'age', 'abspos']}
        concepts_hidden, pat_counts = [], []
        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.dataloader, desc='Batch forward')):
                mask = batch['attention_mask'].detach().numpy().flatten().astype(bool)
                
                seq_len = batch['concept'].shape[1]
                pat_counts.append(np.repeat(np.arange(i*batch_size, (i+1)*batch_size), seq_len)[mask])
                if i==len(self.dataloader):
                    pat_counts.append(
                        np.repeat(np.arange(i*batch_size, i*batch_size+len(batch['concept'])), seq_len)[mask])
                self.to_device(batch, self.device)
                output = self.forward(model, batch)
                hidden_states = torch.flatten(output.hidden_states[-1], end_dim=1).detach().numpy()
                concepts_hidden.append(hidden_states[mask])
                
                for feat in data:
                    feat_arr = batch[feat].detach().numpy().flatten()
                    data[feat].append(feat_arr[mask])
            
                if stop_iter:
                    if i>=(stop_iter-1):
                        break
        data = {k:np.concatenate(v) for k,v in data.items()}      
        data['concept_enc'] = np.concatenate(concepts_hidden)
        data['patient'] = np.concatenate(pat_counts)
        return data

    @staticmethod
    def store_to_df(data: dict, data_path: str)->pd.DataFrame:
        """Store data in dataframe, get concept names and change dtype to reduce memory use."""
        vocab = torch.load(join(split(data_path)[0], 'vocabulary.pt'))
        inv_vocab = {v:k for k,v in vocab.items()}
        info_cols = [k for k in data.keys() if not k.startswith('P_')]
        df = pd.DataFrame(data)
        for info_col in info_cols:
            if info_col!='abspos':
                df[info_col] = df[info_col].astype('int16')
        df['concept_name'] = df.concept.map(inv_vocab) 
        return df    
    
    @staticmethod
    def censor(batch: dict, length: int):
        """Censor a patient based on indices (bs=1)"""
        return {k:v[:,:length] for k,v in batch.items()}

