from os.path import join, split

import numpy as np
import pandas as pd
import torch
from common.logger import TqdmToLogger
from common.config import instantiate
from dataloader.collate_fn import dynamic_padding
from torch.utils.data import DataLoader
from tqdm import tqdm
from trainer.trainer import EHRTrainer


class Forwarder(EHRTrainer):
    def __init__(self, model, dataset, batch_size=50, output_path=None, pooler=None, logger=None, run=None):
        self.model = model
        self.model.eval()
        self.dataset = dataset
        self.batch_size = batch_size
        self.output_path = output_path
        self.pooler =  instantiate(pooler)

        self.logger = logger
        self.run = run
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, collate_fn=dynamic_padding)

    def forward_patients(self)->dict:
        encodings = []
        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.dataloader, desc='Batch Forward',  file=TqdmToLogger(self.logger) if self.logger else None)):
                batch.pop('target', None)
                mask = batch['attention_mask']
                self.to_device(batch)
                output = self.forward_pass(batch)
                hidden = output.last_hidden_state.detach()
                pooled_vec = self.pooler.pool(hidden, mask, output)
                encodings.append(pooled_vec)
        return torch.cat(encodings)

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
            for i, batch in enumerate(tqdm(self.dataloader, desc='Batch Forward', file=TqdmToLogger(self.logger) if self.logger else None)):
                pat_counts = [j for j in range(i*self.batch_size, (i+1)*self.batch_size)]
                if i==len(self.dataloader):
                    pat_counts = [j for j in range(i*self.batch_size, i*self.batch_size+len(batch['concept']))]
                for length in tqdm(range(start_code_id, batch['concept'].shape[1]), desc='Truncation'): 
                    trunc_batch = self.censor(batch, length)
                    self.to_device(trunc_batch)
                    output = self.forward_pass(trunc_batch)
                    mask = trunc_batch['attention_mask'][:,-1].detach().numpy().flatten().astype(bool)
                    masks.append(mask)
                    
                    seqs_len_all = seqs_len_all + [length] * self.batch_size
                    pat_counts_all = pat_counts_all + pat_counts
                    
                    for feat in data: # concept, age, abspos
                        feat_arr = (trunc_batch[feat][:,-1]).detach().numpy().flatten()[mask]
                        data[feat].append(feat_arr)
                    
                    if self.pool_method=='CLS':
                        hidden_vec = output.last_hidden_state[0]
                    else:
                        hidden_vec = self.pool_method(output.last_hidden_state, dim=1)
                    
                    hidden_all.append(hidden_vec.detach().numpy()[mask])
            
        data = {k:np.concatenate(v) for k,v in data.items()}
        masks = np.concatenate(masks)
        data['seq_len'] = np.array(seqs_len_all)[masks]
        data['pat_count'] = np.array(pat_counts_all)[masks]
        data['concept_enc'] = np.concatenate(hidden_all)
        return data

    def produce_concept_embeddings(self)->dict:
        data = {k:[] for k in ['concept', 'age', 'abspos']}
        concepts_hidden, pat_counts = [], []
        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.dataloader, desc='Batch forward',  file=TqdmToLogger(self.logger) if self.logger else None)):
                mask = batch['attention_mask'].detach().numpy().flatten().astype(bool)
                
                seq_len = batch['concept'].shape[1]
                pat_counts.append(np.repeat(np.arange(i*self.batch_size, (i+1)*self.batch_size), seq_len)[mask])
                if i==len(self.dataloader):
                    pat_counts.append(
                        np.repeat(np.arange(i*self.batch_size, i*self.batch_size+len(batch['concept'])), seq_len)[mask])
                self.to_device(batch)
                output = self.forward_pass(batch)
                hidden_states = torch.flatten(output.hidden_states[-1], end_dim=1).detach().numpy()
                concepts_hidden.append(hidden_states[mask])
                
                for feat in data:
                    feat_arr = batch[feat].detach().numpy().flatten()
                    data[feat].append(feat_arr[mask])
                
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

