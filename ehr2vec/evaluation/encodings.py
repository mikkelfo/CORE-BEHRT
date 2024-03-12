import os
from collections import namedtuple
from os.path import join, split

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_curve, roc_curve
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ehr2vec.common.config import get_function, instantiate
from ehr2vec.common.logger import TqdmToLogger
from ehr2vec.dataloader.collate_fn import dynamic_padding
from ehr2vec.trainer.trainer import EHRTrainer
from ehr2vec.trainer.utils import compute_avg_metrics, get_tqdm

class Forwarder(EHRTrainer):
    def __init__(self, model, dataset, batch_size=64, writer=None, pooler=None, logger=None, run=None):
        
        self.model = model
        self.model.eval()
        
        self.dataset = dataset
        self.batch_size = batch_size
        self.pooler =  instantiate(pooler) if pooler else None

        self.logger = logger
        self.run = run
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, collate_fn=dynamic_padding)
        if writer:
            self.writer = writer
        if self.writer:
            self.logger.info(f"Writing encodings to {self.writer.output_path}")

    def forward_patients(self)->None:
        if not self.writer:
            encodings = []
            pids = []

        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.dataloader, desc='Batch Forward',  file=TqdmToLogger(self.logger) if self.logger else None)):
                self.to_device(batch)
                batch_pids = self.dataset.pids[i*self.batch_size:(i+1)*self.batch_size]
                target = batch.pop('target', None)
                mask = batch['attention_mask']
                output = self.forward_pass(batch)
                hidden = output.last_hidden_state.detach()
                pooled_vec = self.pooler.pool(hidden, mask, output)
                
                if self.writer:
                    self.writer.write(pooled_vec, batch_pids, target)
                else:
                    encodings.append(pooled_vec.cpu())
                    pids.extend(batch_pids)
                if self.writer:
                    del output, hidden
                    torch.cuda.empty_cache()
        if not self.writer:
            return torch.cat(encodings), pids
        else:
            return None
        
    def encode_concepts(self, cfg)->None:
        selected_concept_ints = self.retrieve_selected_concept_ints(cfg)
        if not self.writer:
            encodings = []
            labels = []
        for i, batch in enumerate(tqdm(self.dataloader, desc='Batch Forward',  file=TqdmToLogger(self.logger) if self.logger else None)):
            self.to_device(batch)
            mask = batch['attention_mask']
            input = batch['concept']
            output = self.forward_pass(batch)
            hidden = output.last_hidden_state.detach()
            hidden = hidden.view(-1, hidden.shape[-1])
            mask = mask.view(-1).bool()
            hidden = hidden[mask]
            input = input.view(-1)[mask]
            filter_mask = self.obtain_filter_mask(input, selected_concept_ints)
            hidden = hidden[filter_mask]
            input = input[filter_mask]

            if self.writer:
                self.writer.write(hidden, input)
            else:
                encodings.append(hidden.cpu())
                labels.append(input.cpu())
        if not self.writer:
            return torch.cat(encodings), torch.cat(labels)
        else:
            return None
        
    def obtain_filter_mask(self, input:torch.Tensor, selected_concept_ints:torch.Tensor)->torch.Tensor:
        # Expand dimensions to make tensors broadcastable
        input = input.unsqueeze(1)  # Add a new dimension to tensor1
        selected_concept_ints = selected_concept_ints.unsqueeze(0)  # Add a new dimension to tensor2
        # Check for equality
        equality_check = input == selected_concept_ints
        # Check if any value in tensor2 matches a value in tensor1
        return torch.any(equality_check, dim=1)

    def produce_patient_trajectory_embeddings(self, start_code_id:int=4)->dict:
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
    def censor(batch: dict, length: int)->dict:
        """Censor a patient based on indices (bs=1)"""
        return {k:v[:,:length] for k,v in batch.items()}

    def retrieve_selected_concept_ints(self, cfg)->torch.tensor:
        vocabulary = torch.load(join(cfg.paths.model_path, 'vocabulary.pt'))
        return self.get_relevant_ids(cfg, vocabulary)

    def get_relevant_ids(self, cfg, vocabulary:dict)->torch.tensor:
        return torch.tensor([v for k,v in vocabulary.items() if any(k.startswith(char) for char in cfg.filter_concepts)], dtype=int)

class EHRTester:
    def __init__(self, 
        model: torch.nn.Module,
        test_dataset: Dataset = None,
        metrics: dict = {},
        args: dict = {},
        cfg = None,
        logger = None,
        run = None,
        accumulate_logits: bool = False,
        test_folder: str = None,
        mode='test'
    ):
        
        self._initialize_basic_attributes(model, test_dataset, metrics,  cfg, run, accumulate_logits, mode)
        self._set_default_args(args)
        self.logger = logger
        self.test_folder = test_folder
        self._log_basic_info()
        
        self.log("Initialize metrics")
        self.metrics = {k: instantiate(v) for k, v in metrics.items()} if metrics else {}
        
    def _initialize_basic_attributes(self, model, test_dataset, metrics, cfg, run, accumulate_logits, mode):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = model.to(self.device)
        self.test_dataset = test_dataset
        self.metrics = {k: instantiate(v) for k, v in metrics.items()} if metrics else {}
        self.cfg = cfg
        self.run = run
        self.accumulate_logits = accumulate_logits
        self.mode = mode

    def _log_basic_info(self):
        self.log(f"Run on {self.device}")
        self.log(f"Run folder: {self.test_folder}")
        self.log("Send model to device")
        self.log("Initialize metrics")
        if torch.cuda.is_available():
            self.log(f"Memory on GPU: {torch.cuda.get_device_properties(0).total_memory/1e9} GB")
        self.log(f"PyTorch version: {torch.__version__}")
        self.log(f"CUDA version: {torch.version.cuda}")
    
    def _set_default_args(self, args):
        collate_fn = get_function(args['collate_fn']) if 'collate_fn' in args else dynamic_padding
        default_args = {
            'save_every_k_steps': float('inf'),
            'collate_fn': collate_fn}
        self.args = {**default_args, **args}
        if not (self.args['effective_batch_size'] % self.args['batch_size'] == 0):
            raise ValueError('effective_batch_size must be a multiple of batch_size')

    def evaluate(self, epoch: int, mode='test')->tuple:
        """Returns the validation/test loss and metrics"""
        dataloader = self.get_test_dataloader()
        
        self.model.eval()
        loop = get_tqdm(dataloader)
        loop.set_description(mode)
        loss = 0
        
        metric_values = {name: [] for name in self.metrics}
        logits_list = [] if self.accumulate_logits else None
        targets_list = [] if self.accumulate_logits else None

        with torch.no_grad():
            for batch in loop:
                self.batch_to_device(batch)
                outputs = self.model(batch)
                loss += outputs.loss.item()

                if self.accumulate_logits:
                    logits_list.append(outputs.logits.cpu())
                    targets_list.append(batch['target'].cpu())
                else:
                    for name, func in self.metrics.items():
                        metric_values[name].append(func(outputs, batch))

        if self.accumulate_logits:
            metric_values = self.process_binary_classification_results(logits_list, targets_list, epoch, mode=mode)
        else:
            metric_values = compute_avg_metrics(metric_values)
        
        return loss / len(loop), metric_values

    def process_binary_classification_results(self, logits:list, targets:list, epoch:int, mode='val')->dict:
        """Process results specifically for binary classification."""
        targets = torch.cat(targets)
        logits = torch.cat(logits)
        batch = {'target': targets}
        outputs = namedtuple('Outputs', ['logits'])(logits)
        metrics = {}
        for name, func in self.metrics.items():
            v = func(outputs, batch)
            self.log(f"{name}: {v}")
            metrics[name] = v
        self.save_curves(logits, targets)
        self.save_metrics_to_csv(metrics)
        self.save_predictions(logits, targets)
        return metrics

    def get_test_dataloader(self):
        return DataLoader(
            self.test_dataset, 
            batch_size=self.args.get('test_batch_size', self.args['batch_size']), 
            shuffle=self.args.get('shuffle', False), 
            collate_fn=self.args['collate_fn']
        )

    def batch_to_device(self, batch: dict) -> None:
        """Moves a batch to the device in-place"""
        for key, value in batch.items():
            batch[key] = value.to(self.device)

    def log(self, message: str) -> None:
        """Logs a message to the logger and stdout"""
        if self.logger:
            self.logger.info(message)
        else:
            print(message)
    
    def save_curves(self, logits:torch.Tensor, targets:torch.Tensor)->None:
        """Saves the ROC and PRC curves to a csv file in the run folder"""
        roc_name = os.path.join(self.test_folder, f'roc_curve_{self.mode}.npz')
        prc_name = os.path.join(self.test_folder, f'prc_curve_{self.mode}.npz')
        probas = torch.sigmoid(logits).cpu().numpy()
        fpr, tpr, threshold_roc = roc_curve(targets, probas)
        precision, recall, threshold_pr = precision_recall_curve(targets, probas)
        np.savez_compressed(roc_name, fpr=fpr, tpr=tpr, threshold=threshold_roc)
        np.savez_compressed(prc_name, precision=precision, recall=recall, threshold=np.append(threshold_pr, 1))
    
    def save_predictions(self, logits:torch.Tensor, targets:torch.Tensor)->None:
        """Saves the predictions to npz files in the run folder"""
        probas_name = os.path.join(self.test_folder, f'probas_{self.mode}.npz')
        targets_name = os.path.join(self.test_folder, f'targets_{self.mode}.npz')
        probas = torch.sigmoid(logits).cpu().numpy()
        np.savez_compressed(probas_name, probas=probas)
        np.savez_compressed(targets_name, targets=targets)

    def save_metrics_to_csv(self, metrics: dict)->None:
        """Saves the metrics to a csv file"""
        metrics_name = os.path.join(self.test_folder, f'{self.mode}_scores.csv')
        with open(metrics_name, 'w') as file:
            file.write('metric,value\n')
            for key, value in metrics.items():
                file.write(f'{key},{value}\n')