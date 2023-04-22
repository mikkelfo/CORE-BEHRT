import torch
from hydra import initialize, compose
from hydra.utils import instantiate
from torch.optim import AdamW

from trainer.trainer import EHRTrainer
from model.model import HierarchicalBertEHRModel
from transformers import BertConfig


def main():
    with initialize(config_path='configs'):
        cfg: dict = compose(config_name='trainer_h.yaml')

    print('Loading datasets...')
    train_dataset = torch.load(cfg.get('train_dataset', 'dataset.train'))
    val_dataset = torch.load(cfg.get('val_dataset', 'dataset.val'))
    
    # Instantiate model
    print('Instantiating model...')
    model = HierarchicalBertEHRModel(
        BertConfig(
            emb_vocab_size=len(train_dataset.vocabulary),
            vocab_size=len(train_dataset.leaf_nodes),
            type_vocab_size=train_dataset.max_segments,
            **cfg.get('model', {}),
        ), 
        leaf_nodes=train_dataset.leaf_nodes,
        base_leaf_probs = train_dataset.base_leaf_probs
    )

    print('Instantiating optimizer...')
    opt = cfg.get('optimizer', {})
    optimizer = AdamW(
        model.parameters(),
        lr=opt.get('lr', 1e-5),
        weight_decay=opt.get('weight_decay', 0.01),
        eps=opt.get('epsilon', 1e-8),
    )
    
    print('Instantiating metrics...')
    # Instantiate metrics
    if 'metrics' in cfg:
        metrics = {k: instantiate(v) for k, v in cfg.metrics.items()}
    else:
        metrics = None
    
    print('Instantiating trainer...')
    # Instantiate trainer
    trainer = EHRTrainer( 
        model=model, 
        optimizer=optimizer,
        train_dataset=train_dataset, 
        val_dataset=val_dataset, 
        args=cfg.get('trainer_args', {}),
        metrics=metrics,
    )
    trainer.train()


if __name__ == '__main__':
    main()