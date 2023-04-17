import torch
from hydra import initialize, compose
from hydra.utils import instantiate
from torch.optim import AdamW
from data.dataset import CensorDataset
from trainer.trainer import EHRTrainer
from model.model import BertEHRModel
from transformers import BertConfig


def main_finetune():
    with initialize(config_path='configs'):
        cfg: dict = compose(config_name='finetune.yaml')

    # Finetune specific
    train_encoded = torch.load(cfg.get('train_encoded', 'train_encoded.pt'))
    train_outcomes = torch.load(cfg.get('train_outcomes', 'train_outcomes.pt'))
    test_encoded = torch.load(cfg.get('test_encoded', 'test_encoded.pt'))
    test_outcomes = torch.load(cfg.get('test_outcomes', 'test_outcomes.pt'))
    vocabulary = torch.load(cfg.get('vocabulary', 'vocabulary.pt'))
    n_hours, outcome_type, censor_type = cfg.outcome.n_hours, cfg.outcome.censor_type, cfg.outcome.type
    train_dataset = CensorDataset(train_encoded, n_hours=n_hours, outcomes=train_outcomes[outcome_type], censor_outcomes=train_outcomes[censor_type])
    test_dataset = CensorDataset(test_encoded, n_hours=n_hours, outcomes=test_outcomes[outcome_type], censor_outcomes=test_outcomes[censor_type])

    print(f'Setting up finetune task on [{outcome_type}] with [{n_hours}] hours censoring')

    model = BertEHRModel(
        BertConfig(
            vocab_size=len(vocabulary),
            type_vocab_size=train_dataset.max_segments,
            **cfg.get('model', {}),
        )
    )
    model.load_state_dict(torch.load(cfg.get('pretrained_model', 'pretrained_model.pt')))
    model.replace_head()

    opt = cfg.get('optimizer', {})
    optimizer = AdamW(
        model.parameters(),
        lr=opt.get('lr', 1e-4),
        weight_decay=opt.get('weight_decay', 0.01),
        eps=opt.get('epsilon', 1e-8),
    )

    trainer = EHRTrainer( 
        model=model, 
        optimizer=optimizer,
        train_dataset=train_dataset, 
        test_dataset=test_dataset, 
        args=cfg.get('trainer_args', {}),
        metrics=cfg.metrics,
    )
    trainer.test()


if __name__ == '__main__':
    main_finetune()