RANGE_ABSPOS_IN_WEEKS = 520 # 10 years

def adjust_cfg_base(cfg):
    model_cfg = cfg.model
    model_cfg.hidden_dropout_prob = model_cfg.get('hidden_dropout_prob', 0.1)
    model_cfg.attention_probs_dropout_prob = model_cfg.get('attention_probs_dropout_prob', 0.1)
    model_cfg.hidden_act = model_cfg.get('hidden_act','gelu')
    model_cfg.initializer_range = model_cfg.get('initializer_range', 0.02)
    model_cfg.max_segment_embeddings = 2 # visits/segments
    model_cfg.age_vocab_size = 120
    cfg.model = model_cfg
    return cfg

def adjust_cfg_for_behrt(cfg):
    cfg = adjust_cfg_base(cfg)
    cfg.model.embedding = "original_behrt"
    return cfg

def adjust_cfg_for_discrete_abspos(cfg):
    cfg = adjust_cfg_base(cfg)
    cfg.model.type_vocab_size = cfg.model.get('type_vocab_size', RANGE_ABSPOS_IN_WEEKS) # around 15 years in weeks
    cfg.model.embedding = "discrete_abspos"
    return cfg