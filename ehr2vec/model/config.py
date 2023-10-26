def adjust_cfg_for_behrt(cfg):
    model_cfg = cfg.model
    model_cfg.hidden_dropout_prob = model_cfg.get('hidden_dropout_prob', 0.1)
    model_cfg.attention_probs_dropout_prob = model_cfg.get('attention_probs_dropout_prob', 0.1)
    model_cfg.hidden_act = model_cfg.get('hidden_act','gelu')
    model_cfg.initializer_range = model_cfg.get('initializer_range', 0.02)
    model_cfg.max_position_embeddings = model_cfg.type_vocab_size
    model_cfg.max_segment_embeddings = 2 # visits/segments
    model_cfg.age_vocab_size = 120
    model_cfg.embedding = "original_behrt"
    cfg.model = model_cfg
    return cfg