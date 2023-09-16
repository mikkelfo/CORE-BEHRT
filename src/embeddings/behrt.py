import torch


class BehrtEmbeddings(torch.nn.Module):
    """Construct the embeddings from word, segment, age"""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = torch.nn.Embedding(config.vocab_size, config.hidden_size)
        self.segment_embeddings = torch.nn.Embedding(
            config.seg_vocab_size, config.hidden_size
        )
        self.age_embeddings = torch.nn.Embedding(
            config.age_vocab_size, config.hidden_size
        )
        self.posi_embeddings = torch.nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        ).from_pretrained(
            embeddings=self._init_posi_embedding(
                config.max_position_embeddings, config.hidden_size
            )
        )

        self.LayerNorm = torch.nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        input_ids=None,
        age_ids=None,
        seg_ids=None,
        posi_ids=None,
        inputs_embeds=None,
        age=True,
        **kwargs
    ):
        if inputs_embeds is not None:
            return inputs_embeds

        if seg_ids is None:
            seg_ids = torch.zeros_like(input_ids)
        if age_ids is None:
            age_ids = torch.zeros_like(input_ids)
        if posi_ids is None:
            posi_ids = torch.zeros_like(input_ids)
        word_embed = self.word_embeddings(input_ids)
        segment_embed = self.segment_embeddings(seg_ids)
        age_embed = self.age_embeddings(age_ids)
        posi_embeddings = self.posi_embeddings(posi_ids)

        if age:
            embeddings = word_embed + segment_embed + age_embed + posi_embeddings
        else:
            embeddings = word_embed + segment_embed + posi_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    def _init_posi_embedding(self, max_position_embedding, hidden_size):
        def even_code(pos, idx):
            return torch.sin(pos / (10000 ** (2 * idx / hidden_size)))

        def odd_code(pos, idx):
            return torch.cos(pos / (10000 ** (2 * idx / hidden_size)))

        # initialize position embedding table
        lookup_table = torch.zeros(
            (max_position_embedding, hidden_size), dtype=torch.float32
        )

        # reset table parameters with hard encoding
        # set even dimension
        for pos in range(max_position_embedding):
            for idx in torch.arange(0, hidden_size, step=2):
                lookup_table[pos, idx] = even_code(pos, idx)
        # set odd dimension
        for pos in range(max_position_embedding):
            for idx in torch.arange(1, hidden_size, step=2):
                lookup_table[pos, idx] = odd_code(pos, idx)

        return torch.tensor(lookup_table)
