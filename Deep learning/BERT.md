**主要内容** 我实现了 BERT 的预训练模型。

BERT主体是一个 Transformer 的编码器（不同的是，BERT的位置编码需要自己学习），BERT 的目的是提供给一个很好的文本特征提取器，从而便于 NLP 任务的迁移学习。


为了可以很好地提取文本特征，BERT 的训练任务有两个。

1 - 在原文中随机用 \<mask> 遮掩一部分词，希望模型预测这些被遮掩的词，模型为了很好地完成这个任务，会提取文本的上下文特征， 从而可以很好的用于迁移学习。

2 - 预测两个句子是否是上下文关系。

BERT的微调也是有一定技巧的，我们可以着重将编码器输出的不同部分的特征应用于特定的任务。
例如：\<cls> 对应的特征倾向于反应一堆话之间的语义联系。

- 代码部分：
```
class PositionalEncoding(nn.Module):  
    def __init__(self, num_hiddens, dropout, max_len=1000):  
        super(PositionalEncoding, self).__init__()  
        self.dropout = nn.Dropout(dropout)  
        self.P = torch.zeros((1, max_len, num_hiddens))  
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(10000,  
                                                                                  torch.arange(0, num_hiddens, 2,  
                                                                                               dtype=torch.float32) / num_hiddens)  
        self.P[:, :, 0::2] = torch.sin(X)  
        self.P[:, :, 1::2] = torch.cos(X)  
  
    def forward(self, X):  
        X = X + self.P[:, :X.shape[1], :].to(X.device)  
        return self.dropout(X)  
  
  
def transpose_qkv(X, num_heads):  
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)  
    X = X.permute(0, 2, 1, 3)  
    return X.reshape(-1, X.shape[2], X.shape[3])  
  
  
def transpose_output(X, num_heads):  
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])  
    X = X.permute(0, 2, 1, 3)  
    return X.reshape(X.shape[0], X.shape[1], -1)  
  
  
class MultiHeadAttention(nn.Module):  
    def __init__(self, key_size, query_size, value_size, num_hiddens, num_heads, dropout, bias=False, **kwargs):  
        super(MultiHeadAttention, self).__init__(**kwargs)  
        self.num_heads = num_heads  
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)  
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)  
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)  
        self.attention = DotProductAttention(dropout)  
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)  
  
    def forward(self, queries, keys, values, valid_length):  
        queries = transpose_qkv(self.W_q(queries), self.num_heads)  
        keys = transpose_qkv(self.W_k(keys), self.num_heads)  
        values = transpose_qkv(self.W_v(values), self.num_heads)  
  
        if valid_length is not None:  
            valid_length = torch.repeat_interleave(valid_length, repeats=self.num_heads, dim=0)  
  
        output = self.attention(queries, keys, values, valid_length)  
        output_concat = transpose_output(output, self.num_heads)  
        return self.W_o(output_concat)  
  
  
class PositionWiseFFN(nn.Module):  
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_output, **kwargs):  
        super(PositionWiseFFN, self).__init__(**kwargs)  
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)  
        self.relu = nn.ReLU()  
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_output)  
  
    def forward(self, X):  
        return self.dense2(self.relu(self.dense1(X)))  
  
  
class AddNorm(nn.Module):  
    def __init__(self, norm_shape, dropout, **kwargs):  
        super(AddNorm, self).__init__(**kwargs)  
        self.ln = nn.LayerNorm(norm_shape)  
        self.dropout = nn.Dropout(dropout)  
  
    def forward(self, X, Y):  
        return self.ln(self.dropout(Y) + X)  
  
  
class EncoderBlock(nn.Module):  
    def __init__(self, key_size, query_size, value_size, num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,  
                 num_heads, dropout, use_bias=False, **kwargs):  
        super(EncoderBlock, self).__init__(**kwargs)  
        self.attention = MultiHeadAttention(key_size, query_size, value_size, num_hiddens, num_heads, dropout, use_bias)  
        self.add_norm1 = AddNorm(norm_shape, dropout)  
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)  
        self.add_norm2 = AddNorm(norm_shape, dropout)  
  
    def forward(self, X, valid_lens):  
        Y = self.add_norm1(X, self.attention(X, X, X, valid_lens))  
        return self.add_norm2(Y, self.ffn(Y))  
  
  
def get_tokens_and_segments(tokens_a, tokens_b=None):  
    tokens = ['<cls'] + tokens_a + ['<sep>']  
    segments = [0] * len(tokens)  
    if tokens_b is not None:  
        tokens += tokens_b + ['<sep>']  
        segments += [1] * (len(tokens_b) + 1)  
    return tokens, segments  
  
  
class BERTEncoder(nn.Module):  
    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input,  
                 ffn_num_hiddens, num_heads, num_layers, dropout, max_len=1000,  
                 key_size=768, query_size=768, value_size=768, **kwargs):  
        super(BERTEncoder, self).__init__(**kwargs)  
        self.token_embedding = nn.Embedding(vocab_size, num_hiddens)  
        self.segment_embedding = nn.Embedding(2, num_hiddens)  
        self.blks = nn.Sequential()  
        for i in range(num_layers):  
            self.blks.add_module("{}".format(i),  
                                 EncoderBlock(key_size, query_size, value_size,  
                                           num_hiddens, norm_shape,  
                                           ffn_num_input, ffn_num_hiddens,  
                                           num_heads, dropout))  
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, num_hiddens))  
  
    def forward(self, tokens, segments, valid_lens):  
        X = self.token_embedding(tokens) + self.segment_embedding(segments)  
        X = X + self.pos_embedding[:, :X.shape[1], :]  
        for blk in self.blks:  
            X = blk(X, valid_lens)  
        return X  
  
  
class MaskLM(nn.Module):  
    def __init__(self, vocab_size, num_hiddens, num_inputs=768, **kwargs):  
        super(MaskLM, self).__init__(**kwargs)  
        self.mlp = nn.Sequential(nn.Linear(num_inputs, num_hiddens),  
                                 nn.ReLU(),  
                                 nn.LayerNorm(num_hiddens),  
                                 nn.Linear(num_hiddens, vocab_size))  
  
    def forward(self, X, pred_positions):  
        num_pred_positions = pred_positions.shape[1]  
        pred_positions = pred_positions.reshape(-1)  
        batch_size = X.shape[0]  
        batch_idx = torch.arange(0, batch_size)  
        batch_idx = torch.repeat_interleave(batch_idx, num_pred_positions)  
        masked_X = X[batch_idx, pred_positions]  
        masked_X = masked_X.reshape((batch_size, num_pred_positions, -1))  
        mlm_Y_hat = self.mlp(masked_X)  
        return mlm_Y_hat  
  
  
class NextSentencePred(nn.Module):  
    def __init__(self, num_inputs, **kwargs):  
        super(NextSentencePred, self).__init__(**kwargs)  
        self.output = nn.Linear(num_inputs, 2)  
  
    def forward(self, X):  
        return self.output(X)  
  
  
class BERTModel(nn.Module):  
    """The BERT model."""  
  
    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input,  
                 ffn_num_hiddens, num_heads, num_layers, dropout,  
                 max_len=1000, key_size=768, query_size=768, value_size=768, hid_in_features=768, mlm_in_features=768,  
                 nsp_in_features=768):  
        super(BERTModel, self).__init__()  
        self.encoder = BERTEncoder(vocab_size, num_hiddens, norm_shape,  
                                   ffn_num_input, ffn_num_hiddens, num_heads, num_layers,  
                                   dropout, max_len=max_len, key_size=key_size,  
                                   query_size=query_size, value_size=value_size)  
        self.hidden = nn.Sequential(nn.Linear(hid_in_features, num_hiddens), nn.Tanh())  
        self.mlm = MaskLM(vocab_size, num_hiddens, mlm_in_features)  
        self.nsp = NextSentencePred(nsp_in_features)  
  
    def forward(self, tokens, segments, valid_lens=None, pred_positions=None):  
        encoded_X = self.encoder(tokens, segments, valid_lens)  
        if pred_positions is not None:  
            mlm_Y_hat = self.mlm(encoded_X, pred_positions)  
        else:  
            mlm_Y_hat = None  
        nsp_Y_hat = self.nsp(self.hidden(encoded_X[:, 0, :]))  
        return encoded_X, mlm_Y_hat, nsp_Y_hat  
  
  
def _read_wiki_text():  
    file_name = os.path.join('C:\\', 'Users', 'www', 'PycharmProjects',  
                             'Machine_learning', 'wikitext-2', 'wiki.train.tokens')  
    with open(file_name, 'r') as f:  
        lines = f.readlines()  
    paragraphs = [line.strip().lower().split('.')for line in lines if len(line.strip('.')) >= 2]  
    random.shuffle(paragraphs)  
    return paragraphs  
  
  
def _get_next_sentence(sentence, next_sentence, paragraphs):  
    if random.random() < 0.5:  
        is_next = True  
    else:  
        is_next = False  
        next_sentence = random.choice(random.choice(paragraphs))  
    return sentence, next_sentence, is_next  
  
  
def _get_nsp_data_from_paragraph(paragraph, paragraphs, vocab, max_len):  
    nsp_data_from_paragraph = []  
    for i in range(len(paragraph) - 1):  
        tokens_a, tokens_b, is_next = _get_next_sentence(paragraph[i], paragraph[i + 1], paragraphs)  
        if len(tokens_a) + len(tokens_b) + 3 > max_len:  
            continue  
        tokens, segments = get_tokens_and_segments(tokens_a, tokens_b)  
        nsp_data_from_paragraph.append((tokens, segments, is_next))  
    return nsp_data_from_paragraph  
  
  
def _replace_mlm_tokens(tokens, candidate_pred_positions, num_mlm_preds, vocab):  
    mlm_input_tokens = [token for token in tokens]  
    pred_positions_and_labels = []  
    random.shuffle(candidate_pred_positions)  
    for mlm_pred_position in candidate_pred_positions:  
        if len(pred_positions_and_labels) >= num_mlm_preds:  
            break  
        masked_token = None  
        if random.random() < 0.8:  
            masked_token = '<mask>'  
        else:  
            if random.random() < 0.5:  
                masked_token = tokens[mlm_pred_position]  
            else:  
                masked_token = vocab.idx_to_token[random.randint(0, len(vocab) - 1)]  
        mlm_input_tokens[mlm_pred_position] = masked_token  
        pred_positions_and_labels.append((mlm_pred_position, tokens[mlm_pred_position]))  
    return mlm_input_tokens, pred_positions_and_labels  
  
  
def _get_mlm_data_from_tokens(tokens, vocab):  
    candidate_pred_positions = []  
    for i, token in enumerate(tokens):  
        if token in ['<cls>', '<sep>']:  
            continue  
        candidate_pred_positions.append(i)  
    num_mlm_preds = max(1, round(len(tokens) * 0.15))  
    mlm_input_tokens, pred_positions_and_labels = _replace_mlm_tokens(  
        tokens, candidate_pred_positions, num_mlm_preds, vocab)  
    pred_positions_and_labels = sorted(pred_positions_and_labels, key=lambda x: x[0])  
    pred_positions = [v[0] for v in pred_positions_and_labels]  
    mlm_pred_labels = [v[1] for v in pred_positions_and_labels]  
    return vocab[mlm_input_tokens], pred_positions, vocab[mlm_pred_labels]  
  
  
def _pad_bert_inputs(examples, max_len, vocab):  
    max_num_mlm_preds = round(max_len * 0.15)  
    all_token_ids, all_segments, valid_lens = [], [], []  
    all_pred_positions, all_mlm_weights, all_mlm_labels = [], [], []  
    nsp_labels = []  
    for (token_ids, pred_positions, mlm_pred_label_ids, segments, is_next) in examples:  
        all_token_ids.append(torch.tensor(token_ids + [vocab['<pad>']] * (max_len - len(token_ids)), dtype=torch.long))  
        all_segments.append(torch.tensor(segments + [0] * (max_len - len(segments)), dtype=torch.long))  
        valid_lens.append(torch.tensor(len(token_ids), dtype=torch.float32))  
        all_pred_positions.append(torch.tensor(pred_positions + [0] * (max_num_mlm_preds - len(pred_positions)),  
                                               dtype=torch.long))  
        all_mlm_weights.append(  
            torch.tensor([1.0] * len(mlm_pred_label_ids) + [0.0] * (max_num_mlm_preds - len(pred_positions)),  
                         dtype=torch.float32))  
        all_mlm_labels.append(torch.tensor(mlm_pred_label_ids + [0] * (max_num_mlm_preds - len(mlm_pred_label_ids)),  
                                           dtype=torch.long))  
        nsp_labels.append(torch.tensor(is_next, dtype=torch.long))  
    return (all_token_ids, all_segments, valid_lens, all_pred_positions,  
            all_mlm_weights, all_mlm_labels, nsp_labels)  
  
  
def tokenize(lines, token='word'):  
    if token == 'word':  
        return [line.split(' ') for line in lines]  
    elif token == 'char':  
        return [list(line) for line in lines]  
    else:  
        print('错误，未知令牌类型： ' + token)  
  
  
def count_corpus(tokens):  
    if len(tokens) == 0 or isinstance(tokens[0], list):  
        tokens = [token for line in tokens for token in line]  
    return collections.Counter(tokens)  
  
  
class Vocab:  
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):  
        if tokens is None:  
            tokens = []  
        if reserved_tokens is None:  
            reserved_tokens = []  
        count = count_corpus(tokens)  
        self.token_freqs = sorted(count.items(), key=lambda x: x[1], reverse=True)  
        self.unk, uniq_tokens = 0, ['<unk>'] + reserved_tokens  
        uniq_tokens += [token for token, freq in self.token_freqs  
                        if freq >= min_freq and token not in uniq_tokens]  
        self.idx_to_token, self.token_to_idx = [], dict()  
        for token in uniq_tokens:  
            self.idx_to_token.append(token)  
            self.token_to_idx[token] = len(self.idx_to_token) - 1  
  
    def __len__(self):  
        return len(self.idx_to_token)  
  
    def __getitem__(self, tokens):  
        if not isinstance(tokens, (list, tuple)):  
            return self.token_to_idx.get(tokens, self.unk)  
        return [self.__getitem__(token) for token in tokens]  
  
    def to_tokens(self, indices):  
        if not isinstance(indices, (list, tuple)):  
            return self.idx_to_token[indices]  
        return [self.idx_to_token[index] for index in indices]  
  
  
class _WikiTextDataset(torch.utils.data.Dataset):  
    def __init__(self, paragraphs, max_len):  
        paragraphs = [tokenize(  
            paragraph, token='word') for paragraph in paragraphs]  
        sentences = [sentence for paragraph in paragraphs  
                     for sentence in paragraph]  
        self.vocab = Vocab(sentences, min_freq=5, reserved_tokens=[  
            '<pad>', '<mask>', '<cls>', '<seq>'])  
        examples = []  
        for paragraph in paragraphs:  
            examples.extend(_get_nsp_data_from_paragraph(  
                paragraph, paragraphs, self.vocab, max_len))  
        examples = [(_get_mlm_data_from_tokens(tokens, self.vocab)  
                     + (segments, is_next))  
                    for tokens, segments, is_next in examples]  
        (self.all_token_ids, self.all_segments, self.valid_lens,  
         self.all_pred_positions, self.all_mlm_weights,  
         self.all_mlm_labels, self.nsp_labels) = _pad_bert_inputs(examples, max_len, self.vocab)  
  
    def __getitem__(self, idx):  
        return (self.all_token_ids[idx], self.all_segments[idx],  
                self.valid_lens[idx], self.all_pred_positions[idx],  
                self.all_mlm_weights[idx], self.all_mlm_labels[idx],  
                self.nsp_labels[idx])  
  
    def __len__(self):  
        return len(self.all_token_ids)  
  
  
def load_data_wiki(batch_size, max_len):  
    paragraphs = _read_wiki_text()  
    train_set = _WikiTextDataset(paragraphs, max_len)  
    train_iter = torch.utils.data.DataLoader(train_set, batch_size,  
                                             shuffle=True, num_workers=0)  
    return train_iter, train_set.vocab  
  
  
batch_size, max_len = 512, 64  
train_iter, vocab = load_data_wiki(batch_size, max_len)  
net = BERTModel(len(vocab), num_hiddens=128, norm_shape=[128],  
                            ffn_num_input=128, ffn_num_hiddens=256, num_heads=2,  
                            num_layers=2, dropout=0.2, key_size=128, query_size=128,  
                            value_size=128, hid_in_features=128, mlm_in_features=128,  
                            nsp_in_features=128)  
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
loss = nn.CrossEntropyLoss()  
net = nn.DataParallel(net, device_ids=[0, 1])  
net = net.to(device)  
  
  
def _get_batch_loss_bert(net, loss, vocab_size, tokens_X, segments_X, valid_lens_X, pred_positions_X,  
                         mlm_weights_X, mlm_Y, nsp_y):  
    _, mlm_Y_hat, nsp_Y_hat = net(tokens_X, segments_X, valid_lens_X.reshape(-1), pred_positions_X)  
    mlm_loss = loss(mlm_Y_hat.reshape(-1, vocab_size), mlm_Y.reshape(-1))  
    mlm_loss = mlm_loss.sum() / (mlm_weights_X.sum() + 1e-8)  
    nsp_loss = loss(nsp_Y_hat, nsp_y)  
    loss = mlm_loss + nsp_loss  
    return loss  
  
  
def train_bert(train_iter, net, loss, vocab_size, device, num_steps):  
    trainer = torch.optim.Adam(net.parameters(), lr=1e-3)  
    num_steps_reached = False  
    step = 0  
    while step < num_steps and not num_steps_reached:  
        for tokens_X, segments_X, valid_lens_x, pred_positions_X, mlm_weights_X, mlm_Y, nsp_y in train_iter:  
            tokens_X = tokens_X.to(device)  
            segments_X = segments_X.to(device)  
            valid_lens_x = valid_lens_x.to(device)  
            pred_positions_X = pred_positions_X.to(device)  
            mlm_weights_X = mlm_weights_X.to(device)  
            mlm_Y, nsp_y = mlm_Y.to(device), nsp_y.to(device)  
            trainer.zero_grad()  
            l = _get_batch_loss_bert(net, loss, vocab_size, tokens_X, segments_X, valid_lens_x,  
                                                   pred_positions_X, mlm_weights_X, mlm_Y, nsp_y)  
            l.backward()  
            trainer.step()  
  
  
def get_bert_with_encoding(net, tokens_a, device, tokens_b=None):  
    tokens, segments = get_tokens_and_segments(tokens_a, tokens_b)  
    token_ids = torch.tensor(vocab[tokens], device=device).unsqueeze(0)  
    segments = torch.tensor(segments, device=device).unsqueeze(0)  
    valid_len = torch.tensor(len(tokens), device=device).unsqueeze(0)  
    encoded_X, _, _ = net(token_ids, segments, valid_len)  
    return encoded_X
```


