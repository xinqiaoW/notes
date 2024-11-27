![[Pasted image 20241030210231.png]]
- 批量归一化 和 层归一化：
	批量归一化是对特征进行归一化，层归一化是对样本归一化。二者都可以引入随机性，具备正则效果，同时可以令求解函数变得平缓，有利于收敛。
	
- Spatial Group Norm:
	这是为了弥补 LayerNorm 直接在所有通道上归一化（\[N, C, H, W] 中对 \[C, H, W] 归一化），
	的过程中由于不同的 channel 之间统计信息并不平均，造成的结果可能并不完美。因此把通道分割成若干组，使得每个组内部之间的这种统计信息不平均的效应减弱，从而更好地优化。 

- nn.LayerNorm()接口：
	![[Pasted image 20241031104601.png]]
	![[Pasted image 20241031104631.png]]


transformer训练、预测网络图：
![[transformer.jpg]]
Transformer通过纯注意力机制以及编码器解码器结构来预测序列。
Transformer的优势在于其可以将序列靠前信息很好地向后传递，并且并行度好，而且可以很好的抓取重要信息用于决策。

代码部分：
```
class DotProductAttention(nn.Module):  
    def __init__(self, dropout, **kwargs):  
        super(DotProductAttention, self).__init__(**kwargs)  
        self.dropout = nn.Dropout(dropout)  
  
    def forward(self, queries, keys, values, valid_length=None):  
        d = queries.shape[-1]  
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)  
        self.attention_weights = masked_softmax(scores, valid_length)  
        print(self.attention_weights)  
        return torch.bmm(self.dropout(self.attention_weights), values)  
  
  
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
  
  
class TransformerEncoder(Encoder):  
    def __init__(self, vocab_size, key_size, query_size, value_size, num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,  
                 num_heads, num_layers, dropout, use_bias=False, **kwargs):  
        super(TransformerEncoder, self).__init__(**kwargs)  
        self.num_hiddens = num_hiddens  
        self.embedding = nn.Embedding(vocab_size, num_hiddens)  
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)  
        self.blks = nn.Sequential()  
        for i in range(num_layers):  
            self.blks.add_module("block"+str(i),  
                                 EncoderBlock(key_size, query_size, value_size, num_hiddens, norm_shape, ffn_num_input,  
                                              ffn_num_hiddens, num_heads, dropout, use_bias))  
  
    def forward(self, X, valid_lens, *args):  
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))  
        for i, blk in enumerate(self.blks):  
            X = blk(X, valid_lens)  
        return X  
  
  
class DecoderBlock(nn.Module):  
    def __init__(self, key_size, query_size, value_size, num_hiddens,  
                norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,  
                dropout, i, **kwargs):  
        super(DecoderBlock, self).__init__(**kwargs)  
        self.i = i  
        self.attention1 = MultiHeadAttention(key_size, query_size, value_size, num_hiddens, num_heads, dropout)  
        self.addnorm1 = AddNorm(norm_shape, dropout)  
        self.attention2 = MultiHeadAttention(key_size, query_size, value_size, num_hiddens, num_heads, dropout)  
        self.addnorm2 = AddNorm(norm_shape, dropout)  
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)  
        self.addnorm3 = AddNorm(norm_shape, dropout)  
  
    def forward(self, X, state):  
        enc_outputs, enc_valid_lens = state[0], state[1]  
        if state[2][self.i] is None:  
            key_values = X  
        else:  
            key_values = torch.cat((state[2][self.i], X), axis=1)  
        state[2][self.i] = key_values  
        if self.training:  
            batch_size, num_steps, _ = X.shape  
            dec_valid_lens = torch.arange(1, num_steps + 1, device=X.device).repeat(batch_size, 1)  
        else:  
            dec_valid_lens = None  
        X2 = self.attention1(X, key_values, key_values, dec_valid_lens)  
        Y = self.addnorm1(X, X2)  
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)  
        Z = self.addnorm2(Y, Y2)  
        return self.addnorm3(Z, self.ffn(Z)), state  
  
  
class TransformerDecoder(AttentionDecoder):  
    def __init__(self, vocab_size, key_size, query_size, value_size, num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,  
                 num_heads, num_layers, dropout, **kwargs):  
        super(TransformerDecoder, self).__init__(**kwargs)  
        self.num_hiddens = num_hiddens  
        self.num_layers = num_layers  
        self.embedding = nn.Embedding(vocab_size, num_hiddens)  
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)  
        self.blks = nn.Sequential()  
        for i in range(num_layers):  
            self.blks.add_module("block"+str(i), DecoderBlock(key_size, query_size, value_size, num_hiddens, norm_shape,  
                                                              ffn_num_input, ffn_num_hiddens, num_heads, dropout, i))  
        self.dense = nn.Linear(num_hiddens, vocab_size)  
  
    def init_state(self, enc_outputs, enc_valid_lens, *args):  
        return [enc_outputs, enc_valid_lens, [None] * self.num_layers]  
  
    def forward(self, X, state):  
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))  
        for blk in self.blks:  
            X, state = blk(X, state)  
        return self.dense(X), state  
  
    @property  
    def attention_weights(self):  
        return self._attention_weights  
  
  
class EncoderDecoder(nn.Module):  
    def __init__(self, encoder, decoder, **kwargs):  
        super(EncoderDecoder, self).__init__(**kwargs)  
        self.encoder = encoder  
        self.decoder = decoder  
  
    def forward(self, enc_X, dec_X, *args):  
        enc_outputs = self.encoder(enc_X, *args)  
        dec_state = self.decoder.init_state(enc_outputs, *args)  
        return self.decoder(dec_X, dec_state)
```
其他部分类似。
