- nn.Embedding()的用法：
	nn.Embedding 的输入是\[ batch, num_steps]，框架会自动帮你把输入转换为 $one-hot$ 编码的形式，并进行Embedding的操作。
	nn.Embedding()会将数值压缩，转换后长度越大，则每个值越小，为了保证数值稳定性，我们往往需要 乘上 math.sqrt(later_len)，抵消掉长度对数值的影响。
- pytorch 中的 rnn、gru模组的用法：
	输入格式为\[num_steps, batch_size, code_size]
	输出格式为 output, state
	output的格式为\[num_steps, batch_size, num_hiddens]
	state的格式为\[hidden_layers, batch_size, num_hiddens]
- nn.LayerNorm()接口：
	![[Pasted image 20241031104601.png]]
	![[Pasted image 20241031104631.png]]

