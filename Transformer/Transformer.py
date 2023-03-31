# code by Tae Hwan Jung(Jeff Jung) @graykode, Derek Miller @dmmiller612
# Reference : https://github.com/jadore801120/attention-is-all-you-need-pytorch
#           https://github.com/JayParks/transformer
# 本代码借鉴自以下地址，加了中文注释，并修改了我认为不合理的地方（修改的地方有标注）
# https://github.com/graykode/nlp-tutorial/blob/master/5-1.Transformer/Transformer.ipynb
import numpy as np
import torch
import torch.nn as nn


# 将输入的句子用自己定义的简单词典编码
def make_batch(sentences):
    # 注意这里是两层括号，也就是比如input_batch的大小是：1*4
    # Transformer中训练是以句子为单位，就和图像分割中训练是以图像为单位一样，我们这里仅用1个句子来模拟，batch_size = 1
    input_batch = [[src_vocab[n] for n in sentences[0].split()]]
    decoder_input_batch = [[tgt_vocab[n] for n in sentences[1].split()]]
    target_batch = [[tgt_vocab[n] for n in sentences[2].split()]]
    return torch.LongTensor(input_batch), torch.LongTensor(decoder_input_batch), torch.LongTensor(target_batch)


def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # 词表中Pad的设置为0
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # batch_size x 1 x len_k(=len_q), one is masking
    # 返回一个长度为batch_size x len_q x len_k的矩阵，其中pad的位置用True表示，避免pad影响计算注意力得分
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # batch_size x len_q x len_k


"""
    这里的n_position是输入句子的最大长度加一（比如我们的例子中输入是“我爱你”,加一个终止符P，n_position=4）
    每一个词首先都会被one-hot编码，这里最大长度4，所以两位就可以表示，“我爱你P”的编码对照词表为“01 10 11 00”,十进制就是“1230”，输入的数据就表示为[[1, 2, 3, 0]]，这里两个括号是batch_size=1
    接着我们通过一个nn.embedding层，将one-hot编码（两位表示）转变为词嵌入（d_model=512位表示），也就是说输入从1*4转换为了1*4*512，一个词用512位表示
    因此，对于第一个词的位置0，还需要遍历d_model=512，求这个词的实际编码的各个位置，也就是get_immutable_vector（）函数
    最后返回的位置编码也是一个1*4*512的矩阵，表达了每个词编码的位置
"""


def get_sinusoid_encoding_table(n_position, d_model):
    # 这个内部函数获取的position是输入句子中一个词的position，
    def get_immutable_vector(position):
        immutable_vector = []
        for i in range(d_model):
            # 这里原代码是i // 2，论文是 i * 2，我选择了论文的做法
            immutable_vector.append(position / np.power(10000, 2 * i / d_model))
        return immutable_vector

    sinusoid_table = np.array([get_immutable_vector(every_position) for every_position in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # 上面分析了位置编码的矩阵为4*512，比如对于第一个词的编码，偶数位使用sin变换
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
    return torch.FloatTensor(sinusoid_table)

def get_attn_subsequent_mask(seq):
    """
    这个方法也写的很巧妙，对于target: 'S I love you', 第一次看见S，看不到后面，第二次看到S I，看不到后面
    其实就是一个上三角都是1的矩阵
    0 1 1 1
    0 0 1 1
    0 0 0 1
    0 0 0 0
    每次能看到的就是每一行的0代表的序列
    """
    # seq：[batch_size, tgt_len, d_model]
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequent_mask = np.triu(np.ones(attn_shape), k=1)
    subsequent_mask = torch.from_numpy(subsequent_mask).byte()
    return subsequent_mask

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2))
        scores.masked_fill_(attn_mask, 1e-9)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn


class FeedForwardNet(nn.Module):
    def __init__(self):
        super(FeedForwardNet, self).__init__()
        # 论文中有说，两个线性层和一个conv1d效果一样
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, enc_inputs):
        # enc_inputs: [batch * len_q * d_model]
        residual = enc_inputs
        enc_inputs = nn.ReLU()(self.conv1(enc_inputs.transpose(1, 2)))
        enc_inputs = self.conv2(enc_inputs).transpose(1, 2)
        return self.layer_norm(enc_inputs + residual)

class MultiHeadAttn(nn.Module):
    def __init__(self):
        super(MultiHeadAttn, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
        self.linear = nn.Linear(n_heads * d_v, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, Q, K, V, attn_mask):
        residual, batch_size = Q, Q.size(0)
        # Q: [batch_size x len_q x d_model], K: [batch_size x len_k x d_model], V: [batch_size x len_k x d_model]
        # 用具体的数字说明Xw (4*513  513*1 = 4*1)：对于encoder层，自注意力：输入Q=K=V，大小1*4*512
        # W_Q(Q)，就是一个线性映射，从512个输入映射到d_k*n_heads=512（这两个都是自定义参数）个输出，这里的线性方程表示成矩阵就是：512*1的一个参数矩阵
        # 注意这里  不同于  线性回归，输入Q=K=V（1*4*512），线性回归，是把矩阵的每一行，当做一个样本，然后y=Xw (4*513  513*1 = 4*1)，得到每个样本经过神经元后的输出，然后求损失函数这样
        # 这里的Q（1*4*512）我们，进行W_Q(Q)只是把他丢进去做一个线性变换输出的结果是（1*4*（d_k*n_heads））
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # q_s: [batch*n_heads*len_q*d_k]
        # 至于后面为什么要这样切(view函数)再转置，而不是直接像下面这样切，view()中，第一刀先把整个数据分成各个batch_size肯定没问题
        #   然后第二刀，其实是在分割各个单词，也就是说多头的q_i, k_i, v_i来自的是一个句子的每个单词的某些部分，比如“我爱你P”，每个单词由512个数字表示，分8头，第一头就是“我”里取0-64位+“爱”里取0-64位+“你”里取0-64位+“P”里取0-64位
        #   最后的d_k这一刀，则是为了保证QK之间相乘时不报错
        #   而下面这个形式的分割就是：“我爱你P”，第一个头“我”里面取512个，也就是第一个头就处理“我”这个单词，并不是我们想要的，论文中想要的是多头自注意力能够从多个角度（子空间）看待问题
        # q_s = self.W_Q(Q).view(batch_size, n_heads, -1, d_k)  # q_s: [batch*n_heads*len_q*d_k]
        # print(torch.allclose(q_s, q_t, rtol=1e-5))  # False

        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # k_s: [batch*n_heads*len_k*d_k]
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)  # v_s: [batch*n_heads*len_k*d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1,
                                                  1)  # attn_mask : [batch_size x n_heads x len_q x len_k]

        # context: [batch_size x n_heads x len_q x d_v], attn: [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        context, attn = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v)  # context: [batch_size x len_q x n_heads * d_v]
        output = self.linear(context)
        return self.layer_norm(output + residual)  # output: [batch_size x len_q x d_model]


class Encoderlayer(nn.Module):
    def __init__(self):
        super(Encoderlayer, self).__init__()
        self.enc_multi_head_attn = MultiHeadAttn()
        self.ffn = FeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        context= self.enc_multi_head_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        enc_outputs = self.ffn(context)
        return enc_outputs

class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttn()
        self.dec_enc_attn = MultiHeadAttn()
        self.pos_ffn = FeedForwardNet()

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        dec_outputs = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        dec_outputs = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs



class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(src_vocab_size, d_model)
        # 这一行首先nn.Embedding.from_pretrained()就是用一个预训练（给定的）词向量索引矩阵，来转换输入的文本
        # src_vocab_size代表为每一个输入词典中的词，都生成一个位置索引
        # 这一行比较复杂，下面有等价的写法（forward里），比较好理解
        self.pos_emb = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(src_vocab_size, d_model), freeze=True)
        self.layers = nn.ModuleList([Encoderlayer() for _ in range(n_layers)])

    def forward(self, enc_inputs):
        # 这里写的真的非常巧妙，其实就是定义了一个每个编码位置的索引矩阵：1*4*512（src_vocab*d_model），根据论文中的方法给每个词的每一个位置一个编码
        enc_outputs = self.src_emb(enc_inputs) + self.pos_emb(enc_inputs)
        # 也可以这样写，更容易理解：
        """
        # 效果完全等价，读者可以自行调试尝试，但原作者的写法真的非常高级，很巧妙的利用了nn.embedding函数，可见功力之深
        position_vector = get_sinusoid_encoding_table(src_vocab_size, d_model)  # torch.size()为4*512
        # 获取每个词的位置信息：
        pos_emb = []
        for idx in enc_inputs:   # [1, 2, 3, 0]
            pos_emb.append(position_vector[idx])
        pos_emb = torch.cat(pos_emb, dim=0).view(-1, 512)
        enc_outputs = self.src_emb(enc_inputs) + pos_emb
        """
        enc_self_attn_pad_mask = get_attn_pad_mask(enc_inputs, enc_inputs)
        for layer in self.layers:
            enc_outputs = layer(enc_outputs, enc_self_attn_pad_mask)  # enc_outputs: [batch_size x len_q x d_model]
        return enc_outputs


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_emb = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(tgt_vocab_size, d_model), freeze=True)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])
    def forward(self, dec_inputs, enc_outputs, enc_inputs):
        dec_outputs = self.tgt_emb(dec_inputs) + self.pos_emb(dec_inputs)  # dec_inputs: [batch_size * tgt_len * d_model]
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs)  # dec_self_attn_pad_mask: [batch_size * tgt_len * tgt_len]
        dec_self_attn_subsequent_mask = get_attn_subsequent_mask(dec_inputs)
        dec_self_attn_mask = torch.gt(dec_self_attn_pad_mask+dec_self_attn_subsequent_mask, 0)

        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)

        for layer in self.layers:
            dec_outputs = layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
        return dec_outputs


class Transformer(nn.Module):
    """
    编写init函数，先从大局看
    整体看Transformer分为编码，解码，以及结果输出三个模块
    于是分别定义如下
    """

    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False)

    def forward(self, enc_inputs, dec_input):
        enc_outputs = self.encoder(enc_inputs)
        dec_outputs = self.decoder(dec_input, enc_outputs, enc_inputs)
        dec_logits = self.projection(dec_outputs)
        return dec_logits.view(-1, dec_logits.size(-1))

if __name__ == '__main__':
    sentences = ['我 爱 你 P', 'S I love you', 'I love you E']

    # 原句词典
    src_vocab = {'P': 0, '我': 1, '爱': 2, '你': 3}
    src_vocab_size = len(src_vocab)

    # 翻译后的句子词典
    tgt_vocab = {'P': 0, 'I': 1, 'love': 2, 'you': 3, 'S': 4, 'E': 5}
    tgt_vocab_size = len(tgt_vocab)
    number_dict = {i: w for i, w in enumerate(tgt_vocab)}

    src_len = 4
    tgt_len = 4

    d_model = 512  # 输入编码的词向量的长度，以及模型在运行过程中的各个层输入输出向量的维度
    d_ff = 2048  # 2层全连接层的中间维度
    d_k = d_v = 64  # 设置的k, v矩阵的长度
    n_layers = 6  # 编码、解码层重复的次数
    n_heads = 8  # 几头注意力机制
    model = Transformer()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    enc_inputs, dec_inputs, target = make_batch(sentences)

    for epoch in range(20):
        optimizer.zero_grad()
        outputs = model(enc_inputs, dec_inputs)
        loss = criterion(outputs, target.contiguous().view(-1))
        print('[TRAIN] epoch：', '%04d' % (epoch + 1), ', cost：', '%.6f' % loss)
        loss.backward()
        optimizer.step()

    # Test
    predict = model(enc_inputs, dec_inputs)
    predict = predict.data.max(1, keepdim=True)[1]
    print(sentences[0], '->', [number_dict[n.item()] for n in predict.squeeze()])
