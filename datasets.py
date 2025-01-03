"""
code by Tae Hwan Jung(Jeff Jung) @graykode, Derek Miller @dmmiller612, modify by shwei
Reference: https://github.com/jadore801120/attention-is-all-you-need-pytorch
           https://github.com/JayParks/transformer
           http://nlp.seas.harvard.edu/2018/04/03/attention.html#greedy-decoding
"""
import torch
import torch.utils.data as Data

sentences = [
    # 中文和英语的单词个数不要求相同
    # enc_inputs                 dec_inputs                      dec_outputs
    ['我 在 图 书 馆 里 学 习', 'S i study in the library P', 'i study in the library P E'],     # S: 开始符号  # E: 结束符号
    ['我 在 家 里 不 学 习 P', 'S i do not study at home', 'i do not study at home E']]          # P: 占位符号，如果当前句子不足固定长度用P占位

src_vocab = {'P': 0, '我': 1, '在': 2, '图': 3, '书': 4, '馆': 5, '里': 6, '学': 7, '习': 8, '家': 9, '不': 10}
src_idx2word = {i: w for i, w in enumerate(src_vocab)}
src_vocab_size = len(src_vocab)

tgt_vocab = {'P': 0, 'S': 1, 'E': 2, 'i': 3, 'study': 4, 'in': 5, 'the': 6, 'library': 7, 'do': 8, 'not': 9, 'at': 10, 'home': 11}
idx2word = {i: w for i, w in enumerate(tgt_vocab)}
tgt_vocab_size = len(tgt_vocab)

def make_data(sentences):
    """把单词序列转换为数字序列"""
    enc_inputs, dec_inputs, dec_outputs = [], [], []
    for i in range(len(sentences)):
        # a = (len(sentences))
        enc_input = [[src_vocab[n] for n in sentences[i][0].split()]]   # tensor([[ 1,  2,  3,  4,  5,  6,  7,  8], [ 1,  2,  9,  6, 10,  7,  8,  0]])
        dec_input = [[tgt_vocab[n] for n in sentences[i][1].split()]]   # tensor([[ 1,  3,  4,  5,  6,  7,  0], [ 1,  3,  8,  9,  4, 10, 11]])
        dec_output = [[tgt_vocab[n] for n in sentences[i][2].split()]]  # tensor([[ 3,  4,  5,  6,  7,  0,  2], [ 3,  8,  9,  4, 10, 11,  2]])

        enc_inputs.extend(enc_input)
        dec_inputs.extend(dec_input)
        dec_outputs.extend(dec_output)

    return torch.LongTensor(enc_inputs), torch.LongTensor(dec_inputs), torch.LongTensor(dec_outputs)

class MyDataSet(Data.Dataset):
    """自定义DataLoader"""

    def __init__(self, enc_inputs, dec_inputs, dec_outputs):
        super(MyDataSet, self).__init__()
        self.enc_inputs = enc_inputs
        self.dec_inputs = dec_inputs
        self.dec_outputs = dec_outputs

    def __len__(self):
        return self.enc_inputs.shape[0]

    def __getitem__(self, idx):
        return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx]

