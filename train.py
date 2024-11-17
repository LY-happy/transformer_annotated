"""
code by Tae Hwan Jung(Jeff Jung) @graykode, Derek Miller @dmmiller612, modify by shwei
Reference: https://github.com/jadore801120/attention-is-all-you-need-pytorch
           https://github.com/JayParks/transformer
           http://nlp.seas.harvard.edu/2018/04/03/attention.html#greedy-decoding
"""
import torch.nn as nn
from datasets import *
from transformer import Transformer

if __name__ == "__main__":

    enc_inputs, dec_inputs, dec_outputs = make_data(sentences)
    loader = Data.DataLoader(MyDataSet(enc_inputs, dec_inputs, dec_outputs), 2, True)
    
    model = Transformer().to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)  

    for epoch in range(epochs + 1):
        for enc_inputs, dec_inputs, dec_outputs in loader:
            """
            enc_inputs: [batch_size, src_len]
            dec_inputs: [batch_size, tgt_len]
            dec_outputs: [batch_size, tgt_len]
            """
            enc_inputs, dec_inputs, dec_outputs = enc_inputs.to(device), dec_inputs.to(device), dec_outputs.to(device)
            # outputs: [batch_size * tgt_len, tgt_vocab_size]
            outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
            loss = criterion(outputs, dec_outputs.view(-1))  # dec_outputs.view(-1) :[batch_size * tgt_len * tgt_vocab_size]
            if epoch % 20 == 0:
                print(f"------第{epoch}训练开始------  'loss = {loss:.2f}'")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % 20 == 0:
                torch.save(model.state_dict(), f'model_{epoch}.pth')
                print(f'第{epoch}轮模型已保存')
    
