from flask import Flask, render_template, request
#import markovify
import random, re
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
#import torchtext
#from transformers import BertJapaneseTokenizer
from transformers import BertModel, BertConfig
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#学習済ファイル
bertv2 = BertModel.from_pretrained('cl-tohoku/bert-base-japanese-v2')

#コーパス
def readCorpus(txt):
    data = open(txt, 'r').read()
    data = re.sub(r'\n', '', data)
    L = [i for i in data.split()]
    vocabs = set(L)
    #コーパスの作成
    word2id = {}
    for i,word in enumerate(vocabs):
        word2id[word] = i
    
    id2word = {v:k for k,v in word2id.items()}
    return word2id, id2word, len(vocabs)

#モデル
class MyBERT(nn.Module):
    
    def __init__(self, bert, vocab_size):
        super(MyBERT, self).__init__()
        self.bert = bert
        self.vocab_size = vocab_size
        self.ln = nn.Linear(768, vocab_size)
    
    def forward(self, x):
        b_out = self.bert(x)
        bs = len(b_out[0])
        h0 = [ b_out[0][i][0] for i in range(bs)]
        h0 = torch.stack(h0, dim=0)
        return self.ln(h0)



app = Flask(__name__)
#初期画面
@app.route("/")
def index():
    return render_template('index.html', flag=False)

#俳句生成
@app.route('/haiku', methods=["POST"])
def make_haiku():
    word2id, id2word, vocab_size = readCorpus("static/haiku_wakatigaki.txt")
    #重みの読み込み
    model = MyBERT(bertv2, vocab_size)
    model.load_state_dict(torch.load("static/model_80.pth", map_location=device))

    #短文生成
    initial_id = random.choice(list(word2id.values()))
    #print(initial_id)
    outs = [initial_id]
    sample_size=8
    skip_id = [word2id['。']] #ドット(.)だと終了
    #m = nn.Softmax(dim=1)
    cnt = 0
    while len(outs) < sample_size:
        x = np.array(outs[cnt]).reshape(1,1)
        x = torch.from_numpy(x).type("torch.LongTensor")
        pred = model(x)
        
        max_id = int(torch.argmax(pred.squeeze(dim=0)))
        outs.append(max_id)
        cnt+=1
        if max_id in skip_id:
            break

    #print(outs)
    gen_text = [id2word[id] for id in outs]
    print(gen_text)
    dict = {}
    #dict["haiku"] =  ''.join(*gen_text.split())[:-1]#句点を除く
    dict["haiku"] = "".join(gen_text)
    return render_template('index.html', flag=True, dict=dict)


if __name__ == "__main__":
    app.run(debug=True)