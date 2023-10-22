# CC2Vec

*CC2Vec: Combining Typed Tokens with Contrastive Learning for Effective Code Clone Detection*


This is the open-source code repository for under-review paper "CC2Vec: Combining Typed Tokens with Contrastive Learning for Effective Code Clone Detection"


### Getting Started

#### Requirements

```
pytorch
cudatoolkit
datasets
transformers
gensim
```

#### Code Structure

```
CC2Vec/
|--scrpts/                # scripts for CC2Vec
  |--bash.py
  |--dot2sent.py
  |--word2csv.py
  |-- ...
|--train_att.py           # pretrain for CC2Vec
|--evalutate.py           # evaluate models
```



### Usage

#### Pretrain for CC2Vec

```bash
python train_att.py
```

#### Scripts for CC2Vec

```bash
python \scripts\*.py
```