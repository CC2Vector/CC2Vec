import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import re
import string
from string import punctuation

import torch
import gensim
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, WeightedRandomSampler
from torch.autograd import Variable

# Ignore warnings
import warnings
import argparse

# def parse_options():
#     parser = argparse.ArgumentParser(description='')
#     parser.add_argument('-d', '--dir', help='For example: 1/0/, 1/1/, 2/0/', required=True)
#     args = parser.parse_args()
#     return args

warnings.filterwarnings("ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print('Device: ', device)

# **Global Variables**

# arg = parse_options()
# ID_dir = arg.dir

ROOT_PATH = './data/0/'
TRAIN_FILE_PATH = ROOT_PATH + 'test_data.csv'
TRAIN_WEIGHT_FILE_PATH = ROOT_PATH + 'test_weight.csv'
EMBEDDING_PATH = './trained_vector.vector'
EMBEDDING_DIMENSION = 100
EMBEDDING_REQUIRES_GRAD = False
HIDDEN_CELLS = 100
NUM_LAYERS = 1
BATCH_SIZE = 100
THRESHOLD = 0.5
CLASS_NUM = 15


# **Load Train File and check the distribution of Duplicate Questions**

df_train = pd.read_csv(TRAIN_FILE_PATH)
print('Percentage of Duplicate Questions Pair: ', df_train['label'].mean() * 100)

df_train_weight = pd.read_csv(TRAIN_WEIGHT_FILE_PATH)


# **Data Cleansing**

def text_to_wordlist(text, remove_stopwords=False, stem_words=False):
    # Convert words to lower case and split them
    text = text.split()
    return text


# **Convert train data into list of tuples where each tuple is of the form (question1, question2)**

train_questions_pair = []
train_labels = []
for _, row in df_train.iterrows():
    q1 = list()
    q2 = list()
    i = 0
    for _, indd in row.items():
        if i < CLASS_NUM: #-------------------------------------- type num 
            q1.append(text_to_wordlist(str(indd)))
            i = i + 1
        elif (i >= CLASS_NUM and i < 2 * CLASS_NUM): # #-------------------------------------- type num 
            q2.append(text_to_wordlist(str(indd)))
            i = i + 1

    label = int(row['label'])

    if q1 and q2: # 如果等于csv中0，就用0表示
        train_questions_pair.append((q1, q2)) # 改过后，q1和q2是两个list
        train_labels.append(label)

print('Train Data Question Pairs: ', len(train_questions_pair))

train_weight_pair = []
for _, row in df_train_weight.iterrows():
    q1 = list()
    q2 = list()
    i = 0
    for _, indd in row.items():
        if i < CLASS_NUM: #-------------------------------------- type num 
            q1.append(text_to_wordlist(str(indd)))
            i = i + 1
        elif (i >= CLASS_NUM and i < 2 * CLASS_NUM): # #-------------------------------------- type num 
            q2.append(text_to_wordlist(str(indd)))
            i = i + 1

    if q1 and q2:
        train_weight_pair.append((q1, q2))


# **Create a Language class that will keep track of the dataset vocabulary and corresponding indices**

class Language:
    def __init__(self):
        self.word2index = {}
        self.index2word = {}
        self.n_words = 0

    def addSentence(self, sentence):
        for word in sentence:
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words + 1
            self.index2word[self.n_words + 1] = word
            self.n_words += 1


language = Language()
for data in [train_questions_pair]: # 改
    for q_pair in data:
        q1 = q_pair[0]
        q2 = q_pair[1]
        for qq in q1:
            language.addSentence(qq)
        for qq in q2:
            language.addSentence(qq)


# **Create a dataset class which can be indexed to retrieve Questions Pair along with corresponding Label**

class QuestionsDataset(Dataset):
    def __init__(self, questions_list, word2index, labels, train_weight_list):
        self.questions_list = questions_list
        self.labels = labels
        self.word2index = word2index
        self.train_weight_list = train_weight_list

    def __len__(self):
        return len(self.questions_list)

    def __getitem__(self, index):
        questions_pair = self.questions_list[index]
        q1 = questions_pair[0] # q1 is a sentence list
        q1_indices = []
        for sen in q1:
            tmp = []
            for word in sen:
                tmp.append(self.word2index[word])
            q1_indices.append(tmp)

        q2 = questions_pair[1]
        q2_indices = []
        for sen in q2:
            tmp = []
            for word in sen:
                tmp.append(self.word2index[word])
            q2_indices.append(tmp)

        weight_pair = self.train_weight_list[index]
        w1 = weight_pair[0] # w1 is a sentence list
        w1_indices = []
        for sen in w1:
            tmp = []
            for weight in sen:
                tmp.append(float(weight))
            w1_indices.append(tmp)

        w2 = weight_pair[1]
        w2_indices = []
        for sen in w2:
            tmp = []
            for weight in sen:
                tmp.append(float(weight))
            w2_indices.append(tmp)

        return q1_indices, q2_indices, self.labels[index], w1_indices, w2_indices


train_dataset = QuestionsDataset(train_questions_pair, language.word2index, train_labels, train_weight_pair)

n_vocabulary_words = len(language.word2index)
print('Total Unique Vocabulary Words: ', n_vocabulary_words)


# **Custom Collate is implemented to adjust the data in the desired format and calculate lengths which will later be used for padding and packing.**

class CustomCollate: # 定义如何聚合一个batch
    def custom_collate(self, batch):
        # batch = list of tuples where each tuple is of the form ([i1, i2, i3], [j1, j2, j3], label)
        q1_list = []
        q2_list = []
        labels = []
        w1_list = []
        w2_list = []
        for training_example in batch:
            q1_list.append(training_example[0])
            q2_list.append(training_example[1])
            labels.append(training_example[2])
            w1_list.append(training_example[3])
            w2_list.append(training_example[4])

        q1_lengths = []
        for q in q1_list:
            tmp = [len(m) for m in q]
            q1_lengths.append(tmp)

        q2_lengths = []
        for q in q2_list:
            tmp = [len(m) for m in q]
            q2_lengths.append(tmp)


        return q1_list, q1_lengths, q2_list, q2_lengths, labels, w1_list, w2_list

    def __call__(self, batch):
        return self.custom_collate(batch)


# **Split Training Data into Train and Validation Set**

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                           collate_fn=CustomCollate())


print('Training Set Size {}'.format(len(train_questions_pair)))

# **Create Embeding Matrix for the dataset vocabulary using pre-trained Word2Vec Embeddings**

# Load pre-trained embeddings from word2vec
word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(EMBEDDING_PATH, binary=False)
# Convert word2vec embeddings into FloatTensor
word2vec_weights = torch.FloatTensor(word2vec_model.vectors)

# Create a random weight tensor of the shape (n_vocabulary_words + 1, EMBEDDING_DIMENSION) and place each word's embedding from word2vec at the index assigned to that word
# Two key points:
# 1. Weights tensor has been initialized randomly so that the words which are part of our dataset vocabulary but are not present in word2vec are given a random embedding
# 2. Embedding at 0 index is all zeros. This is the embedding for the padding that we will do for batch processing

tempCount = 0

weights = torch.randn(n_vocabulary_words + 1, EMBEDDING_DIMENSION)
weights[0] = torch.zeros(EMBEDDING_DIMENSION)
for word, lang_word_index in language.word2index.items():
    if word in word2vec_model:
        weights[lang_word_index] = torch.FloatTensor(word2vec_model.word_vec(word))
        tempCount += 1

print('tempCount:', tempCount, 'n_vocabulary_words:', n_vocabulary_words)

del word2vec_model
del word2vec_weights


# **Siamese Network with single GRU**

class simple_transformer(nn.Module):
    def __init__(self):
        super(simple_transformer, self).__init__()

        self.transformer_encoder = nn.TransformerEncoderLayer(d_model=HIDDEN_CELLS, nhead=4)

    def forward(self, X):

        X=X.permute(1, 0, 2)

        return self.transformer_encoder(X).permute(1, 0, 2)



class network_no_siamese(nn.Module):
    def __init__(self, pretrained_weights):
        super(network_no_siamese, self).__init__()
        # Creating embedding object from the pre-trained weights
        self.embedding = nn.Embedding.from_pretrained(pretrained_weights)
        self.embedding.weight.requires_grad = EMBEDDING_REQUIRES_GRAD

        # Create a single self-attention since this is a Siamese Network and the weights are shared
        self.attention_list = []
        for j in range(CLASS_NUM):
            tmp_cla = simple_transformer()
            self.attention_list.append(tmp_cla)

        self.sub_transformer_model = simple_transformer()
        

    # Manhattan Distance Calculator
    def exponent_neg_manhattan_distance(self, x1, x2):
        return torch.exp(-torch.sum(torch.abs(x1 - x2), dim=0)).to(device)

    def forward_once(self, x, input_lengths, w, cla):

        # x is of the shape (batch_dim, sequence)
        # e.g. x = [
        #  [i1, i2, i3],
        #  [j1, j2, j3, j4]
        # ]

        # input_lengths is the list that contains the sequence lengths for each sequence
        # e.g. input_lengths = [3, 4]

        # Reverse sequence lengths indices in decreasing order as per the requirement from PyTorch before Padding and Packing
        sorted_indices = np.flipud(np.argsort(input_lengths))
        input_lengths = np.flipud(np.sort(input_lengths))
        input_lengths = input_lengths.copy()

        # Reorder questions in the decreasing order of their lengths
        ordered_questions = [torch.LongTensor(x[i]).to(device) for i in sorted_indices]

        ordered_questions_weight = [torch.FloatTensor(w[i]).to(device) for i in sorted_indices]

        # Pad sequences with 0s to the max length sequence in the batch
        ordered_questions = torch.nn.utils.rnn.pad_sequence(ordered_questions, batch_first=True)

        ordered_questions_weight = torch.nn.utils.rnn.pad_sequence(ordered_questions_weight, batch_first=True)

        # Retrieve Embeddings
        embeddings = self.embedding(ordered_questions).to(device)

        for sen in range(embeddings.shape[0]):  # for sen in range(batch size):

            if embeddings[sen].shape[0] != ordered_questions_weight[sen].view(-1, 1).shape[0]:
                continue

            embeddings[sen] = embeddings[sen] * (ordered_questions_weight[sen].view(-1, 1))

        tmp_attention = self.attention_list[cla].to(device)
        out = tmp_attention(embeddings)

        # The following step reorders the calculated activations to the original order in which questions were passed
        result = torch.FloatTensor(embeddings.size()).to(device)

        for i in range(len(out)):
            result[sorted_indices[i]] = out[i]


        result = result.mean(1)
        
        return result

    def forward(self, q1, q1_lengths, q2, q2_lengths, w1, w2):
        
        second_q1 = torch.zeros(len(q1), CLASS_NUM, HIDDEN_CELLS).to(device)
        second_q2 = torch.zeros(len(q2), CLASS_NUM, HIDDEN_CELLS).to(device)

        for cla in range(len(q1[0])):
            tmp = []
            tmp_len = []
            tmp_weight = []
            for batch_num in range(len(q1)):
                tmp.append(q1[batch_num][cla])
                tmp_len.append(len(q1[batch_num][cla]))
                tmp_weight.append(w1[batch_num][cla])
            testa = self.forward_once(tmp, tmp_len, tmp_weight, cla).to(device)
            for index in range(testa.size()[0]):
                second_q1[index][cla] = testa[index].to(device)
            
        for cla in range(len(q2[0])):
            tmp = []
            tmp_len = []
            tmp_weight = []
            for batch_num in range(len(q2)):
                tmp.append(q2[batch_num][cla])
                tmp_len.append(len(q2[batch_num][cla]))
                tmp_weight.append(w2[batch_num][cla])
            testa = self.forward_once(tmp, tmp_len, tmp_weight, cla).to(device)
            for index in range(testa.size()[0]):
                second_q2[index][cla] = testa[index].to(device)     


        out1 = self.sub_transformer_model(second_q1).to(device)
        out1 = out1.reshape(len(q1), CLASS_NUM, HIDDEN_CELLS)
        out1 = out1.mean(1).to(device)

        out2 = self.sub_transformer_model(second_q2).to(device)
        out2 = out2.reshape(len(q2), CLASS_NUM, HIDDEN_CELLS)
        out2 = out2.mean(1).to(device)

        similarity_score = torch.zeros(out1.size()[0]).to(device)
        # Calculate Similarity Score between both questions in a single pair
        for index in range(out1.size()[0]):
            # Sequence lenghts are being used to index and retrieve the activations before the zero padding since they were not part of original question
            q1 = out1[index].to(device)
            q2 = out2[index].to(device)
            with open('./output/results.txt', 'a+') as f:
                f.write(str(q1) + '---' + str(q2) + '\n')
                f.close()
            similarity_score[index] = self.exponent_neg_manhattan_distance(q1, q2)
        return similarity_score


# **load the model

model = torch.load('./output/model_4.pkl').to(device)
model.embedding = nn.Embedding.from_pretrained(weights.to(device))
model.embedding.weight.requires_grad = EMBEDDING_REQUIRES_GRAD

print(model)

# Threshold 0.5. Since similarity score will be a value between 0 and 1, we will consider all question pair with values greater than threshold as clone
threshold = torch.Tensor([0.5]).to(device)

model.eval()

val_correct_total = 0

with torch.no_grad():

    result = []
    for i, (q1_batch, q1_batch_lengths, q2_batch, q2_batch_lengths, labels, w1_batch, w2_batch) in enumerate(
            train_loader):

        
        net_labels = torch.FloatTensor(labels).to(device)

        # Run the forward pass
        similarity_score = model(q1_batch, q1_batch_lengths, q2_batch, q2_batch_lengths, w1_batch, w2_batch)


        print(i * BATCH_SIZE)

        with open('./output/acc.txt', 'a+') as f:
            for i in similarity_score.cpu().numpy():
                f.write(str(i) + '\n')
            f.close()

        predictions = (similarity_score > threshold).float() * 1

        correct = (predictions == net_labels).sum().item()
        val_correct_total += correct
        
    avg_acc_val =  val_correct_total * 100 / len(train_questions_pair)
    print ('Testing Set Size {}, Correct in test_data_set {}, Accuracy {:.2f}%'.format(len(train_questions_pair), val_correct_total, avg_acc_val))


