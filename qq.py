import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


# x = np.load('./THUCNews/data/vocab.embedding.sougou.npz')["embeddings"]
# print(x[0])
# print(len(x))
# print(len(x[0]))

# lis = []
# f = open('./THUCNews/data/sgns.sogou.char', encoding='UTF-8')
# for i, line in enumerate(f.readlines()):
#     if i == 0:
#         continue
#     lin = line.split(' ')[0]
#     lis.append(lin)
# f.close()
# print(lis)
# if '<PAD>' in lis:
#     print('yes')

# if '<UNK>' in lis:
#     print('yyyes')
# import argparse
# parser = argparse.ArgumentParser(description='TextCNN text classifier')
# # learning
# parser.add_argument('-lr', type=str, default='ppp', help='fff')
# parser.add_argument('-oo', type=str, default=lr+'ooo', help='fff')
# args = parser.parse_args()
# print(args.oo)


# dataset = 'THUCNews'
# embedding = 'embedding_SougouNews.npz'  # random
# embedding_pretrained = torch.tensor(np.load(dataset + '/data/' + embedding)["embeddings"].astype('float32'))
# print(embedding_pretrained.size(1))
# mode_name = 'models.TextCNN'
# from models.+ mode_name import Config, TextCNN


# w  = torch.randn(256)
# H = torch.randn(128, 32, 256)
# M = F.tanh(H)
# out = torch.matmul(M, w)
# print(out.size())


x = torch.randn(128, 32, 256)
y = torch.randn(256)
z = torch.matmul(x, y)
print(z.size())

xx = torch.tensor([[2,3],[3,4]])
yy = xx.unsqueeze(-1)
print(xx)
print(yy)