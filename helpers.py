import torch
import unicodedata
import re
import dictionary
import itertools
from dictionary import PAD,EOS

MAX_LENGTH = 10
MIN_COUNT = 3

def unicodeToAscii(i):
    asciiData = "".join(b for b in unicodedata.normalize("NFD",i) if unicodedata.category(b) != "Mn")
    return asciiData

def stringNormalizer(i):
    i = unicodeToAscii(i.lower().strip())
    i = re.sub(r"([.!?])", r" \1", i)
    i = re.sub(r"[^a-zA-Z.!?]+", r" ", i)
    i = re.sub(r"\s+", r" ", i).strip()
    return i

def dictionaryCreater(file):
    rows = open(file,encoding="utf-8").read().strip().split("\n")
    pairs = [[stringNormalizer(i) for i in b.split('\t')] for b in rows]
    dic = dictionary.Dictionary()
    return dic,pairs

def pairFilter(pairs):
    return [pair for pair in pairs if len(pair[0].split(' ')) < MAX_LENGTH and len(pair[1].split(' ')) < MAX_LENGTH]

def rareWordCropper(dic,pairs,MIN_COUNT):
    dic.wordCropper(MIN_COUNT)
    pairs = []
    for pair in pairs:
        inputData = pair[0]
        outputData = pair[1]
        keepInput = True
        keepOutput = True
        for word in inputData.split(" "):
            if word not in dic.wordIndex:
                keepInput =  False
                break
        for word in outputData.split(" "):
            if word not in dic.wordIndex:
                keepOutput =  False
                break
        if keepInput and keepOutput:
            pairs.append(pair)

    return paris

def indexFromSentences(dic,sentence):
    return[dic.wordIndex[word] for word in sentence.split(" ")] + [EOS]

def binaryMatrixCreater(b,value=PAD):
    g = []
    for i,seq in enumerate(b):
        g.append([])
        for token in seq:
            if token == PAD:
                g[i].append(0)
            else:
                g[i].append(1)
    return g

def zeroPadding(b, fill=PAD):
    return list(itertools.zip_longest(*b,fillvalue=fill))

def dataPreparer(dataSubFolder,file):
    dic,pairs = dictionaryCreater(file,dataSubFolder)
    pairs = pairFilter(pairs)
    for pair in pairs:
        dic.wordsFromSentences(pair[0])
        dic.wordsFromSentences(pair[1])
    return dic,pairs

def inputVariable(i,dic):
    batchIndex = [indexFromSentences(dic,sentence) for sentence in i]
    paddingList = zeroPadding(batchIndex)
    paddingVariable = torch.LongTensor(paddingList)
    lengths = torch.tensor(len(index) for index in batchIndex)
    return paddingVariable, lengths

def outputVariable(i,dic):
    batchIndex = [indexFromSentences(dic,sentence) for sentence in i]
    paddingList = zeroPadding(batchIndex)
    mask = binaryMatrixCreate(paddingList)
    mask = torch.BoolTensor(mask)
    maxTargetLen = max([len(index) for index in batchIndex])
    paddingVariable = torch.LongTensor(paddingList)
    return paddingVariable,mask,maxTargetLen

def batchForTrain(dic,batch):
    batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    inputBatch, outputBatch = [],[]
    for b in batch:
        inputBatch.append(pair[0])
        outputBatch.append(pair[1])
    i,lengths = inputVariable(inputBatch,dic)
    output,mask,maxTargetLen = outputVariable(outputBatch,dic)
    return i,lengths,output,mask,maxTargetLen