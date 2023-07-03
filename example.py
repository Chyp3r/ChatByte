import torch
import data_preprocessing as dp 
import dictionary 
import helpers
import rnn
import trainer
import os 
import codecs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataSubFolder = "moviecorpus"
dataMainFolder = os.path.join("data",dataSubFolder)

datafile = os.path.join(dataMainFolder,"savedData")

delimiter = str(codecs.decode("\t","unicode_escape"))

rows = {}
conversations = {}

rows, conversations = dp.convLoader(os.path.join(dataMainFolder,"utterances.jsonl"))

dp.createConvFile(datafile,conversations)
dp.printData(datafile)
