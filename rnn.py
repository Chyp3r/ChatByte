import torch.nn as nn
import torch.nn.functional as F
from dictionary import SOS

class EncoderRNN(nn.Module):
    def __init__(self,hiddenSize,embedding,nLayers = 1,dropout = 0):
        super(EncoderRNN,self).__init__()
        self.nLayers = nLayers
        self.hiddenSize = hiddenSize
        self.embedding = embedding
        self.gru = nn.GRU(hiddenSize,hiddenSize,nLayers,dropout=(0 if nLayers == 1 else dropout),bidirectional = True)

    def forward(self,inputSeq,inputLen,hidden= None):
        embedded = self.embedding(inputSeq)
        packed == nn.utils.rnn.pack_padded_sequence(embedded,inputLen)
        outputs, hidden = self.gru(packed,hidden)
        outputs,_ = nn.utils.rnn.pad_packed_sequence(outputs)
        outputs = outputs[:,:,:self.hiddenSize] + outputs[:,:,:self.hiddenSize]
        
        return outputs,hidden

class Attn(nn.Module):
    def __init__(self,method,hiddenSize):
        super(Attn,self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hiddenSize = hiddenSize
        if self.method == 'general':
            self.attn = nn.Linear(self.hiddenSize, hiddenSize)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hiddenSize * 2, hiddenSize)
            self.v = nn.Parameter(torch.FloatTensor(hiddenSize))
    
    def dotScore(self, hidden, encoderOutput):
        return torch.sum(hidden * encoderOutput,dim=2)

    def generalScore(self,hidden,encoderOutput):
        return torch.sum(hidden*self.attn(encoderOutput),dim = 2)
    
    def concatScore(self,hidden,encoderOutput):
        return torch.sum(self.v *self.attn(torch.cat((hidden.expand(encoderOutput.size(0), -1, -1), encoderOutput), 2)).tanh(),dim = 2)
    
    def forward(self,hidden,encoderOutput):
        if self.method == "general":
            attn_energies = self.generalScore(hidden,encoderOutput)
        elif self.method == "concat":
            attn_energies = self.concatScore(hidden,encoderOutput)
        elif self.method == "dot":
            attn_energies = self.dotScore(hidden,encoderOutput)
        
        attn_energies = attn_energies.t()

        return F.softmax(attn_energies,dim=1).unsqueeze(1)

class LuongAttnDecoderRNN(nn.Module):
    def __init__(self,attnModel,embedding,hiddenSize,outputSize,nLayers = 1,dropout=0.1):
        super(LuongAttnDecoderRNN, self).__init__()

        self.attnModel = attnModel
        self.hiddenSize = hiddenSize
        self.outputSize = outputSize
        self.nLayers = nLayers
        self.dropout = dropout

        self.embedding = embedding
        self.embeddingDropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hiddenSize,hiddenSize,nLayers,dropout = (0 if nLayers == 1 else dropout))
        self.concat = nn.Linear(hiddenSize*2,hiddenSize)
        self.out = nn.Linear(hiddenSize,outputSize)

        self.attn = Attn(attnModel,hiddenSize)

    def forward(self,inputStep,lastHidden,encoderOutput):
        embedded = self.embedding(inputStep)
        embedded = self.embeddingDropout(embedded)

        rnnOutput,hidden = self.gru(embedded,lastHidden)
        attnWeights = self.attn(rnnOutput,encoderOutput)
        context = attnWeights.bmm(encoderOutput.transpose(0,1))
        
        rnnOutput = rnnOutput.squeeze(0)
        context = context.squeeze(1)
        concatInput = torch.cat((rnnOutput,context),1)
        concatOutput = torch.tanh(self.concat(concatInput))

        output = self.out(concatOutput)
        output = F.softmax(output,dim = 1)

        return output, hidden
    
class GreedySearchDecoder(nn.Module):
    def __init__(self, encoder,decoder):
        super(GreedySearchDecoder,self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self,inputSeq,inputLen,maxLen):
        encoderOutput, encoderHidden = self.encoder(inputSeq,inputLen)
        decoderHidden = encoderHidden[decoder.nLayers]
        decoderInput = torch.ones(1,1,device = device , dtype=torch.long)*SOS
        allTokens = torch.zeros([0], device=device, dtype=torch.long)
        allScores = torch.zeros([0], device=device)

        for _ in range(maxLen):
            decoderOutput, decoderHidden = self.decoder(decoderInput,decoderHidden,encoderOutput)
            decoderScores,decoderInput = torch.max(decoderOutput,dim=1)
            allTokens = torch.cat((allTokens, decoderInput), dim=0)
            allScores = torch.cat((allScores, decoderScores), dim=0)
            decoderInput = torch.unsqueeze(decoderInput,0)
        
        return allTokens,allScores
