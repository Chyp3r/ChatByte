PAD = 0 # Padding short sentence
SOS = 1 # Start of sentence
EOS = 2 # End of sentence

class Dictionary():
    def __init__(self):
        self.wordCount = {}
        self.wordIndex = {}
        self.wordFromIndex = {PAD:"PAD",SOS:"SOS",EOS:"EOS"}
        self.wordNum = 3 # Word count + token count
        self.isCroped = False

    def addWordToDict(self,word):
        if word not in self.wordIndex:
            self.wordCount[word] = 1
            self.wordIndex = self.wordNum
            self.wordFromIndex[self.wordNum] = word
            self.wordNum +=1
        else:
            self.wordCount[word] += 1
    
    def wordsFromSentences(self,sentence):
        for word in sentence.split(" "):
            self.addWordToDict(word)
    
    def wordCropper(self,min):
        if self.isCroped:
            return 
        words = []

        for i,b in self.wordCount.items():
            words.append(i)
        
        self.wordCount = {}
        self.wordIndex = {}
        self.wordFromIndex = {PAD:"PAD",SOS:"SOS",EOS:"EOS"}
        self.wordNum = 3
        
        for word in words:
            self.addWordToDict(word)