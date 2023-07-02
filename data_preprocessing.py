import json
import csv
import os
import unicodedata
import codecs
from io import open

# Take important datas for lines and conversations from data
def convLoader(file):
    print("Data loading...")
    convs = {} # conversation dic
    rows = {} # every row from data
    with open(file,"r",encoding='iso-8859-1') as f:
        for row in f:
            rowFromJson = json.loads(row)
            tempRow = {}

            tempRow["lineID"] = rowFromJson["id"]
            tempRow["characterID"] = rowFromJson["speaker"]
            tempRow["text"] = rowFromJson["text"]
            rows[tempRow["lineID"]] = tempRow

            if rowFromJson["conversation_id"] not in convs:
                tempConv = {}
                tempConv["conversationID"] = rowFromJson["conversation_id"]
                tempConv["movieID"] = rowFromJson["meta"]["movie_id"]
                tempConv["lines"] = [tempRow]
            
            else:
                tempConv = convs[rowFromJson["conversation_id"]]
                tempConv["lines"].insert(0,tempRow)
            
            convs[tempConv["conversationID"]] = tempConv

    return rows,convs

# Find questions and answers from data
def quesitonAnswerFinder(conversations):
    print("QA couples finding...")
    QAs= [] # questions and answers
    for conv in conversations.values():
        for i in range(len(conv["lines"]) - 1):
            # Take questions and answers
            inputRow = conv["lines"][i]["text"].strip()
            targetRow = conv["lines"][i+1]["text"].strip()
            if inputRow and targetRow:
                QAs.append([inputRow,targetRow])
    
    return QAs

# Create txt based data file
def createConvFile(file):
    print("Data saving...")
    with open(file,"w",encoding="utf-8") as save:
        for i in quesitonAnswerFinder(convs):
            csv.writer(save,delimiter = str(codecs.decode("\t", "unicode_escape")),lineterminator="\n").writerow(i)

# Print saved data
def printData(file,rowCount = 5):
    with open(file,"rb") as data:
        rows = data.readlines()
    for row in rows[:rowCount]:
        print(row)
