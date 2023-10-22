import numpy as np
import networkx as nx
# from graphviz import Digraph
import os
import re
import train

INPUT_PATH = './input/'
OUTPUT_PATH = './output/'
OUTPUT_ORIGINAL_SENTENCES_PATH = './output/result.txt'

'''
convert every dot file to original sentences and save in "originalSentences.txt"
'''
def dot2originalSentences():
    print("begin to collect the original sentences...")
    dotFileList = os.listdir(INPUT_PATH)
    
    '''
    special code
    '''
    originalSentences = []

    for numFolder in dotFileList:
        numFolderPath = INPUT_PATH + numFolder + '/'
        subFolderList = os.listdir(numFolderPath)
        for subFolder in subFolderList:
            subFolderPath = numFolderPath + subFolder + '/'
            javaFolderList = os.listdir(subFolderPath)
            for javaFolder in javaFolderList:
                if javaFolder[-4:] == '.txt':
                    continue
                elif not os.path.exists(subFolderPath + javaFolder + '/sootOutput'):
                    continue
                else:
                    dotAndClassFileList = os.listdir(subFolderPath + javaFolder + '/sootOutput/')
                    for ind in range(len(dotAndClassFileList)):
                        print("collecting", subFolderPath + javaFolder + '/sootOutput/' + dotAndClassFileList[ind], "...(", ind + 1, "/", len(dotAndClassFileList), ")")
                        if dotAndClassFileList[ind][-4:] != '.dot':
                            print("#########break_break_break_break#########")
                            continue
                        else:
                            try:
                                g = nx.drawing.nx_pydot.read_dot(subFolderPath + javaFolder + '/sootOutput/' + dotAndClassFileList[ind])
                            except:
                                with open(OUTPUT_PATH + 'failedDot.txt', 'a') as f:
                                    f.write(subFolderPath + javaFolder + '/sootOutput/' + dotAndClassFileList[ind] + '\n')
                                    f.close()
                                continue
                            dic = nx.get_node_attributes(g, 'label')
                            with open(OUTPUT_ORIGINAL_SENTENCES_PATH, 'a') as f:
                                for key, value in dic.items():
                                    originalSentences.append(value[1:-1])
                                    f.write(value[1:-1] + '\n')
                                f.close()
    print("done")
    return originalSentences

def method2ReadFromCorpus():
    originalSentences = []
    with open(OUTPUT_ORIGINAL_SENTENCES_PATH, 'r') as f:
        lines = f.readlines()
        for line in lines:
            originalSentences.append(line[:-1])
    return originalSentences

'''
split the original sentences and set up corpus
'''
def getCorpus(originalSentences):
    print("begin to set up corpus...")
    sentences = []
    for sentence in originalSentences:
        wordList = []
        # tempOutput = ''
        '''
        you can split word in your way
        '''
        # sentence = re.sub(r'\".*\"', 'string_text', sentence)
        # sentence = re.sub(r'\'.*\'', 'string_text', sentence)
        # for word in re.split(r'[ :()\[\]{}.,\"\']', sentence):
        for word in re.split(' ', sentence):
            if word != '':
                wordList.append(word)
                # tempOutput = tempOutput + word + ' '
        # print(wordList)
        sentences.append(wordList)
        # with open('./corpus.txt', 'a') as f:
        #     f.write(tempOutput + '\n')
        #     f.close()   
    # sentencesMat = np.array(sentences)[:, np.newaxis]
    print("done")
    return sentences

def main():
    '''
    method1: set up corpus
    '''
    # originalSentences = dot2originalSentences()
    '''
    method2: read corpus from text
    '''
    originalSentences = method2ReadFromCorpus()

    sentences = getCorpus(originalSentences)
    train.train(sentences)
    train.loadModel("./output/trained_model.model")

if __name__ == "__main__":
    main()
