# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 16:11:18 2017

@author: 代码医生 qq群：40016981，公众号：xiangyuejiqiren
@blog：http://blog.csdn.net/lijin6249
"""

import sys
import os
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt


import numpy as np

from tensorflow.python.platform import gfile
from random import shuffle
#from multiprocessing import Process, Lock
#import time
#from math import floor
#import os
import re 

#获取文件列表
def getRawFileList( path):
    files = []
    names = []
    for f in os.listdir(path):
        if not f.endswith("~") or not f == "":
            files.append(os.path.join(path, f))
            names.append(f)
    return files,names
#读取分词后的中文词
def get_ch_lable(txt_file,Isch=True,normalize_digits=False):  
    labels= list()#""
    labelssz = []
    with open(txt_file, 'rb') as f:
        for label in f: 
            linstr1 =label.decode('utf-8')
            #labels =label.decode('gb2312').split()
            #linstr1 = label.decode('gb2312')
            if normalize_digits :
                linstr1=re.sub('\d+',_NUM,linstr1)
            notoken = basic_tokenizer(linstr1 )
            if Isch:
                notoken = fenci(notoken)
            else:
                notoken = notoken.split()
            #labels =labels+notoken_ci#label.decode('gb2312')
            labels.extend(notoken)
            labelssz.append(len(labels))
    return  labels,labelssz
   
    
    
#获取文件文本
def get_ch_path_text(raw_data_dir,Isch=True,normalize_digits=False):
    text_files,_ = getRawFileList(raw_data_dir)
    labels = []
    
    training_dataszs = list([0])
    #np.reshape(training_dataszs,(1,-1))
    if len(text_files)== 0:
        print("err:no files in ",raw_data_dir)
        return labels
    print(len(text_files),"files,one is",text_files[0])
    shuffle(text_files)
    
    for text_file in text_files:
        training_data,training_datasz =get_ch_lable(text_file,Isch,normalize_digits)
        
#        notoken = basic_tokenizer(training_data)
#        notoken_ci = fenci(notoken)
        training_ci = np.array(training_data)
        training_ci = np.reshape(training_ci, [-1, ])
        labels.append(training_ci)
        
        training_datasz =np.array( training_datasz)+training_dataszs[-1]
        training_dataszs.extend(list(training_datasz))
        print("here",training_dataszs)
    return labels,training_dataszs
    
       
def basic_tokenizer(sentence):    
    _WORD_SPLIT = "([.,!?\"':;)(])"
    _CHWORD_SPLIT = '、|。|，|‘|’'
    str1 = ""
    for i in re.split(_CHWORD_SPLIT,  sentence):
        str1 = str1 +i
    str2 = ""
    for i in re.split(_WORD_SPLIT ,  str1):
        str2 = str2 +i
    return str2

import jieba
jieba.load_userdict("myjiebadict.txt")

def fenci(training_data):
    seg_list = jieba.cut(training_data)  # 默认是精确模式  
    training_ci = " ".join(seg_list)
    training_ci = training_ci.split()
    #以空格将字符串分开
    #training_ci = np.array(training_ci)
    #training_ci = np.reshape(training_ci, [-1, ])
    return training_ci



import collections
#系统字符，创建字典是需要加入
_PAD = "_PAD"
_GO = "_GO"
_EOS = "_EOS"
_UNK = "_UNK"

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

#文字字符替换，不属于系统字符
_NUM = "_NUM"
#Isch=true 中文， false 英文
 #创建词典 max_vocabulary_size=500 500个词  
def create_vocabulary(vocabulary_file, raw_data_dir, max_vocabulary_size,Isch=True, normalize_digits=True):
    texts,textssz = get_ch_path_text(raw_data_dir,Isch,normalize_digits)
    print( texts[0],len(texts)) 
    print("行数",len(textssz),textssz)
# texts ->
    all_words = []  
    for label in texts:  
        print("词数",len(label))   
        all_words += [word for word in label]     
    print("词数",len(all_words))
    
    training_label, count, dictionary, reverse_dictionary = build_dataset(all_words,max_vocabulary_size)
    print("reverse_dictionary",reverse_dictionary,len(reverse_dictionary))
    if not gfile.Exists(vocabulary_file):
        print("Creating vocabulary %s from data %s" % (vocabulary_file, data_dir))
        if len(reverse_dictionary) > max_vocabulary_size:
            reverse_dictionary = reverse_dictionary[:max_vocabulary_size]
        with gfile.GFile(vocabulary_file, mode="w") as vocab_file:
            for w in reverse_dictionary:
                print(reverse_dictionary[w])
                vocab_file.write(reverse_dictionary[w] + "\n")
    else:
        print("already have vocabulary!  do nothing !!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    return training_label, count, dictionary, reverse_dictionary,textssz



def build_dataset(words, n_words):
  """Process raw inputs into a dataset."""
  count = [[_PAD, -1],[_GO, -1],[_EOS, -1],[_UNK, -1]]
  count.extend(collections.Counter(words).most_common(n_words - 1))
  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)
  data = list()
  unk_count = 0
  for word in words:
    if word in dictionary:
      index = dictionary[word]
    else:
      index = 0  # dictionary['UNK']
      unk_count += 1
    data.append(index)
  count[0][1] = unk_count
  reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  return data, count, dictionary, reversed_dictionary
    
#把data中的内存问和答ids数据  放在不同的文件里    
def create_seq2seqfile(data ,sorcefile,targetfile,textssz):
    print("data",data,len(data))
    with open(sorcefile,'w') as sor_f:
        with open(targetfile,'w') as tar_f:
            for i in range(len(textssz)-1):
                print("textssz",i,textssz[i],textssz[i+1],data[textssz[i]:textssz[i+1]])                
                if (i+1)%2:
                    sor_f.write(str(data[textssz[i]:textssz[i+1]]).replace(',',' ')[1:-1]+'\n')
                else:
                    tar_f.write(str(data[textssz[i]:textssz[i+1]]).replace(',',' ')[1:-1]+'\n')


def plot_scatter_lengths(title, x_title, y_title, x_lengths, y_lengths):
	plt.scatter(x_lengths, y_lengths)
	plt.title(title)
	plt.xlabel(x_title)
	plt.ylabel(y_title)
	plt.ylim(0, max(y_lengths))
	plt.xlim(0,max(x_lengths))
	plt.show()

def plot_histo_lengths(title, lengths):
	mu = np.std(lengths)
	sigma = np.mean(lengths)
	x = np.array(lengths)
	n, bins, patches = plt.hist(x,  50, facecolor='green', alpha=0.5)
	y = mlab.normpdf(bins, mu, sigma)
	plt.plot(bins, y, 'r--')
	plt.title(title)
	plt.xlabel("Length")
	plt.ylabel("Number of Sequences")
	plt.xlim(0,max(lengths))
	plt.show()




#将读好的对话文本按行分开，一行问，一行答。存为两个文件。training_data为总数据，textssz为每行的索引
def splitFileOneline(training_data ,textssz):
    source_file = os.path.join(data_dir+'fromids/', "data_source_test.txt")
    target_file = os.path.join(data_dir+'toids/', "data_target_test.txt")
    create_seq2seqfile(training_data,source_file ,target_file,textssz)



def analysisfile(source_file,target_file):
#分析文本    
    source_lengths = []
    target_lengths = []

    with gfile.GFile(source_file, mode="r") as s_file:
        with gfile.GFile(target_file, mode="r") as t_file:
            source= s_file.readline()
            target = t_file.readline()
            counter = 0
            
            while source and target:
                counter += 1
                if counter % 100000 == 0:
                    print("  reading data line %d" % counter)
                    sys.stdout.flush()
                num_source_ids = len(source.split())
                source_lengths.append(num_source_ids)
                num_target_ids = len(target.split()) + 1#plus 1 for EOS token
                target_lengths.append(num_target_ids)
                source, target = s_file.readline(), t_file.readline()
    print(target_lengths,source_lengths)
    if plot_histograms:
        plot_histo_lengths("target lengths", target_lengths)
        plot_histo_lengths("source_lengths", source_lengths)
    if plot_scatter:
        plot_scatter_lengths("target vs source length", "source length","target length", source_lengths, target_lengths)


def initialize_vocabulary(vocabulary_path):
  if gfile.Exists(vocabulary_path):
    rev_vocab = []
    #with gfile.GFile(vocabulary_path, mode="rb") as f:
    with gfile.GFile(vocabulary_path, mode="r") as f:
      rev_vocab.extend(f.readlines())
    rev_vocab = [line.strip() for line in rev_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
    return vocab, rev_vocab
  else:
    raise ValueError("Vocabulary file %s not found.", vocabulary_path)

#将句子转成ids
def sentence_to_ids(sentence, vocabulary,
                           normalize_digits=True,Isch=True):

    if normalize_digits :
        sentence=re.sub('\d+',_NUM,sentence)
    notoken = basic_tokenizer(sentence )
    if Isch:
        notoken = fenci(notoken)
    else:
        notoken = notoken.split()
    #print("notoken",notoken)
    idsdata = [vocabulary.get( w, UNK_ID) for w in notoken]
    #print("data",idsdata)
    return idsdata


#将一个文件转成ids 不是windows下的要改编码格式 utf8
def textfile_to_idsfile(data_file_name, target_file_name, vocab,
                       normalize_digits=True,Isch=True):
  
  if not gfile.Exists(target_file_name):
    print("Tokenizing data in %s" % data_file_name)
    with gfile.GFile(data_file_name, mode="rb") as data_file:
      with gfile.GFile(target_file_name, mode="w") as ids_file:
        counter = 0
        for line in data_file:
          counter += 1
          if counter % 100000 == 0:
            print("  tokenizing line %d" % counter)
          #token_ids = sentence_to_ids(line.decode('gb2312'), vocab,normalize_digits,Isch)
          token_ids = sentence_to_ids(line.decode('utf8'), vocab,normalize_digits,Isch)
          ids_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")

#将文件批量转成ids文件
def textdir_to_idsdir(textdir,idsdir,vocab, normalize_digits=True,Isch=True):
    text_files,filenames = getRawFileList(textdir)
    #np.reshape(training_dataszs,(1,-1))
    if len(text_files)== 0:
        raise ValueError("err:no files in ",raw_data_dir)
        
    print(len(text_files),"files,one is",text_files[0])
    
    for text_file,name in zip(text_files,filenames):
        print(text_file,idsdir+name)
        textfile_to_idsfile(text_file,idsdir+name,vocab, normalize_digits,Isch)


def ids2texts( indices,rev_vocab):
    texts = []
    for index in indices:
        #texts.append(rev_vocab[index].decode('ascii'))
        texts.append(rev_vocab[index])
    return texts



data_dir = "fanyichina/"
raw_data_dir = "fanyichina/yuliao/from"
raw_data_dir_to = "fanyichina/yuliao/to"
vocabulary_fileen ="dicten.txt"
vocabulary_filech = "dictch.txt"


plot_histograms = plot_scatter =True
vocab_size =40000


max_num_lines =1
max_target_size = 200
max_source_size = 200



def main():
    vocabulary_filenameen = os.path.join(data_dir, vocabulary_fileen)
    vocabulary_filenamech = os.path.join(data_dir, vocabulary_filech)
##############################
    创建英文字典
    training_dataen, counten, dictionaryen, reverse_dictionaryen,textsszen =create_vocabulary(vocabulary_filenameen
                                                            ,raw_data_dir,vocab_size,Isch=False,normalize_digits = True)
    print("training_data",len(training_dataen))
    print("dictionary",len(dictionaryen)) 
#########################
    #创建中文字典    
    training_datach, countch, dictionarych, reverse_dictionarych,textsszch =create_vocabulary(vocabulary_filenamech
                                                      ,raw_data_dir_to,vocab_size,Isch=True,normalize_digits = True)
    print("training_datach",len(training_datach))
    print("dictionarych",len(dictionarych)) 
#############################    
    vocaben, rev_vocaben =initialize_vocabulary(vocabulary_filenameen)
    vocabch, rev_vocabch =initialize_vocabulary(vocabulary_filenamech)

    print(len(rev_vocaben))
    textdir_to_idsdir(raw_data_dir,data_dir+"fromids/",vocaben,normalize_digits=True,Isch=False)
    textdir_to_idsdir(raw_data_dir_to,data_dir+"toids/",vocabch,normalize_digits=True,Isch=True)

##########################分析
    filesfrom,_=getRawFileList(data_dir+"fromids/")
    filesto,_=getRawFileList(data_dir+"toids/")
    source_train_file_path = filesfrom[0]
    target_train_file_path= filesto[0]    
    analysisfile(source_train_file_path,target_train_file_path)

    

if __name__=="__main__":
	main()