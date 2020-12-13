import sys
from nltk.corpus import brown
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
import re
import itertools


# Load the Brown corpus with Universal Dependencies tags
# proportion is a float
# Returns a tuple of lists (sents, tags)
tag_dict = {}
def load_training_corpus(proportion=1.0):
    brown_sentences = brown.tagged_sents(tagset='universal')
    num_used = int(proportion * len(brown_sentences))

    corpus_sents, corpus_tags = [None] * num_used, [None] * num_used
    for i in range(num_used):
        corpus_sents[i], corpus_tags[i] = zip(*brown_sentences[i])
    return (corpus_sents, corpus_tags)

# Generate word n-gram features
# words is a list of strings
# i is an int
# Returns a list of strings

def get_ngram_features(words, i):
  list_ngram_features = list()
  list_len = len(words)
  
  #### checking for prevbigram  #######
  prevbigram = ""
  if i == 0:
    prevbigram = "<s>"
  else :
    prevbigram = words[i-1]
  list_ngram_features.append("prevbigram" + "-" + prevbigram)
  
  ### checking the next bigram ########
  nextbigram = ""
  if i+1 == list_len:
    nextbigram = "</s>"
  else:
    nextbigram = words[i+1]
    
  list_ngram_features.append("nextbigram" + "-" + nextbigram)
  
  ### checking for the prev skip ######
  prevskip = ""
  if i < 2:
    prevskip = "<s>"
  else:
    prevskip = words[i-2]
  
  list_ngram_features.append("prevskip" + "-" + prevskip)
  
  ### checking fro the next skip ######
  nextskip = ""
  if i+2 >= len(words):
    nextskip = "</s>"
  else:
    nextskip = words[i+2]
    
  list_ngram_features.append("nextskip" + "-" + nextskip)
  
  ### checking for prevtrigram #######
  prevtrigram1 = ""
  prevtrigram2 = ""
  if i == 0:
    prevtrigram1 = "<s>"
    prevtrigram2 = "<s>"
  elif i == 1:
    prevtrigram1 = words[i-1]
    prevtrigram2 = "<s>"
  else:
    prevtrigram1 = words[i-1]
    prevtrigram2 = words[i-2]
  
  list_ngram_features.append("prevtrigram" + "-" + prevtrigram1 + "-" + prevtrigram2)
  
  ### checking for nexttrigram ######
  nexttrigram1 = ""
  nexttrigram2 = ""
  if (i + 1) >= len(words):
    nexttrigram1 = "</s>"
    nexttrigram2 = "</s>"
  elif (i+1) >= len(words) - 1:
    nexttrigram1 = words[i+1]
    nexttrigram2 = "</s>"
  else:
    nexttrigram1 = words[i+1]
    nexttrigram2 = words[i+2]
    
  list_ngram_features.append("nexttrigram" + "-" + nexttrigram1 + "-" + nexttrigram2)
  
  ### checking for center trigram ###
  centertrigram1 = ""
  centertrigram2 = ""
  if len(words) == 1:
    centertrigram1 = "<s>"
    centertrigram2 = "</s>"
  elif i == 0:
    centertrigram1 = "<s>"
    centertrigram2 = words[i+1]
  elif (i+1) >= len(words):
    centertrigram1 = words[i-1]
    centertrigram2 = "</s>"
  else:
    centertrigram1 = words[i-1]
    centertrigram2 = words[i+1]
  
  list_ngram_features.append("centertrigram" + "-" + centertrigram1 + "-" + centertrigram2)
  
  return list_ngram_features

def get_word_features(word):
    word_features_list = list()
    
    word_features_list.append("word-" + word)

    if word[0].isupper():
        word_features_list.append("capital")

    if word.isupper():
        word_features_list.append("allcaps")
    
    if any(ch.isdigit() for ch in word):
        word_features_list.append("number")
    
    if '-' in word:
        word_features_list.append("hyphen")

    stri = ""
    for ch in word:
        if ch.islower():
            stri+="x"
        elif ch.isupper():
            stri+="X"
        elif ch.isdigit():
            stri+="d"
        else:
            stri+=ch
    word_features_list.append("wordshape-"+stri)

    nstr = ""
    i=0
    while i<len(stri):
        curr = stri[i]
        nstr+=stri[i]
        while i<len(stri) and stri[i]==curr:
            i+=1
    word_features_list.append("short-wordshape-"+nstr)

  
    j = 1 
    while(j <= 4) and j <= len(word):
        word_features_list.append("prefix" + str(j) + "-" + word[0:j])
        j = j + 1
  
    j = 1
    while(j <= 4 and j <= len(word)):
        word_features_list.append("suffix" + str(j) + "-" + word[-1*j:])
        j = j + 1
    
    return word_features_list

def get_features(words, i, prevtag):
  list_word_ngram_features = get_ngram_features(words,i)
  list_word_features = get_word_features(words[i])
  features = list_word_ngram_features + list_word_features
  features.append("tagbigram" + "-" + prevtag)
  lower_features = list(map(lambda x: x if x.startswith("wordshape") or x.startswith("short-wordshape") else x.lower(),features))
  return lower_features

# Remove features that occur fewer than a given threshold number of time
# corpus_features is a list of lists, where each sublist corresponds to a sentence and has elements that are lists of strings (feature names)
# threshold is an int
# Returns a tuple (corpus_features, common_features)
def remove_rare_features(corpus_features, threshold=5):
  n = threshold
  features = corpus_features
  feature_vocabulary = {}
  
  for sentence_feature in features:
    for word_features in sentence_feature:
      for feature in word_features:
        feature_vocabulary[feature] = feature_vocabulary.get(feature,0) + 1
                           
  ###### creating two sets for rare and non rare feature vocab #############
  rare_features = set()
  non_rare_features = set()
  
  for feature,feature_count in feature_vocabulary.items():
    if feature_count < n:
      rare_features.add(feature)
    else:
      non_rare_features.add(feature)
      
  ########## removing the rare features from the training features set ############
  updated_training_features = list()

  for sentence_feature in features:
      word_features_list = list()
      for word_features in sentence_feature:
          word_list = list()
          for word_feature in word_features:
              if word_feature not in rare_features:
                  word_list.append(word_feature)
          word_features_list.append(word_list)
      updated_training_features.append(word_features_list)        

  return (updated_training_features,non_rare_features)

# Build feature and tag dictionaries
# common_features is a set of strings
# corpus_tags is a list of lists of strings (tags)
# Returns a tuple (feature_dict, tag_dict)

def get_feature_and_label_dictionaries(common_features, corpus_tags):
    feature_dict = {}
    tag_dict={}
    index = 0
    for feature_word in common_features:
        if feature_word not in feature_dict:
            feature_dict[feature_word] = index
            index = index + 1

  #### creating tag dictionary where the keys are the 17 tags and the values are the indices assigned to the tag ###
  
    tag_idx = 0
    for taglist in corpus_tags:
        for tag in taglist:
            if tag not in tag_dict:
                tag_dict[tag] = tag_idx
                tag_idx = tag_idx + 1

    return (feature_dict,tag_dict)



# Build the label vector Y
# corpus_tags is a list of lists of strings (tags)
# tag_dict is a dictionary {string: int}
# Returns a Numpy array
def build_Y(corpus_tags, tag_dict):
    Y = list()
    i = 0
    for sentence_tag in corpus_tags:
        for word_tag in sentence_tag:
            if word_tag in tag_dict:
              Y.append(tag_dict[word_tag])
            else:
              tag_dict[word_tag] = i
              Y.append(i)
              i+=1

    return np.array(Y)


# Build a sparse input matrix X
# corpus_features is a list of lists, where each sublist corresponds to a sentence and has elements that are lists of strings (feature names)
# feature_dict is a dictionary {string: int}
# Returns a Scipy.sparse csr_matrix
def build_X(corpus_features, feature_dict):
  examples = []
  feature_index = []
  i=0

  for sentence_features in corpus_features:
    for word_features in sentence_features:
      for feature in word_features:
        if feature in feature_dict:
          examples.append(i)
          feature_index.append(feature_dict[feature])
        
      i += 1

  values = [1] * len(examples)

  examples = np.asarray(examples)
  feature_index = np.asarray(feature_index)
  values = np.asarray(values)

  sparse_matrix = csr_matrix((values, (examples, feature_index)), shape = (i, len(feature_dict)), dtype = np.int32)
  return sparse_matrix


# Train an MEMM tagger on the Brown corpus
# proportion is a float
# Returns a tuple (model, feature_dict, tag_dict)
def train(proportion=1.0):
    corpus_sents = load_training_corpus(0.05)
    train_sentences = list()
    train_tags = list()
      
    for tup in corpus_sents[0]:
      sentence = ""
      for word in tup:
        sentence = sentence + word + " "
      train_sentences.append(sentence)

    for tupl in corpus_sents[1]:
      label_list = list()
      for tags in tupl:
        label_list.append(tags)
      train_tags.append(label_list)
    training_features = list()
  
    for train_sentence in train_sentences:
        indx = train_sentences.index(train_sentence)
        word_feature_list = list()
        i = 0
        for word in train_sentence.split():
            if i == 0:
                prev_tag = '<S>'
            else:
                prev_tag = train_tags[indx][i-1]
            word_feature = get_features(train_sentence,i,prev_tag)
            word_feature_list.append(word_feature)
            i = i + 1
            
        training_features.append(word_feature_list)

    remove_rare_output = remove_rare_features(training_features,5)
    training_features_updated = remove_rare_output[0]
    non_rare_set = remove_rare_output[1]
    tag_vocabulary = list(x for l in train_tags for x in l)
    feature_dict,tag_dict = get_feature_and_label_dictionaries(non_rare_set,tag_vocabulary)

    Y_train = build_Y(train_tags, feature_dict)
    X_train = build_X(training_features_updated,tag_dict)
    print(X_train)
    model = LogisticRegression(class_weight='balanced',solver='saga',multi_class='multinomial').fit(X_train, Y_train) 
    td = tag_dict
    return (model, feature_dict, tag_dict)



# Load the test set
# corpus_path is a string
# Returns a list of lists of strings (words)
def load_test_corpus(corpus_path):
    with open(corpus_path) as inf:
        lines = [line.strip().split() for line in inf]
    return [line for line in lines if len(line) > 0]


# Predict tags for a test sentence
# test_sent is a list containing a single list of strings
# model is a trained LogisticRegression
# feature_dict is a dictionary {string: int}
# reverse_tag_dict is a dictionary {int: string}
# Returns a tuple (Y_start, Y_pred)

def get_predictions(test_sent, model, feature_dict, reverse_tag_dict):
  return (0,0)
  n = len(test_sent)
  T = len(tag_dict)
  Y_pred = np.empty((n-1,T,T))
  Y_start = np.empty((1, T))
  index = 0
  for word in test_sent:
    if index == 0:
        X = build_X(get_features(test_sent, index, "<S>"),feature_dict)
        Y_start = model.predict_log_proba(X)
    else:
        features = []
        for prev_tag in tag_dict.keys():
            feat = get_features(test_sent, index, prev_tag)
            features.append(feat)
            j = tag_dict[prev_tag]
            X = build_X(features,feature_dict)
            model.predict_log_proba(X)
            Y_pred[index-1][j] = model.predict_log_proba(X)
    index += 1
  return (Y_start,Y_pred)


# Perform Viterbi decoding using predicted log probabilities
# Y_start is a Numpy array of size (1, T)
# Y_pred is a Numpy array of size (n-1, T, T)
# Returns a list of strings (tags)
def viterbi(Y_start, Y_pred):
    return [1,2,3,4,5,6,7,8,9,10]
    N = np.shape(Y_pred)[0] + 1
    T = len(tag_dict)
    V = np.empty((N, T))
    BP = np.empty((N, T))

    for j in range(T):
        V[0][j] = Y_start[0][j]
        BP[0][j] = -1

    for i, row in enumerate(Y_pred):
            for k in range(T):
                sum = V[i, :] + Y_pred[i, :, k]
                V[i + 1, k] = max(sum)
                BP[i + 1, k] = int(np.argmax(sum))
    backward_indices = []
    index = np.argmax(V[N-1])
    backward_indices.append(index)
    for n in range(N - 1, 0, -1):
        index = BP[n, int(index)]
        backward_indices.append(index)
    
    for key,value in tag_dict.items():
        i = 0
        for bindex in backward_indices:
            if bindex == value:
                backward_indices[i] = key
            i = i + 1
    



# Predict tags for a test corpus using a trained model
# corpus_path is a string
# model is a trained LogisticRegression
# feature_dict is a dictionary {string: int}
# tag_dict is a dictionary {string: int}
# Returns a list of lists of strings (tags)
def predict(corpus_path, model, feature_dict, tag_dict):
    pass
            

def main(args):
    model, feature_dict, tag_dict = train(0.25)

    predictions = predict('test.txt', model, feature_dict, tag_dict)
    for test_sent in predictions:
        print(test_sent)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
