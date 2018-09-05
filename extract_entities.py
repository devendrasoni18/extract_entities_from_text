
# coding: utf-8

# In[137]:


# Named Entity Recognization using Stanford pre-trained NER model of class 7
# cd /home/NLP/
# wget http://nlp.stanford.edu/software/stanford-ner-2015-04-20
# unzip stanford-ner-2015-04-20.zip
import nltk.tag.stanford as st
import os

# os.environ["CLASSPATH"] = '/home/devendra/NLP/stanford-corenlp-full-2018-02-27'
def space_out_punctuation(text):
    text = re.sub(r',\s', ' , ', text)
    text = re.sub(r'\.\.\.\s', ' ... ', text)
    text = re.sub(r'\.\s', ' . ', text)
    text = re.sub(r';\s', ' ; ', text)
    text = re.sub(r':\s', ' : ', text)
    text = re.sub(r'\?\s', ' ? ', text)
    text = re.sub(r'!\s', ' ! ', text)
    text = re.sub(r'"', ' " ', text)
    text = re.sub(r'\'', ' \' ', text)
    text = re.sub(r'\’', ' \' ', text)
    text = re.sub(r'\s\(', ' ( ', text)
    text = re.sub(r'\)\s', ' ) ', text)
    text = re.sub(r'\s\[', ' [ ', text)
    text = re.sub(r'\]\s', ' ] ', text)
    text = re.sub(r'-', ' - ', text)
    text = re.sub(r'_', ' _ ', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\r', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text


def get_continuous_chunks(tagged_sent):
    continuous_chunk = []
    current_chunk = []

    for token, tag in tagged_sent:
        if tag != "O":
            current_chunk.append((token, tag))
        else:
            if current_chunk: # if the current chunk is not empty
                continuous_chunk.append(current_chunk)
                current_chunk = []
    # Flush the final current_chunk into the continuous_chunk, if any.
    if current_chunk:
        continuous_chunk.append(current_chunk)
    return continuous_chunk

# Initialize stanford tagger model 
stner = st.StanfordNERTagger('/home/NLP/stanford-ner-2015-04-20/classifiers/english.muc.7class.distsim.crf.ser.gz',
                             '/home/NLP/stanford-ner-2015-04-20/stanford-ner.jar')

with open('data/extract_entities.txt', 'r') as f:
    text = f.read()
    text = space_out_punctuation(text)
    tagged_sent = stner.tag(text.split())

    named_entities = get_continuous_chunks(tagged_sent)
    named_entities_str_tag = [(" ".join([token for token, tag in ne]), ne[0][1]) for ne in named_entities]
    print(named_entities_str_tag)

