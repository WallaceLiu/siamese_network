# -*- coding: utf-8 -*-
import numpy as np
from tensorflow.contrib import learn

x_sentences = ['This is a cat', 'This must be boy', 'This is a a dog']
document_lengths = [len(x.split(" ")) for x in x_sentences]
max_document_length = max(document_lengths)

## Create the vocabularyprocessor object, setting the max lengh of the documents.
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)

## Transform the documents using the vocabulary.
x = np.array(list(vocab_processor.fit_transform(x_sentences)))

## Extract word:id mapping from the object.
vocab_dict = vocab_processor.vocabulary_._mapping

## Sort the vocabulary dictionary on the basis of values(id).
## Both statements perform same task.
# sorted_vocab = sorted(vocab_dict.items(), key=operator.itemgetter(1))
sorted_vocab = sorted(vocab_dict.items(), key=lambda x: x[1])

## Treat the id's as index into list and create a list of words in the ascending order of id's
## word with id i goes at index i of the list.
vocabulary = list(list(zip(*sorted_vocab))[0])

print('document_lengths', document_lengths)
print('max_document_length', max_document_length)
print('x', x)
print('vocab_dict', vocab_dict)
print('sorted_vocab', sorted_vocab)
print('vocabulary', vocabulary)

"""output

('document_lengths', [4, 4, 5])
('max_document_length', 5)
('x', array([[1, 2, 3, 4, 0],
       [1, 5, 6, 7, 0],
       [1, 2, 3, 3, 8]]))
('vocab_dict', {'a': 3, 'be': 6, 'boy': 7, 'This': 1, 'is': 2, 'dog': 8, 'cat': 4, '<UNK>': 0, 'must': 5})
('sorted_vocab', [('<UNK>', 0), ('This', 1), ('is', 2), ('a', 3), ('cat', 4), ('must', 5), ('be', 6), ('boy', 7), ('dog', 8)])
('vocabulary', ['<UNK>', 'This', 'is', 'a', 'cat', 'must', 'be', 'boy', 'dog'])

"""
