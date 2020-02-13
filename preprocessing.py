# -*- coding: utf-8 -*-
import os
import pandas as pd
from tensorflow.contrib import learn
import fasttext
import numpy as np
from sklearn.utils import shuffle
import re


# 内容
# - 利用 fasttext 对语料 text_corpus.txt 训练模型，保存 ft_skipgram_ws5_dim64.bin
# - 利用 tensorflow 对 questions.csv 进行word to ids 处理
class PreProcessing:
    def __init__(self, data_src):
        self.current_index = 0
        self.embeddings_model = fasttext.train_unsupervised("./data_repository/text_corpus.txt", model='skipgram',
                                                            lr=0.1, dim=64,
                                                            ws=5, epoch=50)
        self.embeddings_model.save_model("./model_siamese_network/ft_skipgram_ws5_dim64.bin")
        print('FastText training finished successfully.')

        # corpus
        print('corpus...')
        # 问题的相似对
        self.similar_pairs = self._build_corpus('./data_repository/questions.csv')
        print('shape', self.similar_pairs.shape)
        print('corpus.')
        input_X = list(self.similar_pairs['question1'])
        input_Y = list(self.similar_pairs['question2'])
        wc_list_x = list(len(x.split(' ')) for x in input_X)
        wc_list_y = list(len(x.split(' ')) for x in input_Y)
        wc_list = []
        wc_list.extend(wc_list_x)
        wc_list.extend(wc_list_y)
        number_of_elements = len(input_X)
        # 建立word到idx的映射关系
        # 1.首先将列表里面的词生成一个词典；
        # 2.按列表中的顺序给每一个词进行排序，每一个词都对应一个序号(从1开始，<UNK>的序号为0)
        # 3.按照原始列表顺序，将原来的词全部替换为它所对应的序号
        # 4.同时如果大于最大长度的词将进行剪切，小于最大长度的词将进行填充
        # 5.然后将其转换为列表，进而转换为一个array
        # 我们使用这些索引值做embedding，然后才能将数据转换成神经网络需要的格式
        # or use a constant like 16, select this parameter based on your understanding of what could be a good choice
        print('vocab_processor...')

        max_document_length = 16
        self.vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
        full_corpus = []
        full_corpus.extend(input_X)
        full_corpus.extend(input_Y)
        full_data = np.asarray(list(self.vocab_processor.fit_transform(full_corpus)))
        self.embeddings_lookup = []
        for word in list(self.vocab_processor.vocabulary_._mapping):
            try:
                self.embeddings_lookup.append(self.embeddings_model[str(word)])
            except:
                pass
        self.embeddings_lookup = np.asarray(self.embeddings_lookup)
        self.vocab_processor.save('./model_siamese_network/vocab')

        print('vocab_processor...', list(self.vocab_processor.vocabulary_._mapping)[:5])

        # metadata
        metadata_path = os.path.join('model_siamese_network', 'metadata.tsv')
        print('metadata_path', metadata_path)
        self._write_metadata(metadata_path, list(self.vocab_processor.vocabulary_._mapping))

        print('Vocab processor executed and saved successfully.')

        self.X = full_data[0:number_of_elements]
        self.Y = full_data[number_of_elements:2 * number_of_elements]
        self.label = list(self.similar_pairs['is_duplicate'])

    def _preprocess(self, x):
        try:
            tk_x = x.lower()

            # list of characters which needs to be replaced with space
            space_replace_chars = ['?', ':', ',', '"', '[', ']', '~', '*', ';', '!', '?', '(', ')', '{', '}', '@', '$',
                                   '#', '.', '-', '/']
            tk_x = tk_x.translate({ord(x): ' ' for x in space_replace_chars})

            non_space_replace_chars = ["'"]
            tk_x = tk_x.translate({ord(x): '' for x in non_space_replace_chars})

            # remove non-ASCII chars
            tk_x = ''.join([c if ord(c) < 128 else '' for c in tk_x])

            # replace all consecutive spaces with one space
            tk_x = re.sub('\s+', ' ', tk_x).strip()

            # find all consecutive numbers present in the word, first converted numbers to * to prevent conflicts while replacing with numbers
            regex = re.compile(r'([\d])')
            tk_x = regex.sub('*', tk_x)
            nos = re.findall(r'([\*]+)', tk_x)
            # replace the numbers with the corresponding count like 123 by 3
            for no in nos:
                tk_x = tk_x.replace(no, "<NUMBER>", 1)

            return tk_x.strip().lower()
        except:
            return ""

    def _build_corpus(self, filepath):
        similar_items = pd.read_csv(filepath)
        selected_cols = ['question1', 'question2', 'is_duplicate']
        similar_items = similar_items[selected_cols]
        similar_items['question1'] = similar_items['question1'].apply(self._preprocess)
        similar_items['question2'] = similar_items['question2'].apply(self._preprocess)
        similar_items = shuffle(similar_items)
        similar_items = similar_items.drop_duplicates()
        question_list = list(similar_items['question1'])
        question_list.extend(list(similar_items['question2']))
        pd.DataFrame(question_list).to_csv('./data_repository/text_corpus.txt', index=False)
        print('Text corpus generated and persisted successfully.')
        return similar_items

    def _write_metadata(self, filename, labels):
        with open(filename, 'w') as f:
            f.write("Index\tLabel\n")
            for index, label in enumerate(labels):
                f.write("{}\t{}\n".format(index, label))

        print('Metadata file saved in {}'.format(filename))

    def get_siamese_batch(self, n):
        last_index = self.current_index
        self.current_index += n
        return self.X[last_index: self.current_index, :], self.Y[last_index: self.current_index, :], np.expand_dims(
            self.label[last_index: self.current_index], axis=1)


PreProcessing('')
