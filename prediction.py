# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from scipy.spatial.distance import cdist
import os
import time
from preprocessing import PreProcessing
from model import SiameseNetwork
import pandas as pd
from tensorflow.contrib import learn
import fasttext
import faiss
import re

model_path = './model_siamese_network/'

question_pairs = pd.read_csv('./data_repository/questions.csv')
question_pairs.fillna("", inplace=True)

selected_cols = ['question1', 'question2', 'is_duplicate']
question_pairs = question_pairs[selected_cols]


def preprocess(x):
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


# 预处理
# 1, question1列预处理
# 2, question2列预处理
# 3, 删除重复行
question_pairs['question1'] = question_pairs['question1'].apply(preprocess)
question_pairs['question2'] = question_pairs['question2'].apply(preprocess)
question_pairs = question_pairs.apply(lambda x: x.astype(str).str.lower())
question_pairs = question_pairs.drop_duplicates('question2')

print(question_pairs.columns)
print(question_pairs.head(10))

# load vocab_processor
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(model_path + 'vocab')
item_db = np.asarray(list(vocab_processor.fit_transform(list(question_pairs['question2']))))

embeddings_model = fasttext.load_model(model_path + "ft_skipgram_ws5_dim64.bin")
embeddings_lookup = []
for word in list(vocab_processor.vocabulary_._mapping):
    try:
        embeddings_lookup.append(embeddings_model[str(word)])
    except:
        pass
embeddings_lookup_ = np.asarray(embeddings_lookup)

print('# of items to be indexed: \t', item_db.shape[0])
print('Embeddings dimension: \t\t', item_db.shape[1])

# Model Hyperparameters
embedding_dim = 64


def model_output(feed_data):
    checkpoint_file = tf.train.latest_checkpoint(model_path)
    graph = tf.Graph()
    with graph.as_default():
        sess = tf.Session()
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            anchor_input = graph.get_operation_by_name("left_input").outputs[0]

            # Tensors we want to evaluate
            predictions = graph.get_operation_by_name("output/output_embedding").outputs[0]

            # Collect the predictions here
            all_predictions = []

            batch_predictions = sess.run(predictions, {anchor_input: feed_data})
    return batch_predictions


similar_pairs_list = list(question_pairs['question2'])


# Compute Vector representation for each training images and normalize those
def generate_db_normed_vectors():
    train_vectors = model_output(item_db)
    normalized_train_vectors = train_vectors / np.linalg.norm(train_vectors, axis=1).reshape(-1, 1)
    return normalized_train_vectors


# Find k nearest neighbour using cosine similarity
def find_k_nn(normalized_train_vectors, vec, k):
    dist_arr = np.matmul(normalized_train_vectors, vec.T)
    print(-1 * np.sort(-dist_arr.flatten())[:k])
    print(max(dist_arr.flatten()))
    return np.argsort(-dist_arr.flatten())[:k]


normalized_training_vectors = generate_db_normed_vectors()

# Building faiss index
print('')
print('Building faiss index...')
print('')
print('embedding_dim：')
print(embedding_dim)
print('normalized_training_vectors：')
print(normalized_training_vectors.shape)

db_index = faiss.IndexFlatIP(embedding_dim)
# add vectors to the index
db_index.add(normalized_training_vectors)

print('DB indexing done...')


# get_top_k_item
def get_top_k_item(query, k=1):
    print('')
    print('top_k_item...')
    query = [query, 'milk']
    stime = time.time()
    query = [query[0].lower(), query[1].lower()]
    query_vectors = list(vocab_processor.fit_transform(query))
    input_queries = np.asarray(query_vectors)
    print('query', query)
    print('query_vectors', query_vectors)
    print('input_queries', input_queries)
    search_vector = model_output([input_queries[0]])
    normalized_search_vec = search_vector / np.linalg.norm(search_vector)
    s_time = time.time()
    # candidate_index_i = find_k_nn(normalized_training_vectors, normalized_search_vec, k)
    _, candidate_index = db_index.search(normalized_search_vec, k)
    candidate_index = candidate_index[0]
    print('Total time to find nn: {:0.2f} ms'.format((time.time() - s_time) * 1000))
    print('------------------------------------------------------')
    print('Query: ', query[0])
    print('------------------------------------------------------')
    return candidate_index


query = "Is it healthy to eat egg whites every day"
candidate_index = get_top_k_item(query.lower(), 10)
print(candidate_index)
for index in candidate_index:
    print(similar_pairs_list[index])
