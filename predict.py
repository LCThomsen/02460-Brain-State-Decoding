# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 14:52:18 2017
@author: Nicolai

Caption generating network
"""

import h5py
import numpy as np
from keras.models import load_model
import pickle
import sys

data_path = 'data/'
results_path = 'results/'

def generate_captions(model, model_name, batch_size=512, max_caption_len=59, data_set='val'):
    """
    For each image id a captions is predicted using a given input model.

    # Arguments
    model: A trained network
    model_name: A given model name, which is used as the name for the saved prediction file
    batch_size: Number of samples to predict in each batch
    max_caption_len: Maximum caption length, if not max, the caption is zero-padded

    # Returns
    A text file is saved containing the image ID and the predicted caption.

    """

    word_to_index = pickle.load(open(data_path + 'word_to_index_train.pckl', 'rb'))
    index_to_word = pickle.load(open(data_path + 'index_to_word_train.pckl', 'rb'))

    image_file = h5py.File(data_path + data_set + '.h5', 'r')

    unique_image_id = list(image_file.keys())

    batch_size_delta = len(unique_image_id) % batch_size

    def predict_caption(data):
        # Initiate image caption with start token '<start>'

        r, c = data.shape
        captions = np.zeros((r, max_caption_len), dtype=np.int)
        captions[:, 0] = word_to_index['<start>']

        dont_update_list = np.zeros(r, dtype=bool)
        for jj in range(1, max_caption_len):
            next_words_prob = model.predict_on_batch([data, captions])  # probability distribution over vocabulary
            captions[:, jj] = next_words_prob.argmax(axis=1)
            captions[dont_update_list, jj] = 0

            if any(captions[:, jj] == end_index):  # stop generation caption
                dont_update_list += captions[:, jj] == end_index

            if all(dont_update_list):
                break

        return captions

    with open(results_path + model_name + '/test_predicted_captions_' + data_set + '.txt', 'w') as data_file:
        data_file.write('image_id\tcaption\n')

        data_file.flush()

        end_index = word_to_index['<end>']
        for ii in range(0, len(unique_image_id), batch_size):
            print("Images {} of {}".format(ii, len(unique_image_id)))

            if ii == len(unique_image_id) - batch_size_delta:
                batch_size = batch_size_delta

            batch_unique_image_id = unique_image_id[ii:(ii+batch_size)]
            images = np.stack([image_file[img_id].value for img_id in batch_unique_image_id])

            captions = predict_caption(images)

            for kk in range(0, batch_size):
                caption_word = []
                caption_idx = captions[kk]
                img_id = batch_unique_image_id[kk]
                for idx in caption_idx:
                    if idx == 0:
                        break
                    caption_word.append(index_to_word[idx])

                cap = " ".join(caption_word)
                data_file.write(img_id+"\t"+cap+"\n")

    print('Done..')

# def generate_captions(model, model_name, batch_size=512, max_caption_len=59, data_set='val'):
#     """
#     For each image id a captions is predicted using a given input model.
#
#     # Arguments
#     model: A trained network
#     model_name: A given model name, which is used as the name for the saved prediction file
#     batch_size: Number of samples to predict in each batch
#     max_caption_len: Maximum caption length, if not max, the caption is zero-padded
#
#     # Returns
#     A text file is saved containing the image ID and the predicted caption.
#
#     """
#
#     word_to_index = pickle.load(open(data_path + 'word_to_index_train.pckl', 'rb'))
#     index_to_word = pickle.load(open(data_path + 'index_to_word_train.pckl', 'rb'))
#
#     image_file = h5py.File(data_path + data_set + '.h5', 'r')
#
#     unique_image_id = list(image_file.keys())
#
#     batch_size_delta = len(unique_image_id) % batch_size
#
#     def predict_caption(data):
#         # Initiate image caption with start token '<start>'
#
#         r, c = data.shape
#         captions = np.zeros((r, max_caption_len), dtype=np.int)
#         captions[:, 0] = word_to_index['<start>']
#
#         dont_update_list = np.zeros(r, dtype=bool)
#         for jj in range(1, max_caption_len):
#             next_words_prob = model.predict_on_batch([data, captions])  # probability distribution over vocabulary
#             next_words_idx = next_words_prob.argmax(axis=1)
#             captions[:, jj] = next_words_idx
#             captions[dont_update_list, jj] = 0
#
#             if any(captions[:, jj] == end_index):  # stop generation caption
#                 dont_update_list += captions[:, jj] == end_index
#
#             if all(dont_update_list):
#                 break
#
#         return captions
#
#     with open(results_path + model_name + '/test_predicted_captions_' + data_set + '.txt', 'w') as data_file:
#         data_file.write('image_id\tcaption\n')
#
#         data_file.flush()
#
#         end_index = word_to_index['<end>']
#         for ii in range(0, len(unique_image_id), batch_size):
#             print("Images {} of {}".format(ii, len(unique_image_id)))
#
#             if ii == len(unique_image_id) - batch_size_delta:
#                 batch_size = batch_size_delta
#
#             batch_unique_image_id = unique_image_id[ii:(ii+batch_size)]
#             images = np.stack([image_file[img_id].value for img_id in batch_unique_image_id])
#
#             captions = predict_caption(images)
#
#             for kk in range(0, batch_size):
#                 caption_word = []
#                 caption_idx = captions[kk]
#                 img_id = batch_unique_image_id[kk]
#                 for idx in caption_idx:
#                     if idx == 0:
#                         break
#                     caption_word.append(index_to_word[idx])
#
#                 cap = " ".join(caption_word)
#                 data_file.write(img_id+"\t"+cap+"\n")
#
#     print('Done..')


if __name__ == '__main__':

    # investigate why it lowers all characters
    model_architecture_name = 'image_bn_two_lstm_dropout_best'
    model_name1 = '/model-exhaustive-train-2018_5_22-10_18.h5'
    try:
        model1 = load_model(results_path + model_architecture_name + model_name1)
    except:
        sys.exit('Trained model does not exist')

    generate_captions(model1, model_architecture_name, batch_size=32, max_caption_len=59, data_set='test')
    generate_captions(model1, model_architecture_name, batch_size=32, max_caption_len=59, data_set='val')

