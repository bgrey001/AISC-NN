#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# data: https://download.pytorch.org/tutorial/data.zip
import io
import os
import unicodedata
import string
import glob

import torch
import random

# alphabet small + capital letters + " .,;'"
ALL_LETTERS = string.ascii_letters + " .,;'"
N_LETTERS = len(ALL_LETTERS)

# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
def unicode_to_ascii(s): # convert data from unicode to ascii
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in ALL_LETTERS
    )

def load_data(): # loads all the files in /names and loads all the names and gets the country as well
    # Build the category_lines dictionary, a list of names per language
    category_lines = {}
    all_categories = []
    
    def find_files(path):
        return glob.glob(path)
    
    # Read a file and split into lines
    def read_lines(filename):
        lines = io.open(filename, encoding='utf-8').read().strip().split('\n')
        return [unicode_to_ascii(line) for line in lines]
    
    for filename in find_files('../../data/misc_data/names/*.txt'):
        category = os.path.splitext(os.path.basename(filename))[0]
        all_categories.append(category)
        
        lines = read_lines(filename)
        category_lines[category] = lines
        
    return category_lines, all_categories

# =============================================================================
# x = ('apple', 'banana', 'cherry')
# y = enumerate(x) # enumerate(iterable, start)
# for i, letter in y:
#     print(i)
# 
# x = 'Benedict'
# y = enumerate(x)
# for i, letter in y:
#     print(i, letter)
# =============================================================================

"""
To represent a single letter, we use a “one-hot vector” of 
size <1 x n_letters>. A one-hot vector is filled with 0s
except for a 1 at index of the current letter, e.g. "b" = <0 1 0 0 0 ...>.
To make a word we join a bunch of those into a
2D matrix <line_length x 1 x n_letters>.
That extra 1 dimension is because PyTorch assumes
everything is in batches - we’re just using a batch size of 1 here.
"""

# Find letter index from all_letters, e.g. "a" = 0
def letter_to_index(letter):
    return ALL_LETTERS.find(letter)

# Just for demonstration, turn a letter into a <1 x n_letters> Tensor
def letter_to_tensor(letter):
    tensor = torch.zeros(1, N_LETTERS)
    tensor[0][letter_to_index(letter)] = 1
    return tensor

# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def line_to_tensor(line):
    tensor = torch.zeros(len(line), 1, N_LETTERS)
    for i, letter in enumerate(line): # the enumerate() function adds a counter as the key of tghe enumerate object (for iterating through tuples when access to an index is needed)
        tensor[i][0][letter_to_index(letter)] = 1
    return tensor


def random_training_example(category_lines, all_categories):
    
    def random_choice(a):
        random_idx = random.randint(0, len(a) - 1)
        return a[random_idx]
    
    category = random_choice(all_categories)
    line = random_choice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = line_to_tensor(line)
    return category, line, category_tensor, line_tensor


def generate_test_set(category_lines, all_categories, set_size):
    
    def random_choice(a):
        random_idx = random.randint(0, len(a) - 1)
        return a[random_idx]
    
    training_set = []
    for i in range(set_size):
        category = random_choice(all_categories)
        line = random_choice(category_lines[category])
        category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
        line_tensor = line_to_tensor(line)
        training_set.append([category, line, category_tensor, line_tensor])
        
    return training_set
    



# =============================================================================
# if __name__ == '__main__':
# =============================================================================
# =============================================================================
#     print(ALL_LETTERS)
#     print(unicode_to_ascii('Ślusàrski'))
#     
#     category_lines, all_categories = load_data()
#     print(category_lines['Italian'][:5])
# =============================================================================
    
# =============================================================================
#     print(letter_to_tensor('b').size()) # [1, 57]
# =============================================================================
# =============================================================================
#     print(line_to_tensor('Jones').size()) # [5, 1, 57]
# =============================================================================
