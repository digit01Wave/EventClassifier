# -*- coding: utf-8 -*-
"""
Created on Fri May 20 13:04:05 2016

@author: jessica
"""

import naive_bayes_detector as nb
import my_googlenet_classify as gClassify

    

def info(foldername):
    b = gClassify.get_folder_tags("/home/jessica/Documents/birthday/"+foldername)
    nb.dict_to_file(b, 'dict_img/b'+foldername+'.txt')

class Converter:
    def __init__(self):
        self._convert_set = self._get_convert_set()
    def _get_convert_set(self):
        s=set()
        with open('synset_words.txt', 'r') as my_file:
            for line in my_file:
                temp =  line[10:].split(',')
                for i in xrange(1, len(temp)):
                    s.add(temp[i].strip())
        return s
    def convert(self, filename):
        ans = {}
        for key in d:
            if key not in self._convert_set:
                ans[key] = d[key]
        return ans
    
    
'''
features_list = get_features_list(r_seed)
    train_set = features_list[:n_train]
    test_set = features_list[n_train:]
    classifier = nltk.NaiveBayesClassifier.train(train_set)
    classifier.show_most_informative_features(n_features)
    return '{:.4f}'.format(nltk.classify.accuracy
    '''