# -*- coding: utf-8 -*-
"""
Created on Fri May 20 13:04:05 2016

@author: jessica
"""
from __future__ import division
import my_googlenet_classify as gClassify
from os import listdir
from os.path import isfile, join
import file_dictionary as fd


def info(foldername):
    b = gClassify.get_folder_tags("/home/jessica/Documents/children_birthday/"+foldername)
    fd.dict_to_file(b, 'dict_img/children_birthday'+foldername+'.txt')

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
    def convert(self, folder_path):
        filenames = [join(folder_path,f) for f in listdir(folder_path) if isfile(join(folder_path, f))]
        for filename in filenames:        
            d = fd.file_to_dict(filename)
            new_d = {}
            for key in d:
                if key not in self._convert_set:
                    new_d[key] = d[key]
            fd.dict_to_file(new_d, filename, append=False)

