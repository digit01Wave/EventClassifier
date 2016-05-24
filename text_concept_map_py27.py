# -*- coding: utf-8 -*-
"""
This module provides the building blocks of tag matching
"""
from __future__ import division
import urllib, json




    
'''
Accesses datamuse to get all the words associated with given key and puts them
in a dictionary in the format {word:key}

Parameter: key(str)
Returns: dict{Dm word: key}
'''
def getDmWord(key):
    response = urllib.urlopen("https://api.datamuse.com/words?ml="+key)
    data = json.loads(response.read())
    return_dict = {}
    for item in data:
        return_dict[item["word"]] = key 
    return return_dict

'''
Given a list of dictionaries, return a set of all the keys they have in common

Parameter: dict_list (list[dict])
Returns: set(str)
'''
def findMatch(dict_list):
    ans = set()
    for i in xrange(len(dict_list)):
        for j in xrange(i+1, len(dict_list)):
            dict1 = dict_list[i]
            dict2 = dict_list[j]
            for key in dict1:
                if key in dict2:
                    ans.add(key)
    return ans
    
    
'''
modifies given dictionary to include all the words related to a topic.
each of these words will come with a topic, and a closeness number
'Closeness' score will be determined by how far away the word was from the topic

Parameter: 
----------
    return_dict: dict{word: list[list[topic1(str), closeness(int)]]} 
    topic(str): key that we want to query from datamuse
    dig: how many levels we want to wearch in datamuse
Returns: Nothing
----------            
    '''
def levelTopicDict(return_dict, topic, dig=1):
    to_explore = [topic] #items we need to explore
    i = 0 #index
    c = 1
    level_len = 1 #ending index of this level
    while c <= dig:
        while i < level_len: #iterate through all terms at current level          
            response = urllib.urlopen("https://api.datamuse.com/words?ml="+to_explore[i])
            data = json.loads(response.read())
            for item in data:
                if(item["word"] not in return_dict):
                    return_dict[item["word"]] = [topic, 1/c]
                    to_explore.append(item["word"])
            i+=1
        c+=1
        level_len = len(to_explore)
