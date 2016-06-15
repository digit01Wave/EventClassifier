from __future__ import division

"""
Created on Wed Jun  1 07:13:03 2016

@author: jessica

This module provides the building blocks of tag matching
"""
import urllib, json
import re

###################################################
#WIKIPEDIA API
######################################################

pattern=re.compile("<rev.*>(.*?)</rev>", re.MULTILINE|re.DOTALL)
anchor_pattern=re.compile("\[\[(.*?)\]\]", re.MULTILINE|re.DOTALL)
#https://en.wikipedia.org/w/api.php?format=json&action=query&prop=revisions&rvprop=content&exintro=&explaintext=&titles=wedding

def get_objects(event, dict_set, syn_lvl = 1):
    """
    Takes in the event, and returns all objects in the dictionary that match
    said event.
    
    Parameters
    ------------
    event(str): Event that person wishes to know about
    dict_set(set): set of words we are allowed to use
    syn_lvl (int): the number of levels we are allowed to look in wordnet synonyms 
    
    Returns
    -----------
    dict{str:int or boolean}: dictionary of all objects that match the dict_set
        Maps word to whether or not it was an anchor tag and how many times it appeared
        Format: {'count':<word count>, 'anchor':true|false}
    """
    #obtain url response/text    
    url = "https://en.wikipedia.org/w/api.php?format=xml&action=query&prop=revisions&rvprop=content&exintro=&explaintext=&titles="+event
    response = urllib.urlopen(url)    
    s = response.read()

    #get all content
    m = re.search(pattern, s)
    s=m.group(1)
    
    #get only words in our dictionary
    ans = {}
    for word in s.split():
        if word in dict_set:
            if word in ans:
                ans[word]['count']+=1
            else:
                ans[word]={}
                ans[word]['count']=1
                ans[word]['anchor']=False
    #get only anchor tags
    m = re.findall(anchor_pattern, s)
    for anchor in m:
        if anchor in ans:
            ans[anchor]['anchor']=True
    return ans
    
def convert_concept_to_dict(event, concept_dict):
    """
    Converts the given event concept dictionary into the appropriate feature dictionary
    that can be fed into the EventClassifier.classify_feature function
    
    Parameters
    ---------------
    concept_dict (dict{str:{'anchor':boolean, 'count':int or float}}):
        Dictionary that maps each tag to whether or not it is an anchor tag and the
        number of times it appears. Is what is returned by get_objects function.
        e.g. {'conforter':{'anchor':False, 'count':2}}
    
    Returns
    ------------
    Counter
        Dictionary that maps each tag to the weighted number of times
    
    """
    pass

def get_anchor_tags(event):
    """
    Takes in event of interest ang obtains the anchor tags listed on that wikipedia page
    Parameters:
    ------------
    event(str): event we wish to find
    
    Returns
    -----------
    list(str): the list of anchor tags found on the main wikipedia page
    """
    #obtain url response/text    
    url = "https://en.wikipedia.org/w/api.php?format=xml&action=query&prop=revisions&rvprop=content&exintro=&explaintext=&titles="+event
    response = urllib.urlopen(url)    
    s = response.read()

    #get all content
    m = re.search(pattern, s)
    s=m.group(1)
    
    #get only anchor tags
    m = re.findall(anchor_pattern, s)
    return m

    


def get_event(objects, syn_lvl = 1):
    """
    Uses the objects to determine what event we are in
    
    Parameters
    ----------
    objects (list[str]): List of objects user believes is in event
    syn_lvl (int): the number of levels we are allowed to look in wordnet synonyms    
    
    Returns
    -----------
    str: The event that best matches the given objects
    """
    pass

def get_inception_set(include_new = False):
    """
    obtains all the words that are in the google inception
    """
    s=set()
    with open('synset_words.txt', 'r') as my_file:
        for line in my_file:
            temp =  line[10:].split(',')
            for i in xrange(len(temp)):
                s.add(temp[i].strip())
    if include_new:
        s.add('cake')
        s.add('christmas tree')
    return s


def getDmWord(key):
    '''
    Accesses datamuse to get all the words associated with given key and returns them
    
    Parameter: key(str)
    Returns: set(str)
    '''
    response = urllib.urlopen("https://api.datamuse.com/words?ml="+key)
    data = json.loads(response.read())
    return_set = set()
    for item in data:
        return_set.add(item["word"]) 
    return return_set

def getSpecialDmWords(key, special, level=1):
    """
    Only obtains words that are in the given iterable 
    
    Parameter
    ----------
    key(str):
        word we wish to search for
    special(str):
        Anything that can be used with 'in' in order to obtain proper dm words
    level(str):
        How many synonyms we wish to go down
    Returns:
    ----------
    set(str): set of all words that match
    """
    data = levelTopicDm(key, dig=level)
    return_set = set()
    for item in data:
        if(item in special):
            return_set.add(item) 
    return return_set

def levelTopicDm(topic, dig=1):
    """
    Obtains a set of DM words with given words
    """
    to_explore = [topic] #items we need to explore
    i = 0 #index
    c = 1
    level_len = 1 #ending index of this level
    return_set = set()
    while c <= dig:
        while i < level_len: #iterate through all terms at current level          
            response = urllib.urlopen("https://api.datamuse.com/words?ml="+to_explore[i])
            data = json.loads(response.read())
            for item in data:
                if(item["word"] not in return_set):
                    return_set.add(item["word"])
                    to_explore.append(item["word"])
            i+=1
        c+=1
        level_len = len(to_explore)            
    return return_set

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



