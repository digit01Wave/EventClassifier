# -*- coding: utf-8 -*-
"""
Created on Tue May 17 10:34:02 2016

@author: jessica
"""

import nltk
import math
import json
import my_googlenet_classify as gClassify
from collections import defaultdict
import operator
import py_bing_search as bing
import urllib

###############################################################################
#DATASCRAPING
###############################################################################
def jsonUrl(url):
    '''
    Accesses the JSON retuned from a given url
    
    Parameter: url(str)
    Returns: list[dict]
    '''
    response = urllib.urlopen(url)
    data = json.loads(response.read())
    return data

#http://sln.ics.uci.edu:8085/eventshoplinux/rest/sttwebservice/search/70/box/33.554234134963096,-117.96170501708986/33.782820548089866,-117.67674713134767/1456531200000/1457740800000

def krumbsJsonFeatures(url):
    """
    Accesses the eventshop JSON array from given url and exxtracts the given 
    features (not in any particular order)
    1. id
    2. time
    3. image_url
    4. intent (name, synonym)
    5. geo:(lat, lng) #may or may not be there
    6. caption: #may or may not be there
    
    Returns
    -------------
    list[dict{str:str}]
    """
    response = urllib.urlopen(url)
    data = json.loads(response.read())
    toReturn = [None]*len(data)
    for i in range(len(data)):
        print "we are on index ", i
        toReturn[i] = {}
        toReturn[i]["id"] = data[i]["stt_id"]
        try: #geo coordinates
            toReturn[i]["geo"] = tuple(data[i]["stt_where"]["point"])        
        except:
            pass
        
        toReturn[i]["time"] = data[i]["stt_when"]["datetime"]
        
        try: #image url        
            toReturn[i]["image_url"] = data[i]["stt_what"]["media_source"]["value"]
        except:
            toReturn[i]["image_url"] = data[i]["stt_what"]["media_source_photo"]["value"]
        
        try: #intent
            toReturn[i]["intent"] = (data[i]["stt_what"]["intent_name"]["value"], data[i]["stt_what"]["intent_used_synonym"]["value"])
        except:
            toReturn[i]["intent"] = (None, data[i]["stt_what"]["intent_used_synonym"]["value"])
        
        try: #caption
            toReturn[i]["caption"] = data[i]["stt_what"]["caption"]["value"]
        except:
            pass
        
        try: #location
            toReturn[i]["location"] = data[i]["raw_data"]["media"][0]["where"]["revgeo_places"][0]["name"]
        except:
            print "AAAAHH locations: ", data[i]["raw_data"]
            print "type of location", type(data[i]["raw_data"])
    return toReturn
    
def bing_image_search(query, output_filename, lim):
    """
    search bing for lim number of images matching the query and output the 
    following information into the given output filename
        1. title
        2. url
        3. width
        4. height
        5. size
        6. content type
    Parameters
    ------------
        output_filename: str: file to output to
        query: str: search term/terms (space delimited)
        limi: number of results we want
    
    Returns
    ------------
        int: number of items added to output_filename
    """
    if(type(query) != str):
        raise TypeError("query must be of type string")
    if(type(lim) != int or lim <= 0):
        raise ValueError("limit must be positive integer")
    
    searcher = bing.PyBingImageSearch(query)
    result = searcher.search_all(limit=lim, format='json')
    bing.image_results_to_file(result, output_filename, append=True)
    return len(result)
    
###############################################################################
#FEATURES
###############################################################################
    
def get_tags(image_filename):
    """
    Returns the tags from the given image   
    Parameters
    ----------
    image_filename : str
        file name/path of image

    Returns
    -------
    list[str:float]
        list of tags and their respective confidence scores
    """
    return gClassify.run_inference_on_image(image_filename)
    
    
def _get_image_links(filename, delimiter=";;;", index=1):
    """
    pulls imagelinks from given filename that is in some delimited format
    
    Parameters
    ------------
    filename:str: path to txt file
    delimeter:str: separating character string used to separate pieces of info in same entry
    index: index of delimited strings where link is
    
    Returns
    ----------
    list[str]: list of pulled links
    """
    ans = []
    with open(filename, 'r') as myFile:
        for line in myFile:
            temp = line.split(delimiter)
            ans.append(temp[index])
    return ans


def dict_to_file(d, filename, append=False):
    action = 'w'
    if append:
        action = 'a'
    with open(filename, action) as myfile:
        for key in d:
            try:            
                myfile.write("{}:{}\n".format(key, d[key]))
            except:
                print("ERROR: Unable to input {}:{}".format(key, d[key]))

def file_to_dict(filename):
    d = defaultdict(float)
    with open(filename, 'r') as myFile:
        for line in myFile:
            temp = line.split(":")
            if(len(temp) == 2):
                d[temp[0]] = float(temp[1].strip())
    return d
        
#Note: to do image classification on a list of links, it is best to run the following
#>>> a = _get_image_links
#>>> result = gClassify.get_collection_tags(your_image_link, rel_threshold=0.3, current_dict=your_default_dict) 




###############################################################################
# NAIVE BAYES CLASSIFICATION
###############################################################################
class EventClassification:
    
    def __init__(self, dict_list, occurrance):
        """sets up inverse tags (see _create_inverse_tag_feature for more information"""
        self.tag_feature = self._create_event_tag_feature(dict_list, occurrance)        
        self.inverse_tag_feature = self._create_inverse_tag_feature(dict_list, occurrance)        
    
    def _create_event_tag_feature(self, dict_list, occurance):
        """
        creates a dictionary that maps events to theit tags with adjusted probabilitites

        Parameters
        ------------------
        dict_list: list[(str,dict{str:float})]
            list of tuples where the first item is the event title, and the second
            item is the dictionaries of tags with the number of times they occured  
        occurance
            percentage of times tags need to appear in order to be included in the dictionary
        
        Returns
        ------------------
        dict{str:dict[str: float]}: aka dict{event:dict{tag:probablility}}
            a dictionary that maps each event to list of all the tag/term along with
            its importance to the event (all event importance will add up to 1)
        
        """
        ans = defaultdict(dict)
        
        for event, tag_dict in dict_list:  
            #find total occurance of items in dictionary        
            total = sum([v for k,v in tag_dict.iteritems()])
            
            #add tag to our event dictionary if it meets the threshold requirement
            count = 0
            for tag in tag_dict: #tag = key in event_dict{tag, occured}
                if(tag_dict[tag]/total >= occurance):
                    count += tag_dict[tag]
                    ans[event][tag] =  tag_dict[tag] ##############################MAYBE CHANGE THIS PROBABILITY (tf-idf?)
            #normalize the tags
            for tag in ans[event]:
                ans[event][tag] = ans[event][tag]/count
        
            
        return ans      
        
    def _create_inverse_tag_feature(self, dict_list, occurance):
        """
        creates a dictionary of inverse terms from given event tag list
        (i.e. each tuple shows how important that tag is to that event)
        
        Parameters
        ------------------
        dict_list: list[(str,dict{str:float})]
            list of tuples where the first item is the event title, and the second
            item is the dictionaries of tags with the number of times they occured  
        occurance
            percentage of times tags need to appear in order to be included in the dictionary
        
        Returns
        ------------------
        dict{str:list[(str, float)]}
            a dictionary that maps each tag/term to   list of all the events that tag
            appeared in. Each event is represetned by a 2-tuple with the event's title
            and the percentage of times the tag had appeared in that event
        """
        ans = defaultdict(list)
        
        for event, tag_dict in dict_list:  
            #find total occurance of items in dictionary        
            total = sum([v for k,v in tag_dict.iteritems()])
            
            #add tag to our inverse_tag_dict if it meets the occurance threshold
            for tag in tag_dict: #tag = key in event_dict{tag, occured}
                if(tag_dict[tag]/total >= occurance):
                    ans[tag].append((event, tag_dict[tag]/total)) ##############################MAYBE CHANGE THIS PROBABILITY (tf-idf?)
        return ans
        
    def get_event_label(self, img_filename, rel_occurance=0.3):
        temp = gClassify.run_inference_on_image(img_filename)
        
        #get relevant tags    
        tags = []
        for tup in temp: #each tupple in form (tag, image_probability)
            if(tup[1] > temp[0][1]*rel_occurance):
                tags.append(tup[0])
        
        #get label based on the relevant tags
        scores = defaultdict(float)
        for event, tag_dict in self.event_tag_feature.iteritems():
            for tag in tags:
                if(tag in tag_dict):
                    scores[event]+=tag_dict[tag]
        sorted_x = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)
        print("here are your top 10 event rankings:")
        print(sorted_x[0:10])
        return sorted_x[0]
                    

def train_naive_Bayes(n_train, n_features, r_seed):
    """
    Trains naive Bayes classifier using nltk package.

    The function performs the following steps:
        - First, import the data features 
        - Define 'train_set' using the first 'n_train' elements of features_list,
            and 'test_set' using the remainder.  Note that the data will have already been
            shuffled which is why splitting the data into training and testing data based on
            their input order is fine
        - Train a naive Bayes classifier with nltk.NaiveBayesClassifier.train().
        - Show the 'n_features' most informative features
            using the function classifier.show_most_informative_features().
        - Calculate the accuracy of the 'test_set' and return the value in percentage.

    Parameters
    ----------
    n_train : int
        Number of elements that are used for training.
    n_features : int
        Number of the most informative features that will be printed out.
    r_seed : int
        Random seed for shuffling data.

    Returns
    -------
    str
        Accuracy of the naive Bayes classifier in percentage, rounded to four decimal places.

    Examples
    --------
    >>> test_naive_Bayes(5000, 3, 10)
    Most Informative Features
          last_character = 'a'            female : male   =     37.6 : 1.0
          last_character = 'k'              male : female =     28.8 : 1.0
          last_character = 'd'              male : female =     11.4 : 1.0
    '74.3546'
    """
    train_set = features_list[:n_train]
    classifier = nltk.NaiveBayesClassifier.train(train_set)
    classifier.show_most_informative_features(n_features)
    return classifier



def cross_validate_naive_Bayes(n_fold, r_seed):
    """
    Subdivide the original data into 'n_fold' subsets, each of which is called a fold.
    For each of these folds, train a model using all the data except the data in that fold
    and test the model on the held-out fold which is treated as a validation set. Save accuracies
    from each fold and return the accuracies as a list of strings, each of which contains an
    accuracy value rounded to four decimal places.

    Your function should perform the following steps:
        - Import the features from the names corpus using 'get_features_list'.
        - Compute the size of each validation data set as M = floor(total_n_of_names / n_fold).
        - For each of the n_fold folds:
            - Let the validation set be the data in this fold
            - Let the training set be the data in all the other folds
            - Train a naive Bayes classifier on the training set
            - Record the trained model's accuracy on the validation set as the accuracy for
              this fold
        - (There may be a few examples at the end that were not included in any test set. This is
            ok.)
        - At the end you should have 'n_fold' accuracies from the 'n_fold' validation sets.
          The function should return a list of the 'n_fold' validation accuracies as a list of
          'n_fold' percentages, i.e., a list of floats all between 0 and 100 (rounded to four
          decimal places).

    As an example, if names were of length 100 and n_fold=3, then M = floor(100/3) = 33.
        The first validation set would contain the data from names 0 to 32
        The first training set would contain the data from names 33 to 99.
        The second validation set would contain the data from the names 33 to 65.
        The second training set would contain the data from names 0 to 32 and 66 to 99.
        The third validation set would contain the data from names 66 through 98.
        The third training set would contain the data from names 0 through 65 and the data from
            name 99.
        The data from name 99 would not be included in any of the validation sets in this example.

    Parameters
    ----------
    n_fold: int
        Number of subsets(folds) for cross-validation.
    r_seed: int
        Random seed for shuffling data.

    Returns
    -------
    List[str]
        List of accuracies in percentage (rounded to four decimal places) for each folds.

    Examples
    --------
    >>> cross_validate_naive_Bayes(5, 3)
    ['75.0000', '75.1889', '73.9295', '72.6071', '73.8665']
    """
    # Check if the inputs have valid numbers
    if n_fold <= 1 or r_seed <= 0:
        raise ValueError('must have n_fold > 1, r_seed > 0')

    ### YOUR SOLUTION STARTS HERE###
    features_list = get_features_list(r_seed)
    M = math.floor(len(features_list)/n_fold)
    accuracy_list = [] #to record trained model's accuracy on validation set
    start = 0 #start index of current fold
    fold_num = 1 #fold we are on
    while fold_num <= n_fold:
        test_set = features_list[start:fold_num*M]
        train_set = features_list[0:start]+ features_list[fold_num*M:]
        classifier = nltk.NaiveBayesClassifier.train(train_set)
        accuracy_list.append('{:.4f}'.format(nltk.classify.accuracy(classifier, test_set)*100))
        start = fold_num*M
        fold_num+=1
    return(accuracy_list)

def start():
    while True:
        url = raw_input('Please input url (or q for quit): ')
        if(url == 'q'):
            return
        try:
            images = jsonUrl(url)
            break
        except:
            print "That was an invalid url."
    print(images)


if __name__ == '__main__':
    # You may use this section as you please, but the contents won't be graded.
    pass
    
    
    
    
    