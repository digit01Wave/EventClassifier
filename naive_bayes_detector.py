# -*- coding: utf-8 -*-
"""
Created on Tue May 17 10:34:02 2016

@author: jessica
"""
from __future__ import division

import json
import my_googlenet_classify as gClassify
from collections import defaultdict
import operator
import py_bing_search as bing
import urllib
from collections import Counter
import math



#convenient imports to other modules
import test as t
import file_dictionary as fd

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
    

    
#Note: to do image classification on a list of links, it is best to run the following
#>>> a = _get_image_links
#>>> result = gClassify.get_collection_tags(your_image_link, rel_threshold=0.3, current_dict=your_default_dict) 




###############################################################################
# Special Classification
###############################################################################
#Some source code taken form NLTK 3.0 documentation for Naive Bayes Classifier
#http://www.nltk.org/_modules/nltk/classify/naivebayes.html
#
class EventClassifier:
    """
    Implementation of a specialized naive bayes classifier for image collection
    event recognition.
    
    unlike in nltk's implementation of Naive Bayes, it counts tags as positive values
    rather than as netative values
    """
    
    def __init__(self, tag_counters = None, doc_importance = 0.01, extra_features={}):
        """
        instance variables
        ------------------
        self._tag_counters (dict{str:Counter}):
            Dictionary that maps each event to a corresponding dictionary of tags
            Each tag dictionary maps a tag to the number of times it appears
            
        self._salient_tags (dict{str:set(str)})
            Dictionary that maps each 
        
        self._event_full_totals: Counter
            Dictionary that maps each event to the number of items is in it.
            
        self._event_div_totals: Counter
            Dictionary that maps eaach event to the number of salient items in it.
        
        self._tag_doc_freq (dict{str,int}): 
            dictionary that maps a tag to the number of labels it occurs in
            This will be used to help calculate idf (inverse-document-frequency)
            
        self._doc_importance (flaot):
            What percentage of the document the tag must appear in before it is
            considered to be added as "existing" in that particular event
        
        self._extra_features (dict{str: anything})
            Any extra features that need to be added (e.g. time, geolocation)
        """
        #initialize everython
        self._doc_importance = doc_importance
        self._tag_counters = {}
        self._event_full_totals = Counter()
        self._event_div_totals = Counter() 
        self._salient_tags = {}
        self._tag_doc_freq = Counter()
        self._extra_features = extra_features
        
        #do specialized initialization
        if(tag_counters is not None):
            #check validity
            if(type(tag_counters) != dict):
                raise TypeError("tag features must be of type dict")
            
            #add to class while checking more types
            for event in tag_counters:
                self._add_event_tag_feature(event, tag_counters[event])
                
        
    def print_important_features(self, event=None):
        """
        Prints out to console the salient tags of the given event along with their
        relative 'weights' (which range between 0-1 and add up to 1)
        
        Parameter
        -------------
        event(str):
            Name of event we wish to print information about.
            If none given, will print info about all of them
        
        Returns
        --------------
            Nothing
        """
        print "Printing important features..."
        print "document importance =", self._doc_importance
        if event is None:
            for e in self._salient_tags:
                self._print_single_important(e)
        else:
            self._print_single_important(event)
    
    def _print_single_important(self, event):
        """
        helper function
        """
        total = self._event_div_totals[event]
        print event
        print "\tSalient Tag total:", total
        
        temp = {}
        for tag in self._salient_tags[event]:
            temp[tag] = self._tag_counters[event][tag]/total
        
        #sort temp and print it in order
        sorted_x = sorted(temp.items(), key=operator.itemgetter(1), reverse=True)
        for item in sorted_x:
            print '\t', item[0], ': ', item[1]
    
    def print_important_labels(self, tag):
        temp = {}
        for event in self._salient_tags:
            if tag in self._salient_tags[event]:
                temp[event] = self._tag_counters[event][tag]/self._event_div_totals[event]
        
        sorted_x = sorted(temp.items(), key=operator.itemgetter(1), reverse=True)  
        for item in sorted_x:
            print '\t', item[0], ': ', item[1]
        
        
    def train_features(self, feature_list):
        """
        trains classifier on given feature list
        
        Adds to self._tag_counters maps events to their tags and adjustes 
        appropriate counts in the appropriate dictionaries.

        Parameters
        ------------------
        feature_list: list[(str,dict{str:float or int})]
            List of tuples where the first item is the event title, and the second
            item is its dictionary of tags with the number of times they occured in given event  

        Returns
        ------------------
        Nothing
        """
        for feature_dict, label in feature_list:
            self._add_event_tag_feature(label, feature_dict)
    
    def train_collection(self, event, folder_path):
        """
        Trains classifier on a folder filled with images of a given collection
        for the given event
        
        Adds to self._tag_counters that maps events to their tags and adjustes 
        appropriate counts in the appropriate dictionaries.

        Parameters
        ------------------
        event (str): name of event/key to add/add to       
        folder_path(str): folder filled with jpg images to train on  

        Returns
        ------------------
        Nothing
        """
        print "Beginning training process..."
        temp = gClassify.get_folder_tags(folder_path)
        self._add_event_tag_feature(event, temp)
    
    def train_link_collection(self, event, link_list, rel_thresh=0.3):
        """
        Trains classifier on set of image links
        
        Parameters
        -----------
        event (str): 
            name of event/key to add/add to       
        link_list (list[str]):  
            list of links from which to download training images from
        rel_threshold (float):
            since gClassify gives many results, rel_occurance determines how close
            the other results have to be to be included in our dictionary
        
        Returns
        ----------
        Nothing
        """
        tag_dict = gClassify.get_collection_tags(link_list, rel_threshold=rel_thresh)
        self._add_event_tag_feature(event, tag_dict)
        
    def classify_folder(self, img_foldername, rel_threshold=0.3):
        """
        Given a collection of images, return an appropriate event label
        
        Calculate probablilty of each event by findiing all the objects detected
        within the image collection based on the formula:
            P(event|)
            
        Parameters
        ------------ 
        img_foldername (str):
            Path to folder containing image collection to test on
        rel_threshold (float):
            since gClassify gives many results, rel_occurance determines how close
            the other results have to be to be included in our dictionary
        
        Returns
        ------------
        str: the predicted label of the event
        """
        #get collection tags dict{tag:tag_count}
        c_tags = gClassify.get_folder_tags(img_foldername, rel_threshold)
        
        #get label based on the tags
        return self.classify_feature(c_tags)
        
    def classify_img(self, img_filename, rel_occurrence=0.3):
        """
        From single image, try to determine the appropriate event label.
        
        It should be noted, that this method may be significanly less accurate than
        classifying on a collection of images unless this image is particularly
        determining/important to the event
        
        Parameters
        ------------
        img_filename (str):
            Path to file containing image to test on
        rel_occurrence (float):
            Since gClassify gives many results, rel_occurance determines how close
            the other results have to be to be included in our dictionary
            
        Returns
        ------------
        str: the predicted label of the event
        """
        #get tags list[tuple(tag, probability correct)]
        temp_tags = gClassify.run_inference_on_image(img_filename)
        
        #get relevant tags    
        tags = {} #going to end in the form dict{str:int or float}
        for tup in temp_tags: #each tupple in form (tag, image_probability)
            if(tup[1] > temp_tags[0][1]*rel_occurrence):
                #accept only the first equivalence class
                temp = tup[0].split(",")
                tags[temp[0].strip()] = 1
        
        #get label based on the relevant tags
        return self.classify_feature(tags)
        
    def classify_feature(self, feature_tag_dict, alpha = 1.0):
        """
        Classifies event based on given feature_dict
        
        Parameters
        -------------
        feature_dict (dict{str:float or int}):
            features that map a tag to the number of times it had appeared
        alpha (float):
            The weight given the the number of occurances in given feature_dict
        """        
        if(len(feature_tag_dict) == 0):
            print("WARNING: feature dictionary given is empty")
            return None
        
        #get label based on the relevant tags
        scores = defaultdict(float) #our calculated scores

                
        feature_tag_total = sum(feature_tag_dict.values()) #the total in given tag_dict
        N = len(self._tag_counters) #number of labels
        total = 0.0 #adding up scores and use to normalize
  
        
        for event, tag_set in self._salient_tags.iteritems():
            weight = 1.0
            tag_total = self._event_div_totals[event]
            for tag in feature_tag_dict:
                #if the tag exists as a salent tag for current event, include its probability
                if(tag in tag_set):
                    #calculate probability
                    scores[event]+=(self._tag_counters[event][tag]/tag_total)*math.log(N/self._tag_doc_freq[tag])*(alpha*(feature_tag_dict[tag]/feature_tag_total))
                elif feature_tag_dict[tag]/feature_tag_total > 0.2:
                    #scores[event] *= (feature_tag_dict[tag]/feature_tag_total)
                    weight *= self._tag_counters[event][tag]/self._event_full_totals[event]                  
                    #break
            scores[event] *= weight
            total+=scores[event]
        
        if(total == 0.0):
            print("warning: no events match given tags")
            return None
        #normalize to make all probabilities add to 1
        for event in scores:
            scores[event] /= total
            
        sorted_x = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)
        return [(i[0], i[1]*100) for i in sorted_x[0:10]]
        
    
    
    @staticmethod
    def store_classifier(classifier, filename):
        """
        stores classifier in a txt file in the form of a dictionary
        
        In the end, the file will contain 2 lines in the following format:
            line1: tag_counters: event1|||tag1:count...,tagn:count;;;event2...
            line2: doc_importance
            line3: extra1|||tag:value...,
        
        Parameters
        -----------
            classifier(EventClassifier): classifier to be stored
            filename(str): file to store classifier in
                
        Returns
        ------------
            Nothing
        
        """
        with open(filename, 'w') as myFile:
            #store tag_counters
            for event in classifier._tag_counters:
                myFile.write(event + "|||")
                for key in classifier._tag_counters[event]:
                    myFile.write(key + ":" + str(classifier._tag_counters[event][key]) + ",")
                myFile.write(";;;")
            #storoe doc importance
            myFile.write('\n')
            myFile.write(str(classifier._doc_importance))
            #store extra features
            myFile.write('\n')
            for event in classifier._extra_features:
                myFile.write(event + "|||")
                for key in classifier._extra_features[event]:
                    myFile.write(key + ":" + str(classifier._extra_features[event][key]) + ",")
                myFile.write(";;;")
        
    
    @staticmethod
    def load_classifier(filename):
        """
        loads classifier from the given txt file and returns it
        
        Parameters
        -----------
            filename(str): file to store classifier in
                
        Returns
        ------------
            EventClassifier
        """
        with open(filename, 'r') as my_file:
            temp = my_file.readline()
            
            #obtain the _tag_features from first line
            d1 = {}
            collection = temp.split(";;;") #separates different events in collection
            for event in collection:
                e_split = event.split("|||") #split event from their tags
                if(len(e_split) == 2):
                    #initialize new event entry
                    key = e_split[0].strip()
                    d1[key] = defaultdict(float)
                    
                    #split each tag and add them to the event dict
                    t_split = e_split[1].strip().split(",")
                    for tag in t_split:
                        d_split = tag.split(':')
                        if(len(d_split) == 2):
                            d1[key][d_split[0]] = float(d_split[1])
            
            #obtain document importance
            document_importance = float(my_file.readline().strip())
            
            #obtain any extra features            
            d4 = {}
            temp = my_file.readline()
            collection = temp.split(";;;") #separates different events in collection
            for event in collection:
                e_split = event.split("|||") #split event from their tags
                if(len(e_split) == 2):
                    #initialize new event entry
                    key = e_split[0].strip()
                    d4[key] = {}
                    
                    #split each tag and add them to the event dict
                    t_split = e_split[1].strip().split(",")
                    for tag in t_split:
                        d_split = tag.split(':')
                        if(len(d_split) == 2):
                            d4[key][d_split[0]] = d_split[1]
            
            
        return EventClassifier(tag_counters=d1, doc_importance = document_importance, extra_features=d4)
                
    def _add_event_tag_feature(self, event, tags_dict):
        """
        Adds to self._tag_counters the given event to its tags

        for each tag in tags_dict:        
            If the tag or event does not exist for the event,
                The tag will be added the the appropriate entry in self._tag_doc_freq
                and the appropriate entries in self._tag_counters will be modified.
            If the tag already exists for the event, 
                it's value will simply be added to the existing event[tag] value.

        Parameters
        ------------------
        event(str):
            event we wish to add or create with the given occurances
        tags_dict: dict{str:float}
            tags where the key is the tag name and the value is the number of times it appears

        
        Returns
        ------------------
        Nothing
        
        """ 
        
        #some primitive type checking
        if(type(event)!= str):
                raise TypeError("Each event must be a valid string")
                
        if(event in self._tag_counters):
            
            for tag in tags_dict:
                future_total = self._event_full_totals[event]+tags_dict[tag] #the number of tiems in event later
                future_count = self._tag_counters[event][tag]+tags_dict[tag]
                #if this tag is important enough to be considered in division
                if future_count/future_total >= self._doc_importance:
                    
                    #if the tag had not been important in the event previously
                    if(tag not in self._salient_tags[event]):
                        self._tag_doc_freq[tag] += 1
                        
                    #perform necessary additions since it had been deemed important
                    self._event_div_totals[event] += tags_dict[tag]
                    self._salient_tags[event].add(tag)
                
                #ok now add to our normal stuff
                self._tag_counters[event][tag] += tags_dict[tag]
                self._event_full_totals[event] += tags_dict[tag]
            
            #check if new additions changed the saliency at all
            for tag in self._salient_tags[event]:
                if self._tag_counters[event][tag]/future_total < self._doc_importance:
                    self._tag_doc_freq[tag] -= 1
                    self._salent_tags[event].remove(tag)
                    self._event_div_totals -= self._tag_counters[event][tag]
            
        else:     
            self._tag_counters[event] = Counter()
            self._salient_tags[event] = set()
            tag_total = 0.0
            try:
                for tag in tags_dict:
                    self._tag_counters[event][tag] += tags_dict[tag]
                    tag_total+= tags_dict[tag]
                self._event_full_totals[event] = tag_total
                
                #add tag to doc only if it occured a good enough amount of times
                for tag in self._tag_counters[event]:                 
                    if self._tag_counters[event][tag]/tag_total >= self._doc_importance:
                       self._tag_doc_freq[tag] += 1 
                       self._event_div_totals[event] += self._tag_counters[event][tag]
                       self._salient_tags[event].add(tag)
            except:
                raise ValueError("Values in event {} did not contain str tags with values of type either float or int".format(event))
        
             
        
                    
'''
###############################################################################
# NAIVE BAYES CLASSIFICATION
###############################################################################


    

def train_naive_Bayes(train_set):
   
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

'''

if __name__ == '__main__':
    # You may use this section as you please, but the contents won't be graded.
    from os.path import isfile, join
    from os import listdir
    final = fd.load_nested_dict("newer_features.txt")
    classifier = EventClassifier(tag_counters=final)
    #classifier = EventClassifier.load_classifier("current_classifier.txt")
    
    folder_path = "dict_img"
    filenames = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]
    s_count = 0
    err_count = 0
    for filename in filenames:
        if(filename.startswith('b')):
            if(filename != "badminton.txt"):
                continue
        temp = fd.file_to_dict(join(folder_path,filename))    
        ans = classifier.classify_feature(temp)
        if(ans is None):
            print "#######################################################"
            print "FAILURE: Mapped ", filename, "to None"
            print "#######################################################"
            err_count+=1
        elif filename.startswith(ans[0][0]):
            print "SUCCESS!!! FOR", filename,  "(", ans[0][1]-ans[1][1] 
            s_count+=1
        else:
            if(ans[0][0] == "birthday-party" and (filename.startswith('b') or filename.startswith("children_birthday"))):
                print "SUCCESS!!! FOR", filename,"(", ans[0][1]- ans[1][1], ")" 
                s_count+=1
            elif(ans[0][0] == "art-exhibition" and filename.startswith("exhib")):
                print "SUCCESS!!! FOR", filename, "(", ans[0][1]- ans[1][1], ")"
                s_count+=1
            else:
                print "#######################################################"
                print "FAILURE: Mapped ", filename, "to", ans[0][0], "(", ans[0][1]-ans[1][1],")" 
                print "#######################################################"
                err_count+=1
    print "FINAL STATISITCS:"
    print "\tsuccesses:", s_count, "\n\terrors:", err_count, "\nPercentage:", s_count/(err_count+s_count)
    
    
    
    
    