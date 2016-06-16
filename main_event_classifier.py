# -*- coding: utf-8 -*-
"""
Created on Tue May 17 10:34:02 2016

@author: jessica
"""
from __future__ import division

#imports
import my_googlenet_classify as gClassify
from collections import defaultdict
import operator
from collections import Counter
import math
import matplotlib.pyplot as plt
import re
import random
import numpy as np
import csv
from os.path import isfile, join
from os import listdir

#convenient imports to other modules
import file_dictionary as fd
import datascrape as ds
import event_concept as ec


class EventClassifier:
    """
    Implementation of a classifier for image collection event recognition.
    
    Keeps track of how many times different tags/concepts appear for each event
    and which ones are 'salient', appearing >= doc_importance of the time
    """
    
    def __init__(self, tag_counters = None, doc_importance = 0.008, naive_importance=0.16):
        """
        instance variables
        ------------------
        self._tag_counters (dict{str:Counter}):
            Dictionary that maps each event to a corresponding dictionary of tags
            Each tag dictionary maps a tag to the number of times it appears in the event
        
        self._salient_tags (dict{str:set(str)})
            Dictionary that maps each event to the set of its tags that 
            appears>doc_importance of the time 
        
        self._event_full_totals: Counter
            Dictionary that maps each event to the total number of items in it.
            
        self._event_div_totals: Counter
            Dictionary that maps eaach event to the number of salient items in it.
        
        self._tag_doc_freq (dict{str,int}): 
            Dictionary that maps a tag to the number of labels it occurs saliently in
            This will be used to help calculate idf (inverse-document-frequency)
            
        self._doc_importance (flaot):
            What percentage of the event the tag must appear in before it is
            considered to be added as "existing" in that particular event
        
        self._naive_importance:
            What percentage of unclassified event a tag must appear for it to be
            considered a mandatory presence in the trained classifier
        """
        #initialize instance variables
        self._doc_importance = doc_importance
        self._tag_counters = {}
        self._event_full_totals = Counter()
        self._event_div_totals = Counter() 
        self._salient_tags = {}
        self._tag_doc_freq = Counter()
        self._naive_importance=naive_importance
        
        #If given a tag_counters given, adjust appropriate variables accordingly
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
            
    def print_important_labels(self, tag):
        """
        Finds and prints to console all the event labels that have the given tag
        as a salient feature
        """
        temp = {}
        for event in self._salient_tags:
            if tag in self._salient_tags[event]:
                temp[event] = self._tag_counters[event][tag]/self._event_div_totals[event]
        
        sorted_x = sorted(temp.items(), key=operator.itemgetter(1), reverse=True)  
        for item in sorted_x:
            print '\t', item[0], ': ', item[1]
    
    def _print_single_important(self, event):
        """
        helper function for print_important_features
        """
        total = self._event_div_totals[event]
        print event
        print "\tOriginal Total:", self._event_full_totals[event] 
        print "\tSalient Tag Total:", total
        
        temp = {}
        for tag in self._salient_tags[event]:
            temp[tag] = self._tag_counters[event][tag]/total
        
        #sort temp and print it in order
        sorted_x = sorted(temp.items(), key=operator.itemgetter(1), reverse=True)
        for item in sorted_x:
            print '\t', item[0], ': ', item[1]
        
    def salient_to_csv(self, file_path):
        with open(file_path, 'wb') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            for event in self._salient_tags:
                writer.writerow([event])
                for tag in self._salient_tags[event]:
                    writer.writerow([tag, self._tag_counters[event][tag]/self._event_div_totals[event]])

    def train_features(self, feature_list):
        """
        trains classifier on given feature list
        
        Adds to self._tag_counters maps events to their tags and adjustes 
        appropriate counts in the appropriate dictionaries.

        Parameters
        ------------------
        feature_list: list[(str,dict{str:float or int})]
            List of tuples where the first item is the event title, and the second
            item is its dictionary of tags with the number of times they occured 
            in given event. (e.g. {'wedding':{'groom':27, 'cake':3.0}}  

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
        print "Finished trainign process"
    
        
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
        feature_tag_dict (dict{str:float or int}):
            Features that map a tag to the number of times it had appeared
        alpha (float):
            The weight given the the number of occurances in given feature_dict
        
        Returns
        -------------
        list[tuple(str:float)]
            List of top-10 event labels ordered by score.
            Each tuple consists of the event label and score.
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
                elif feature_tag_dict[tag]/feature_tag_total > self._naive_importance:
                    #apply penalty if commonly occuring tag is not salient in event
                    if(self._event_full_totals[event]==0):
                        weight=0
                        break
                    weight *= self._tag_counters[event][tag]/self._event_full_totals[event]                  
            
            #apply penalty
            scores[event] *= weight
            
            total+=scores[event]
        
        if(total == 0.0):
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
        
            #storoe naive_importance
            myFile.write('\n')
            myFile.write(str(classifier._naive_importance))
        
    
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
            
            #obtain naive_importance         
            n_importance = float(my_file.readline().strip())
            
            
        return EventClassifier(tag_counters=d1, doc_importance = document_importance, naive_importance = n_importance)
    
    def get_labels(self):
        """
        Returns all the event labels in the classifier in the form of list[str]
        """
        ans = set()
        for event in self._tag_counters:
            ans.add(event)
        return ans   

         
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
            toRemove = []
            for tag in self._salient_tags[event]:
                if self._tag_counters[event][tag]/future_total < self._doc_importance:
                    self._tag_doc_freq[tag] -= 1
                    toRemove.append(tag)
                    self._event_div_totals[event] -= self._tag_counters[event][tag]
            for tag in toRemove:
                self._salient_tags[event].remove(tag)
            
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
        
             
###############################################################################
  #Evaluation Section
###############################################################################
def plot_data(data, x_axis=None, x_label="", y_label="", title=""):
    """
    Returns a plot for given data. Use for visualization.
    
    Parameters
    ---------------
    data(list[float]): y-values
    x-axis(list[float]): 
        Corresponding x-values
        If none are given, then will just count from 1
    x_label(str): x-axis label
    y_label(str):y-axis label
    title(str): title
    """
    if(x_axis is None):
        x_axis = [i+1 for i in xrange(len(data))]
    plt.plot(x_axis, data)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.ylim(ymin=0)
    plt.show()

    
def sdf(d, name, num, to_add=None):
    fd.dict_to_file(d, 'newer/dict_img/'+name+num+'.txt')
    if to_add is not None:
        for key in to_add:
            d[key]=to_add[key]
    fd.dict_to_file(d, 'newer/specialized/'+name+num+'.txt')
    
    
def convert_data_to_dict(folder_path, filenames):
    """
    Converts given list of filenames into a dictionary
    
    Parameter
    -----------------
    folder_path(str):
        folder that all the files are under
    filenames(list[str]):
        list of all the filenames to convert
    
    Returns
    -----------------
    dictionary{str:Counter}
        Dictionary that maps each event to its respective tag Counter
    """
    ans = {}    
    for filename in filenames:
        temp_split= re.split('[\d.]+', filename)
        category = temp_split[0]
        if category not in ans:
            ans[category]=defaultdict(float)
        temp = fd.file_to_dict(join(folder_path,filename))
        for key in temp:
            ans[category][key] += temp[key]
    return ans
    
def PEC_cross_validate(n_fold, original=True, doc_imp=0.008, naive_imp=0.16):
    """
    Perform n-fold cross validation on the appropriate PEC data and returns the 
    n-length list of accuracies from each fold, along with which labels were 
    incorrectly labeled, and the classifier.
        
    Parameters
    ----------------
    n_fold(int): 
        How many folds desired
    original(boolean): 
        Whether or not to use the 1000-class classifier as opposed to the 1002
        class classifier
    doc_imp (float): 
        Parameter for EventClassifier
    naive_imp(float): 
        Parameter for EventClassifier
    
    Returns
    -------------------
    tuple(list[floats], dictionary{str:float}, classifier)
        3-length tuple with following information:
            [1] List of average accuracies across all k folds, where each item 
                is the average for the top-index choices. 
            [2] Dictionary that that maps each mislabeled event to the incorrect 
                event, which is in turn mapped to the average percent this mapping occured.
            [3] the trained classifier
    """
    #get all_files
    folder_path = "dict_img" if original else "specialized"
    filenames = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]
    random.seed(5)
    random.shuffle(filenames)  

    #cross validatate
    M = int(math.floor(len(filenames)/n_fold))
    accuracy_list = [] #to record trained model's accuracy on validation set
    start = 0 #start index of current fold
    fold_num = 1 #fold we are on
    type_dict={}
    
    
    while fold_num <= n_fold:
        test_set = filenames[start:fold_num*M]
        train_dict = convert_data_to_dict(folder_path, filenames[0:start]+ filenames[fold_num*M:])
        classifier = EventClassifier(tag_counters=train_dict, doc_importance=doc_imp, naive_importance=naive_imp)
        temp_accuracy, temp_type_dict = test_PEC(classifier, folder_path, test_set)
        accuracy_list.append(temp_accuracy)
        for event in temp_type_dict:
            if event not in type_dict:
                type_dict[event] = defaultdict(float)
            for tag in temp_type_dict[event]:
                type_dict[event][tag]+=temp_type_dict[event][tag]/n_fold
        start = fold_num*M
        fold_num+=1
    
    return(np.average(accuracy_list, axis=0), type_dict, classifier)
    
def test_PEC(classifier, folder_path, test_list):
    """
    Uses classifier to test on the given list of files, prints appropriate information
    and returns the overall accuracy along with the error percentages for each type.
    
    Parameters
    ------------------
    classifier(EventClassifier): the classifier we are testing with
    folder_path(str): folder containing all the features we are testing on
    test_list(list[str]): list of filenames in the folder that we wish to test on
    
    Return
    -------------
    tuple(float, dict{str:dict{str:float}}):
        Returns a tuple where the first item is the overall accuracy and the second
        item is a dictionary that maps each event to incorrecly labled tags, which
        in turn are mapped to the counts of all the tags they got incorrect
    """
    err_list = [0]*10
    type_dict = {}
    type_count=defaultdict(int)
    for filename in test_list:
        temp_split= re.split('[\d.]+', filename)
        category = temp_split[0]
        type_count[category]+=1
        temp = fd.file_to_dict(join(folder_path,filename))    
        ans = classifier.classify_feature(temp)
        if(ans is None):
            for i in xrange(len(err_list)):
                err_list[i]+=1
            if category not in type_dict:
                type_dict[category] = defaultdict(float)
            type_dict[category][None] += 1
        elif category == ans[0][0]:
            pass
        else:
            
            if(ans[0][0] == "birthday-party" and (category.startswith('bd') or category == "children_birthday")):
                pass
            else:
                err_list[0]+=1
                if category not in type_dict:
                    type_dict[category] = defaultdict(float)
                type_dict[category][ans[0][0]] += 1
                for i in xrange(1, len(err_list)):
                    try:
                        if category == ans[i][0]:
                            break 
                        elif (ans[i][0] == "birthday-party" and (category.startswith('bd') or category == "children_birthday")):
                            break
                        else:
                            err_list[i] += 1
                    except:
                        raise ValueError("Category [{}] is not one of classifier's categories".format(category))
    
    f=len(test_list)           
    accuracy = [0]*10
    for i in xrange(len(err_list)):
        accuracy[i] = (f-err_list[i])/f

    #normalize all the errors
    for event in type_dict:
        for tag in type_dict[event]:
            type_dict[event][tag] = type_dict[event][tag]/type_count[event]*100    
    return accuracy, type_dict
    
def test_bing(original=True, only_8=True, doc_imp=0.008, naive_imp=0.16):
    """
    Performs test using bing trained classifier.
    
    Parameters
    -----------
    original(boolean):
        Whether or not to use the original inception classifier (no cake or christmas tree)
    keep-8(boolean):
        Whether or not to keep only the 8-classes or include all extra classes
    doc_imp (float): 
        Parameter for EventClassifier
    naive_imp(float): 
        Parameter for EventClassifier
    
    Returns
    ---------
    tuple(list[float], dict{str:dict{str, float}}, EventClassifier):
        A 3 length tuple with the follwoing items:
            (0) a list of the accuracies in the top k guessed labels,
            (1) a dictionary that maps each mislabeled collection to the 
                distribution of which label it labeled the collection
            (2) the trained classifier
    """
    to_load = "no_cake_features.txt" if original else "special_features.txt"
   
    temp_final = fd.load_nested_dict(to_load)

    #keep only the 8-classes
    if(only_8):
        final = {}
        for event in ['wedding', 'birthday-party', 'skiing', 'concert', 'christmas', 'cruise', 'exhibition', 'graduation']:
            final[event] = temp_final[event]
    else:
        final = temp_final
            
    
    #create trained classifier
    classifier = EventClassifier(tag_counters=final, doc_importance=doc_imp, naive_importance=naive_imp)
    
    #obtain files to test on
    folder_path = "dict_img" if original else "specialized"
    filenames = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]  
    
    type_dict = {} #will store event{event_mapped:count} 
    type_count=defaultdict(int)
    err_list = [0]*10
    for filename in filenames:
        temp_split= re.split('[\d.]+', filename)
        category = temp_split[0]
        type_count[category] += 1
        temp = fd.file_to_dict(join(folder_path,filename))    
        ans = classifier.classify_feature(temp)
        if(ans is None):
            for i in xrange(len(err_list)):
                err_list[i]+=1
            if category not in type_dict:
                type_dict[category] = defaultdict(float)                
            type_dict[category][None] += 1
            
        elif category == ans[0][0]:
            #print "SUCCESS!!! FOR", filename,  "(", ans[0][1]-ans[1][1] 
            pass
        else:
            if(ans[0][0] == "birthday-party" and (category.startswith('bd') or category == "children_birthday")):
                #print "SUCCESS!!! FOR", filename,"(", ans[0][1]- ans[1][1], ")" 
                pass
            else:
                print "FAILED. MAPPED {} to {}".format(filename, ans[0][0])
                #if still wrong keep going down k-labels until correct
                err_list[0]+=1
                if category not in type_dict:
                    type_dict[category] = defaultdict(float)                
                type_dict[category][ans[0][0]] += 1
                for i in xrange(1, len(err_list)):
                    if category == ans[i][0]:
                        break 
                    elif (ans[i][0] == "birthday-party" and (category.startswith('bd') or category == "children_birthday")):
                        break
                    else:
                        err_list[i] += 1
    
    f=len(filenames)

    #determine and normalize accuracy and errors into percentages           
    accuracy = [0]*10
    for i in xrange(len(err_list)):
        accuracy[i] = (f-err_list[i])/f
    for event in type_dict:
        for tag in type_dict[event]:      
            type_dict[event][tag] = type_dict[event][tag]/type_count[event]*100 #converting erros to percentage of error
            
    return accuracy, type_dict, classifier
    

###############################################################################
#Learn Best Parameters for EventClassifier
###############################################################################
"""
This part can be ignored unless one wants to find a different parameter.
The doc_importance and naive_importance is assumed independent in these tests
"""
def find_best_doc_param(fixed_naive, epsilon=.001, pec=True, orig=True):
    top = 0.25 #highest we can be
    bottom = 0.0 #lowest we can be
    best = 0.1 #current best parameter for naive
    if pec:
        func = lambda x: PEC_cross_validate(5, original=orig, doc_imp=x, naive_imp=fixed_naive)[0][0]
    else:
        func = lambda x: test_bing(original=orig, doc_imp=x, naive_imp=fixed_naive)[0][0]
    best_score=func(best)#PEC_cross_validate(5, original=True, doc_imp=best, naive_imp=fixed_naive)[0][0]
    while True:
        print 'current best', best
        print 'current score', best_score
        next_value = (best+bottom)/2
        if(best-next_value<epsilon):
            break
        temp = func(next_value)
        if temp > best_score:
            top=best
            best=next_value
            best_score=temp
        else:
            next_value = (best+top)/2
            if(next_value-best<epsilon):
                break
            temp = func(next_value)
            if temp>best_score:
                bottom=best
                best=next_value
                best_score=temp
            else:
                top=next_value
                bottom=(best+bottom)/2
    return best

def find_best_naive_param(fixed_doc, epsilon=.001, pec=True, orig=True):
    top = 0.25 #highest we can be
    bottom = 0.0 #lowest we can be
    best = 0.1 #current best parameter for naive
    if pec:
        func = lambda x: PEC_cross_validate(5, original=orig, doc_imp=fixed_doc, naive_imp=x)[0][0]
    else:
        func = lambda x: test_bing(original=orig, doc_imp=fixed_doc, naive_imp=x)[0][0]
    best_score=func(best)#PEC_cross_validate(5, original=True, doc_imp=best, naive_imp=fixed_naive)[0][0]
    while True:
        print 'current best', best
        print 'current score', best_score
        next_value = (best+bottom)/2
        if(best-next_value<epsilon):
            break
        temp = func(next_value)
        if temp > best_score:
            top=best
            best=next_value
            best_score=temp
        else:
            next_value = (best+top)/2
            if(next_value-best<epsilon):
                break
            temp = func(next_value)
            if temp>best_score:
                bottom=best
                best=next_value
                best_score=temp
            else:
                top=next_value
                bottom=(best+bottom)/2
    return best
                
            
##################################################
            #MAIN
#################################################
if __name__ == '__main__':   
    
    #a, t, c =  PEC_cross_validate(5, original=False, doc_imp=0.008, naive_imp=0.16)
    a, t, c = test_bing(original=False, only_8=False, doc_imp=0.008, naive_imp=0.16)#PEC_cross_validate(5, True)
    
    print 'Accuracy List:', a
    print "Error statistics"
    print('type errors')
    print '\nPERCENTAGE ERRORS'
    for event in sorted(t):
        current = 0
        print event 
        for tag in sorted(t[event]):
            print '\t{}: {:.2f}'.format(tag,t[event][tag])
            current+=t[event][tag]
        print '\tTOTAL:{:.2f}'.format(100-current)
        
        temp_final = fd.load_nested_dict("special_features.txt")

    pass
    