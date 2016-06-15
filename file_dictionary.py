# -*- coding: utf-8 -*-
"""
Created on Tue May 31 02:10:18 2016

@author: jessica

This module deals with files and dictionary features
"""
from os import listdir
from os.path import isfile, join
from collections import defaultdict, Counter


def list_to_file(L, filename, append=False):
    """
    Writes a list of lists into the given file. It assumes each inner list has at
    least one item
    
    Example:
        input list: [[hey1, hey2, hey3], [hey4]]
        output file:
            hey1, hey2, hey3
            hey4
    
    Parameter
    -----------
    L (list[list[str]]): Lists of lists to write
    filename(str): location to store L
    
    Returns
    --------
    Nothing
    """
    action = 'w'
    if append:
        action = 'a'
    with open(filename, action) as myfile:
        for item in L:
            for i in xrange(len(item)-1):
                myfile.write(item[i]+',')
            myfile.write(item[-1]+'\n')

def file_to_list(filename):
    """
    Does the opposite of list_to_file. Takes in a filename and returns the nested list
    """
    ans = []
    with open(filename, 'r') as my_file:
        for line in my_file:
            ans.append([])
            temp = line.split(",")
            for item in temp:
                ans[-1].append(item.strip())
    if ans[-1] == []:
        ans.pop()
    return ans

        

def dict_to_file(d, filename):
    """
    Converts a feature dictionary to a text file and stores it in filename
    
    Parameters
    ----------
    d(dict{str:float}): dictionary to store
    filename(str): location to store dictionary
    
    Return
    ---------
    Nothing
    """
    with open(filename, 'w') as myfile:
        for key in d:
            try:
                myfile.write("{}:{}\n".format(key, d[key]))
            except:
                print("ERROR: Unable to input {}:{}".format(key, d[key]))

def file_to_dict(filename):
    """
    Converts a file into a dictionary (opposite of dict to file)
    
    Parameters:
    ---------------
    filename(str): name of file containing dictionary
    
    Returns:
    ---------
    dict{str:float}
    """
    d = defaultdict(float)
    with open(filename, 'r') as myFile:
        for line in myFile:
            temp = line.split(":")
            if(len(temp) == 2):
                d[temp[0]] = float(temp[1].strip())
    return d


def folder_to_dict(folder_path):
    """
    Helper function that runs through an entire folder and creates a dictionary
    out of each internal file's contents.
    
    Parameter
    ---------------
    folder_path (str):
        Path to folder containing text files of dictionaries with format key:value\n
    
    Returns
    ----------------
    dict{str:defaultdict{str, float or int}}
        Dictionary that maps a filename to the dictionry of that filename in the folder_path
    """
    ans = {}
    filenames = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]
    for filename in filenames:
        ans[filename[:-4]] = file_to_dict(join(folder_path,filename))    
    return ans

def zip_entries(dict_list):
    """
    Creates a new dictionary that has all the keys in each dict_list item added
    together
    
    Parameter
    -----------
    dict_list(list[dict{str:float}]): 
        List of dictionaries to zip together
    
    Return
    ------------
    defaultdict(str:float): Final zipped up dictionary
    """
    ans = defaultdict(float)
    for d in dict_list:
        for key in d:
            ans[key]+=d[key]
    return ans

def combine_entries(d, keys, new_key):
    """
    combines the given two keys in the dictionary and deletes the original entries
    
    Parameters
    ----------------
        d (dict{str:defaultdict{str:float or int}}): (can also be a Counter) 
            dictionary we wish to combine the keys of i
        keys(list[str]): 
            the keys we wish to combine 
        new_key(str):
            the name of the new key that the combined keys will be in
    Returns
    ----------------
    Nothing
    """
    #perform checks
    for key in keys:
        if key not in d:
            raise ValueError("all keys (2nd parameter) must be in the dictionary")
    
    if(new_key not in d):
        d[new_key] = defaultdict(float)
    for key in keys:
        if(key != new_key):
            for item in d[key]:
                d[new_key][item]+=d[key][item]
            del d[key]

def super_combo(d):
    L = ["art-exhibition", 'awards-ceremony',"badminton", 'bday', 'beach-bonfire', 'beach-play', 'birthday-party', 'car-accident', 'concert', 'concertAll', 'croquet', 'cruise', 'easter', 'graduation', 'luncheon', 'office-meeting', 'office-meetingAll', 'rock-climbing', 'rowing', 'sailing', 'seminar', 'skiing', 'tennis', 'wedding']
    for item in L:
        try:
            combine_entries(d, [item+"50", item+"50_200"], item)
            print "successfully combined to make", item
        except:
            print "############################################################"
            print "FAILURE: did not combine {} and {} to become {}".format(item+"50.txt", item+"50_200.txt", item)
            print "############################################################"
            
def full_combo(d):
    final_d = Counter()
    for event in d:
        for tag in d[event]:
            final_d[tag]+=d[event][tag]
    return final_d
    
def store_nested_dict(d, filename):
    with open(filename, 'w') as myFile:
        for key in d:
            myFile.write(key + "|||")
            for key2 in d[key]:
                myFile.write(key2 + ":" + str(d[key][key2]) + ",")
            myFile.write(";;;")

def load_nested_dict(filename):
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
    return d1