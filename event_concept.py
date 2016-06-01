# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 07:13:03 2016

@author: jessica
"""


    
def get_objects(event, dictionary, syn_lvl = 1):
    """
    Takes in the event, and returns all objects in the dictionary that match
    said event.
    
    Parameters
    ------------
    event(str): Event that person wishes to know about
    dictionary(set): set of words we are allowed to use
    syn_lvl (int): the number of levels we are allowed to look in wordnet synonyms 
    
    Returns
    -----------
    list[str]: list of objects found in dictionary that match the said event
    """
    pass


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