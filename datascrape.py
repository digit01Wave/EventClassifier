# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 05:46:30 2016

@author: jessica

Module used to download and obtain image links from bing
Also includes methods to download from Krumbs
"""
import py_bing_search as bing
import urllib
import json

###############################################################################
#DATASCRAPING
###############################################################################
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
#1464739200000/1464825600000
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

def extract_krumbs_links(k_list):
    """
    from list of krumbs json dictionaries, extract just the link and returns a list of links
    """
    ans = []
    for item in k_list:
        ans.append(item["image_url"])
    return ans
    