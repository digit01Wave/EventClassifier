# -*- coding: utf-8 -*-
"""
Created on Tue May 17 01:21:59 2016

@author: Jessica

This module is meant to mimic the properties of a topic map
"""

'''
Represents Events
Every event has a name and a set of objects it is connected to.
The object connections are stored in a dictionary whose value is the object's name
'''
class Event:    
    def __init__(self, name, objects=[]):
        self._name = name
        self._objects = {}
        for item in objects:
            self._objects[item.name] = item
    
    
    def addObject(self, obj=None):
        '''we add the given object if name not already contained by event.
        Successful insertons return 1, while unsuccessful ones return -1'''
        if(obj.name in self._objects):
            return -1
        else:
            self._objects[obj.name] = obj
            return 1
        
    def removeObject(self, name):
        '''removes object with given name from connected object dictionary
        Returns 1 for successful removal and -1 for unsuccessful'''
        try:
            del[name]
            return 1
        except:
            return -1

'''
Represents Objects
Every object has a name and a set of events it is connected to.
The event connections are stored in a dictionary whose value is the event's name
'''
class Object:
    def __init__(self, name, topic_type, events=[]):
        self._name = name
        self._topic_type = topic_type
        self._events = {}
        for item in events:
            self._events[item.name] = item
    

    def addEvent(self, event):
        '''we add the given event if name not already connected to object.
        Successful insertons return 1, while unsuccessful ones return -1'''
        if(event.name in self._events):
            return -1
        else:
            self._events[event.name] = event
            return 1
        
    def removeEvent(self, name):
        '''removes event with given name from connected event dictionary
        Returns 1 for successful removal and -1 for unsuccessful'''
        try:
            del[name]
            return 1
        except:
            return -1

            
class Topic_Map:
    def __init__(self):
        self._events = {}
        self._objects = {}
    
    def add_event(self, event):
        self._events[event.name] = event
        
    def add_object(self, obj):
        self._objects[obj.name] = obj

    def add_connection(self, e_name, o_name):
        try:
            self._events[e_name].addObject(self._object[o_name])
            self._objects[o_name].addEvent(self._events[e_name])
        except:
            raise KeyError("Given event_name [{}] and object_name [{}] do not\
            exist in this topic map".format(e_name, o_name))