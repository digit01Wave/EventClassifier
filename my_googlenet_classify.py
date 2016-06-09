
# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Simple image classification with Inception.

Run image classification with Inception trained on ImageNet 2012 Challenge data
set.

This program creates a graph from a saved GraphDef protocol buffer,
and runs inference on an input JPEG image. It outputs human readable
strings of the top 5 predictions along with their probabilities.

Change the --image_file argument to any jpg image to compute a
classification of that image.

Please see the tutorial and website for a detailed description of how
to use this script to perform image recognition.

https://tensorflow.org/tutorials/image_recognition/
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import re
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf

#myimports
from collections import defaultdict
from os import listdir
from os.path import isfile, join



# classify_image_graph_def.pb:
#   Binary representation of the GraphDef protocol buffer.
# imagenet_synset_to_human_label_map.txt:
#   Map from synset ID to a human readable string.
# imagenet_2012_challenge_label_map_proto.pbtxt:
#   Text representation of a protocol buffer mapping a label to synset ID.
"""Path to classify_image_graph_def.pb, """
"""imagenet_synset_to_human_label_map.txt, and """
"""imagenet_2012_challenge_label_map_proto.pbtxt."""
model_dir = '/tmp/imagenet'


"""Display this many predictions."""
num_top_predictions = 5

# pylint: disable=line-too-long
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
# pylint: enable=line-too-long


class NodeLookup(object):
  """Converts integer node ID's to human readable labels."""

  def __init__(self,
               label_lookup_path=None,
               uid_lookup_path=None):
    if not label_lookup_path:
      label_lookup_path = os.path.join(
          model_dir, 'imagenet_2012_challenge_label_map_proto.pbtxt')
    if not uid_lookup_path:
      uid_lookup_path = os.path.join(
          model_dir, 'imagenet_synset_to_human_label_map.txt')
    self.node_lookup = self.load(label_lookup_path, uid_lookup_path)

  def load(self, label_lookup_path, uid_lookup_path):
    """Loads a human readable English name for each softmax node.

    Args:
      label_lookup_path: string UID to integer node ID.
      uid_lookup_path: string UID to human-readable string.

    Returns:
      dict from integer node ID to human-readable string.
    """
    if not tf.gfile.Exists(uid_lookup_path):
      tf.logging.fatal('File does not exist %s', uid_lookup_path)
    if not tf.gfile.Exists(label_lookup_path):
      tf.logging.fatal('File does not exist %s', label_lookup_path)

    # Loads mapping from string UID to human-readable string
    proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readlines()
    uid_to_human = {}
    p = re.compile(r'[n\d]*[ \S,]*')
    for line in proto_as_ascii_lines:
      parsed_items = p.findall(line)
      uid = parsed_items[0]
      human_string = parsed_items[2]
      uid_to_human[uid] = human_string

    # Loads mapping from string UID to integer node ID.
    node_id_to_uid = {}
    proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()
    for line in proto_as_ascii:
      if line.startswith('  target_class:'):
        target_class = int(line.split(': ')[1])
      if line.startswith('  target_class_string:'):
        target_class_string = line.split(': ')[1]
        node_id_to_uid[target_class] = target_class_string[1:-2]

    # Loads the final mapping of integer node ID to human-readable string
    node_id_to_name = {}
    for key, val in node_id_to_uid.items():
      if val not in uid_to_human:
        tf.logging.fatal('Failed to locate: %s', val)
      name = uid_to_human[val]
      node_id_to_name[key] = name

    return node_id_to_name

  def id_to_string(self, node_id):
    if node_id not in self.node_lookup:
      return ''
    return self.node_lookup[node_id]



def create_graph():
  """Creates a graph from saved GraphDef file and returns a saver."""
  # Creates graph from saved graph_def.pb.
  with tf.gfile.FastGFile(os.path.join(
      model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')
    
def create_graph2(graph_path = '/home/jessica/Documents/classifier_files/retrained_graph.pb'):
  """Creates a graph from saved GraphDef file and returns a saver."""
  # Creates graph from saved graph_def.pb.
  with tf.gfile.FastGFile(graph_path, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')


def run_inference_on_image(image):
  """Runs inference on an image.

  Args:
    image: Image file name.

  Returns:
    list[tuple(str, float)]
        list of tuples (tag, score) where tag can be comma diliminated (e.g. "groom, bridegroom")
  """
  if not tf.gfile.Exists(image):
    tf.logging.fatal('File does not exist %s', image)
  image_data = tf.gfile.FastGFile(image, 'rb').read()

  # Creates graph from saved GraphDef.
  create_graph()

  with tf.Session() as sess:
    # Some useful tensors:
    # 'softmax:0': A tensor containing the normalized prediction across
    #   1000 labels.
    # 'pool_3:0': A tensor containing the next-to-last layer containing 2048
    #   float description of the image.
    # 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG
    #   encoding of the image.
    # Runs the softmax tensor by feeding the image_data as input to the graph.
    softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
    predictions = sess.run(softmax_tensor,
                           {'DecodeJpeg/contents:0': image_data})
    predictions = np.squeeze(predictions)

    # Creates node ID --> English string lookup.
    node_lookup = NodeLookup()

    ans = []
    top_k = predictions.argsort()[-num_top_predictions:][::-1]
    for node_id in top_k:
      human_string = node_lookup.id_to_string(node_id)
      score = predictions[node_id]
      ans.append((human_string, score))
    return ans

def get_collection_tags(image_collection_links, rel_threshold=0.3, current_dict = None):
    """
    Returns a counter dict representing the features for the given list of image urls. The features will be used to
    classify the collection to an event. The dict includes the following fields (the order doesn't matter):

    Parameters
    ----------
    image_collection : list[str]
        List of image file paths (can be relative or absolute)
    rel_threshold:float:
        only take tags that have a probability of at least this fraction of the first result

    Returns
    -------
    Dict[dict[str:count] or str:str]
        3 features for each collection.

    """
    import urllib
    
    filename ="/home/jessica/Documents/temp.jpg"
    if(current_dict == None):
        ans = defaultdict(float)
    else:
        ans = current_dict
    err_count = 0
    
    # Creates graph from saved GraphDef.
    create_graph()
            
    for i, img_link in enumerate(image_collection_links):
        if(img_link.endswith("png")): #skip items that are png images
            print("###########################################################")
            print("Error: picture is a png (need jpg)")
            print("###########################################################")  
            continue
        #get url
        try:
            print("{}: retreiving url({})".format(i, img_link))
            urllib.urlretrieve(img_link, filename)
            print("retrieval complete!")
        except:
            print("###########################################################")
            print("Error: URL is not retreivalbe")
            print("###########################################################")            
            continue
        
        print("now processing...")
        
        
        try:
            image_data = tf.gfile.FastGFile(filename, 'rb').read()
            
            with tf.Session() as sess:
                # Some useful tensors:
                # 'softmax:0': A tensor containing the normalized prediction across
                #   1000 labels.
                # 'pool_3:0': A tensor containing the next-to-last layer containing 2048
                #   float description of the image.
                # 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG
                #   encoding of the image.
                # Runs the softmax tensor by feeding the image_data as input to the graph.
                softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
                predictions = sess.run(softmax_tensor,
                                       {'DecodeJpeg/contents:0': image_data})
                predictions = np.squeeze(predictions)
            
                # Creates node ID --> English string lookup.
                node_lookup = NodeLookup()
            
                temp_tags = []
                top_k = predictions.argsort()[-num_top_predictions:][::-1]
                for node_id in top_k:
                    human_string = node_lookup.id_to_string(node_id)
                    score = predictions[node_id]
                    temp_tags.append((human_string, score))
                
        except:
            print("ERROR ###################################")
            err_count +=1
            print("Could not give tags to image.")
            print("ERROR ###################################")
            continue
        
        #keep only the tags that overcome the certain threshold
        print("tags = [{}]".format(temp_tags))
        thresh = temp_tags[0][1]*rel_threshold #get top score of first result to find threshold
                
        for tag_score in temp_tags:
            if(tag_score[1] > thresh):
                tag_split = tag_score[0].split(",")
                ans[tag_split[0].strip()] += 1 
                print("\tadded " + tag_score[0])
    print("Final Error Count:", err_count)
    return ans
    
def get_folder_tags(folder_path, rel_threshold=0.3, current_dict = None):
    """
    Returns a counter dict representing the features for the given list of image urls. The features will be used to
    classify the collection to an event. The dict includes the following fields (the order doesn't matter):

    Parameters
    ----------
    image_collection : list[str]
        List of image file paths (can be relative or absolute)
    rel_threshold:float:
        only take tags that have a probability of at least this fraction of the first result

    Returns
    -------
    Dict[dict[str:count] or str:str]
        3 features for each collection.

    """
    
    if(current_dict == None):
        ans = defaultdict(float)
    else:
        ans = current_dict
    err_count = 0
    
    # Creates graph from saved GraphDef.
    create_graph()
    

    im_names = [join(folder_path,f) for f in listdir(folder_path) if isfile(join(folder_path, f))]

    for i, filename in enumerate(im_names):
        print("{}: now processing ({})".format(i, filename))
        if(not filename.endswith(".jpg")):
            print("SKIPPING THIS ONE BECAUSE NOT A PHOTO~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            continue
        
        try:
            image_data = tf.gfile.FastGFile(filename, 'rb').read()
            
            with tf.Session() as sess:
                # Some useful tensors:
                # 'softmax:0': A tensor containing the normalized prediction across
                #   1000 labels.
                # 'pool_3:0': A tensor containing the next-to-last layer containing 2048
                #   float description of the image.
                # 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG
                #   encoding of the image.
                # Runs the softmax tensor by feeding the image_data as input to the graph.
                softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
                predictions = sess.run(softmax_tensor,
                                       {'DecodeJpeg/contents:0': image_data})
                predictions = np.squeeze(predictions)
            
                # Creates node ID --> English string lookup.
                node_lookup = NodeLookup()
            
                temp_tags = []
                top_k = predictions.argsort()[-num_top_predictions:][::-1]
                for node_id in top_k:
                    human_string = node_lookup.id_to_string(node_id)
                    score = predictions[node_id]
                    temp_tags.append((human_string, score))
                
        except:
            print("ERROR ###################################")
            err_count +=1
            print("Could not give tags to image.")
            print("ERROR ###################################")
            continue
        
        #keep only the tags that overcome the certain threshold
        print("tags = [{}]".format(temp_tags))
        thresh = temp_tags[0][1]*rel_threshold #get top score of first result to find threshold
                
        for tag_score in temp_tags:
            if(tag_score[1] > thresh):
                tag_split = tag_score[0].split(",")
                ans[tag_split[0].strip()] += 1 
                print("\tadded " + tag_score[0])
    print("Final Error Count:", err_count)
    return ans
        
###############################################################################
        #NEW CLASSIFIER
###############################################################################

special_nodes = {0:'cake', 1:'christmas tree', 2:'null'}

def get_folder_tags2(folder_path, threshold=0.8, graph_path='/home/jessica/Documents/classifier_files/retrained_graph.pb', current_dict = None):
    """
    Returns a counter dict representing the features for the given list of image urls. The features will be used to
    classify the collection to an event. The dict includes the following fields (the order doesn't matter):

    Parameters
    ----------
    image_collection : list[str]
        List of image file paths (can be relative or absolute)
    rel_threshold:float:
        only take tags that have a probability of at least this fraction of the first result

    Returns
    -------
    Dict[dict[str:count] or str:str]
        3 features for each collection.

    """
    
    if(current_dict == None):
        ans = defaultdict(float)
    else:
        ans = current_dict
    err_count = 0
    
    # Creates graph from saved GraphDef.
    create_graph2(graph_path)
    

    im_names = [join(folder_path,f) for f in listdir(folder_path) if isfile(join(folder_path, f))]

    for i, filename in enumerate(im_names):
        print("{}: now processing ({})".format(i, filename))
        if(not filename.endswith(".jpg")):
            print("SKIPPING THIS ONE BECAUSE NOT A PHOTO~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            continue
        
#        try:
        image_data = tf.gfile.FastGFile(filename, 'rb').read()
        
        with tf.Session() as sess:
            # Some useful tensors:
            # 'softmax:0': A tensor containing the normalized prediction across
            #   1000 labels.
            # 'pool_3:0': A tensor containing the next-to-last layer containing 2048
            #   float description of the image.
            # 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG
            #   encoding of the image.
            # Runs the softmax tensor by feeding the image_data as input to the graph.
            softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
            predictions = sess.run(softmax_tensor,
                                   {'DecodeJpeg/contents:0': image_data})
            predictions = np.squeeze(predictions)
        
            # Creates node ID --> English string lookup.
            #node_lookup = NodeLookup()
        
            temp_tags = []
            top_k = predictions.argsort()[-num_top_predictions:][::-1]
            for node_id in top_k:
                #human_string = node_lookup.id_to_string(node_id)
                score = predictions[node_id]
                temp_tags.append((node_id, score))
                
                
#        except:
#            print("ERROR ###################################")
#            err_count +=1
#            print("Could not give tags to image.")
#            print("ERROR ###################################")
#            continue
        
        #keep only the tags that overcome the certain threshold
        print("tags = [{}]".format(temp_tags))
                
        for tag_score in temp_tags:
            if(tag_score[0] != 2 and tag_score[1] > threshold):
                ans[special_nodes[tag_score[0]]] += 1 
                print("\tadded " + special_nodes[tag_score[0]])
    print("Final Error Count:", err_count)
    return ans
    
def get_collection_tags2(image_collection_links, threshold = 0.8, current_dict = None):
    """
    Returns a counter dict representing the features for the given list of image urls. The features will be used to
    classify the collection to an event. The dict includes the following fields (the order doesn't matter):

    Parameters
    ----------
    image_collection : list[str]
        List of image file paths (can be relative or absolute)
    rel_threshold:float:
        only take tags that have a probability of at least this fraction of the first result

    Returns
    -------
    Dict[dict[str:count] or str:str]
        3 features for each collection.

    """
    import urllib
    
    filename ="temp.jpg"
    if(current_dict == None):
        ans = defaultdict(float)
    else:
        ans = current_dict
    err_count = 0
    
    # Creates graph from saved GraphDef.
    create_graph2()
            
    for i, img_link in enumerate(image_collection_links):
        if(img_link.endswith("png")): #skip items that are png images
            print("###########################################################")
            print("Error: picture is a png (need jpg)")
            print("###########################################################")  
            continue
        #get url
        try:
            print("{}: retreiving url({})".format(i, img_link))
            urllib.urlretrieve(img_link, filename)
            print("retrieval complete!")
        except:
            print("###########################################################")
            print("Error: URL is not retreivalbe")
            print("###########################################################")            
            continue
        
        print("now processing...")
        
        
        try:
            image_data = tf.gfile.FastGFile(filename, 'rb').read()
            
            with tf.Session() as sess:
                # Some useful tensors:
                # 'softmax:0': A tensor containing the normalized prediction across
                #   1000 labels.
                # 'pool_3:0': A tensor containing the next-to-last layer containing 2048
                #   float description of the image.
                # 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG
                #   encoding of the image.
                # Runs the softmax tensor by feeding the image_data as input to the graph.
                softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
                predictions = sess.run(softmax_tensor,
                                       {'DecodeJpeg/contents:0': image_data})
                predictions = np.squeeze(predictions)
            
                temp_tags = []
                top_k = predictions.argsort()[-num_top_predictions:][::-1]
                for node_id in top_k:
                    score = predictions[node_id]
                    temp_tags.append((node_id, score))
            
        except:
            print("ERROR ###################################")
            err_count +=1
            print("Could not give tags to image.")
            print("ERROR ###################################")
            continue
        
        #keep only the tags that overcome the certain threshold
        print("tags = [{}]".format(temp_tags))
                
        for tag_score in temp_tags:
            if(tag_score[0] != 2 and tag_score[1] > threshold):
                ans[special_nodes[tag_score[0]]] += 1 
                print("\tadded " + special_nodes[tag_score[0]])
    print("Final Error Count:", err_count)
    return ans

def maybe_download_and_extract():
  """Download and extract model tar file."""
  dest_directory = model_dir
  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (
          filename, float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
  tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def setup():
    maybe_download_and_extract()
  


if __name__ == '__main__':
    setup()
