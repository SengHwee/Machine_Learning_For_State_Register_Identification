import sys
sys.path.insert(1, "./myModules")

import numpy as np
import pandas as pd
import fastRELIC_v5
from collections import OrderedDict
from pprint import pprint
#All functions here, requires word_id.find_shapes output dictionary as an input

def vector_creation(word_id_shape):
	#Filtering
	shape_dict_DFF = {dff: word_id_shape[dff] for dff in word_id_shape if "DFF" in dff}
	pprint(shape_dict_DFF)
	#creation node types Dictionary
	values_types_dict = {}
	for key in shape_dict_DFF:
		values_types = sorted(set(shape_dict_DFF[key]))
		for type in values_types:
			values_types_dict[type] = []

	#Reordering dictionary keys
	shape_dict_DFF_list = sorted(shape_dict_DFF.keys(), key=lambda number: int(number.split("_")[1]))
	shape_dict_DFF = {shape_dict_DFF_list[i]: shape_dict_DFF[shape_dict_DFF_list[i]] for i in range(len(shape_dict_DFF_list))}

	#calculating number of node types there are per key in shape Dictionary
	for key in shape_dict_DFF:
		values_types = sorted(set(shape_dict_DFF[key]))
		for type in values_types_dict.keys():
			values_types_dict[type].append(shape_dict_DFF[key].count(type))
		shape_dict_DFF[key] = []

	#creating vector for number of node types in shape Dictionary
	for index, key_DFF in enumerate(shape_dict_DFF):
		for key_type in values_types_dict:
			shape_dict_DFF[key_DFF].append(values_types_dict[key_type][index])
	# print(shape_dict_DFF)

	return shape_dict_DFF



def cosine_similarity(word_id_shape, threshold):
	# Doing dot product of one key to the rest, and finding cos(theta)
	shape_dict_DFF = vector_creation(word_id_shape)
	similarity_score_dict = shape_dict_DFF.copy()
	for key_DFF in shape_dict_DFF:
		# similarity_score = []
		count = 0
		for key_DFF2 in shape_dict_DFF:
			cos_similarity = np.dot(np.asarray(shape_dict_DFF[key_DFF]), np.asarray(shape_dict_DFF[key_DFF2]))
			cos_similarity = cos_similarity/(np.linalg.norm(np.asarray(shape_dict_DFF[key_DFF])) * np.linalg.norm(np.asarray(shape_dict_DFF[key_DFF2])))
			# print (cos_similarity)
			# similarity_score.append(cos_similarity)
			if cos_similarity > threshold:
				count += 1
		# similarity_score_dict[key_DFF] = similarity_score
		similarity_score_dict[key_DFF] = count-1	# -1 to remove itself
		# similarity_score_dict[key_DFF] = np.average(similarity_score)
	# print(similarity_score_dict)
	return similarity_score_dict


def euclidean_distance(word_id_shape, threshold):
	shape_dict_DFF = vector_creation(word_id_shape)
	pprint(shape_dict_DFF)
	similarity_score_dict = shape_dict_DFF.copy()
	for key_DFF in shape_dict_DFF:
		# similarity_score = []
		count = 0
		for key_DFF2 in shape_dict_DFF:
			element_subtraction = np.subtract(np.asarray(shape_dict_DFF[key_DFF]), np.asarray(shape_dict_DFF[key_DFF2]))
			euclidean_dist = np.sqrt(np.sum(np.square(element_subtraction)))
			# print(euclidean_dist)
			# similarity_score.append(euclidean_dist)
			if euclidean_dist < threshold:
				count += 1
		# print(similarity_score, "\n")
		# similarity_score_dict[key_DFF] = similarity_score
		similarity_score_dict[key_DFF] = (count-1)/len(shape_dict_DFF.keys())	# -1 to remove itself
		# similarity_score_dict[key_DFF] = np.average(similarity_score)
	# print(similarity_score_dict)

	# total_score = 0
	# for key in similarity_score_dict:
	# 	total_score += similarity_score_dict[key]
	#
	# for key in similarity_score_dict:
	# 	similarity_score_dict[key] = sigmoid_func(similarity_score_dict[key], total_score/2)
	return similarity_score_dict

def fastRELIC_similarity(word_id_shape, nx_graph, depth, threshold):
	similarity_score_dict = {dff: word_id_shape[dff] for dff in word_id_shape if "DFF" in dff}
	
	dff_list = list(similarity_score_dict.keys())
	# create counter of similarity scores
	counter = {}
	for x in dff_list:
		counter[x] = 0

	# create merged graph. 
	node_list=list(nx_graph.nodes())
	designGraph_merge=fastRELIC_v5.mergeGraph(nx_graph, node_list)


	done_keys = []
	special_cases = ["INVX1_42", "INVX1_44", "INVX1_46", "INVX1_48"]
	for key in dff_list:
		# similarity_score = []

		# only compare if not previously compared
		done_keys.append(key)
		smaller_dff_list = dff_list.copy()
		for x in done_keys:
			smaller_dff_list.remove(x)

		for key2 in smaller_dff_list:
			
			RELICcounter=0
			# work not with DFF, but with predecesssors which are not clock = prep1, prep2 
			prep1_list = [x for x in nx_graph.predecessors(key)]
			prep2_list = [x for x in nx_graph.predecessors(key2)]

			for clock1, clock2 in zip(prep1_list, prep2_list):
				# print(clock1.lower(), clock2.lower())
				if "c" in clock1.lower() and "l" in clock1.lower() and "k" in clock1.lower():
					prep1_list.remove(clock1)
				if "c" in clock2.lower() and "l" in clock2.lower() and "k" in clock2.lower():
					prep2_list.remove(clock2)
			# prep1_list.remove("INPUT_clock")
			# prep2_list.remove("INPUT_clock")
			
			if len(prep1_list) != 1 or len(prep2_list) != 1:
				for gate in special_cases:
					if gate in prep1_list:
						prep1_list.remove(gate)
					if gate in prep2_list:
						prep2_list.remove(gate)
			
			# check that only one predecessor left
			if len(prep1_list) != 1 or len(prep2_list) != 1:
				raise ValueError("Something is wrong when creating the prep1 and prep2 values, probably the clock is not called INPUT_clock")
				
			prep1 = prep1_list[0]
			prep2 = prep2_list[0]

		
			# calculate simscore
			SimScore, RELICcounter = fastRELIC_v5.HelperRELICstructure(prep1, prep2, designGraph_merge, depth, RELICcounter)
			# if simscore is above a threshold, count as "similar"
			if SimScore > threshold:
				counter[key]=counter[key]+1
				counter[key2]=counter[key2]+1

	pprint(counter)

	for score in counter:
		counter[score] = counter[score]/len(similarity_score_dict) 
	
	return counter

if __name__== "main":
    # Testing similarity score calculations
    similarity_score_dict = vector.euclidean_distance(shape_dict)
    # similarity_score_dict = vector.cosine_similarity(shape_dict)
    similarity_score_dict = pd.DataFrame.from_dict(similarity_score_dict, orient='index').T
    # print(similarity_score_dict)
