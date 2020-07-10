
import tueisec_myparsing.myparsing_ply as myparsing
import networkx as nx
import glob
import os
import os.path
import sys
from pprint import pprint
sys.path.insert(1, "./myModules")

import word_id__vector__cosine_similarity__euclidean_distance as vector
import matplotlib.pyplot as plt
import pydot
import pygraphviz as pgv
import word_id
import fastRELIC_v5
import numpy as np
import pandas as pd
from pprint import pprint

pd.set_option('display.max_colwidth', None)

def read_verilog(file_path, liberty_file_path):
	design_name = os.path.splitext(os.path.basename(file_path))[0]
	parsed_verilog = myparsing.parse_to_obj(file_path, liberty_file_path)
	graph = parsed_verilog.graph
	nets = parsed_verilog.nets
	nodes = parsed_verilog.nodes
	nets_to_nodes = parsed_verilog.nets_to_nodes
	graph = nx.DiGraph(graph)

	node_type = parsed_verilog.node_types

	PIs = list(nodes[0].keys())
	POs = list(nodes[2].keys())

	# add primary inputs to cellFanin Dictionary and to nodeTypes
	for PI in PIs:
		node_type[PI] = "INPUT"

	# add output/input to nodetypes
	for PO in POs:
		node_type[PO] = "OUTPUT"

	return graph, node_type, design_name


if __name__ == '__main__':
	# file_path = "designs/synthesize_altered_woB/"
	file_path = "designs/test/"
	liberty_file_path = "cell_library/osu035_stdcells.lib"
	networkx_graph = "preprocess_networkx_graph/"
	CSVDIR = 'csvs/'
	CSVDIR_MODIFIED = 'csvs/modified/'

	# ~ adjlist_output_folder = "output/adjlist/"
	# ~ csv_output_folder = "output/csv/"
	if not os.path.exists(CSVDIR_MODIFIED):
	    os.makedirs(CSVDIR_MODIFIED)
	if not os.path.exists(networkx_graph):
	    os.makedirs(networkx_graph)


	bad = []
	for design in glob.glob(file_path + "*.v"):
		print(design)
		design_V = design.split("/")[2]
		# read verilog and convert to network DiGraph
		graph, node_type, design_name = read_verilog(design, liberty_file_path)

		# do something with it
		# ~ nx.write_adjlist(graph, adjlist_output_folder + design_name + ".txt")

		# print(graph.edges())
		# print(graph.nodes())

		#creating of netwo object and saving dot file
		graph.add_nodes_from(graph.nodes())
		graph.add_edges_from(graph.edges())
		nx.drawing.nx_agraph.write_dot(graph, networkx_graph + "{}.dot".format(design_name) )

		#getting node type
		shape_dict = word_id.find_shapes(graph, node_type)
	

		# Testing similarity score calculations
		# similarity_score_dict = vector.cosine_similarity(shape_dict, 0.9)
		similarity_score_dict = vector.euclidean_distance(shape_dict, 2.4)
		# similarity_score_dict_2 = vector.fastRELIC_similarity(shape_dict, graph, 6, 0.7)

		similarity_score_dict = pd.DataFrame.from_dict(similarity_score_dict, orient='index').T
		# similarity_score_dict_2 = pd.DataFrame.from_dict(similarity_score_dict_2, orient='index').T
		# print(similarity_score_dict)

		# Appending calculations to csv
		# design_CSV = pd.read_csv("~/Thesis/thesis_neural_network/"+ CSVDIR + design_V.replace(".v", ".csv"))
		# feature_col = design_CSV.pop(design_CSV.columns[0])
		# design_CSV = design_CSV.reindex(sorted(design_CSV.columns, key=lambda number: int(number.split("_")[1])), axis=1) #sorting columns
		# for c in design_CSV.columns:		#renaming columns to match word_id dictionary name
		# 	new_c=c.split("_")[0:2]
		# 	new_c="_".join(new_c)
		# 	design_CSV.rename(columns={c : new_c}, inplace=True)

		# design_CSV.insert(0, "", feature_col)
		# design_CSV = design_CSV.append(similarity_score_dict,ignore_index=True)
		# design_CSV.iloc[-1, 0] = "Euclidean"
		# design_CSV = design_CSV.append(similarity_score_dict_2,ignore_index=True)
		# design_CSV.iloc[-1, 0] = "fastRELIC_score"
		# design_CSV.to_csv("~/Thesis/thesis_neural_network/"+ CSVDIR_MODIFIED + design_V.replace(".v", ".csv"), index=False)


		# plt.clf()
