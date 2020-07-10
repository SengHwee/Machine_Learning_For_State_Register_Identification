#!usr/bin/ipython3
## @package fastRELIC
#  Main source file for fastRELIC
#
# Voraussetzung: synthetisiertes Design, Knowledge about functionality of gates, Knowledge what are inputs and outputs, Knowledge about what are Registers and how are they defined and what are their input pins (clk, D, ...)

#parameter

import sys
scriptpath = "."
sys.path.append((scriptpath))
from collections import OrderedDict

import os
from pprint import pprint
from datetime import datetime
import networkx as nx
import ast
from ast import literal_eval
from collections import OrderedDict
import pydot
import time

import pdb #debugging

#for better Graph/Matrix Analysis: import NumPy

# import scripts
# import myparsing
import timing
import func
# import parsing

#global variables
MemSim=OrderedDict()
MemSim_inv=OrderedDict()
design=str()
global new_graph
global new_graph_INV
new_graph=nx.Graph()
new_graph_INV=nx.Graph()


## Preparation and Parsing
#
#  ...of the design as well as the external inputs and outputs.
def main_prep(designPath, libPath, design):

	####################################################################
	# Step 0: Preparation & Parsing
	####################################################################

	print("####################################################################")
	print("#######################  PREP & PARSING ############################")
	print("####################################################################")

	## start timing.
	start = timing.clock()
	# start timing for prep and parsing
	#~ startPars = timing.clock()

	# ~ t13=time.time() 	#debugging

	# set info dict to global empty dictionary for later info File writing
	func.set_info_dict()
	func.set_current_step("Prep")

	# print info on command line arguments
	func.print_info("Command Line Input: " + str(sys.argv[1:]))

	# read process variables from command line arguments
	#process_word_id, process_data_path, process_subcircuit, process_circuit_id, process_postP, process_cluster  = func.read_process(sys.argv)

	# set global func.design_name


	#~ # parse constants and set as global
	#~ constants_dictionary = parsing.read_constants("constants.txt")
	#~ func.set_constants(constants_dictionary)

	# parse own constants and set as global
	constants_dictionary , design, OutputPath = parsing.read_argruments()				#original read_constants("own_constants.txt")	#own_constants_R.txt is the generated file from TestRELICParm.sh and own_constants.txt the original one
	func.set_design_name(design)
	func.set_OutputPath(OutputPath)
	if not func.design_name:
		raise ValueError("No design selected")
	func.set_constants(constants_dictionary)

	# path for designInfo files
	#~ designInfoPath = "../designs//designInfo/"

	# create output dir for output files etc., if not already existant
	#~ func.create_paths()

	# specify designFile, and check whether designInfo exists
	designFile = func.design_name + ".v"


	# parsing to create designGraph, PIs, POs etc
	func.print_info("Prep: Parsing Design")

	# cell_lib, node_lookup, nets_to_nodes are global

	# if bench file, read as bench, not as verilog
	## @var nodes
	# A list of all nodes of the given design netlist.
	## @var cell_lib
	# A dictionary which contain most of the parts of the parsed cell library.
	try:
		graph, nets, nodes, nodeTypes, cell_lib, nets_to_nodes = myparsing.parse_it(designPath + designFile, libPath)
	except:
		designFile = func.design_name + ".bench"
		graph, nets, nodes, nodeTypes, cell_lib, nets_to_nodes = myparsing.parse_it_bench(designPath + designFile, libPath)

	# set nodetypes as global
	func.set_node_types(nodeTypes)

	func.print_info("Finished parsing Design")
	func.print_info("Design contains " + str(len(graph)) + " gates")

	PIs = list(nodes[0].keys())
	POs = list(nodes[2].keys())

	# convert to networkx DiGraph
	## @var designGraph
	# A networkx Graph build from the given design netlist.
	designGraph = nx.DiGraph(graph)

	## removing unconnected nodes and inputs /outputs
	func.print_info("removing unconnected nodes")
	PIs_reduced = []
	POs_reduced = []
	for node in PIs:
		if node in designGraph.nodes():
			PIs_reduced.append(node)
	for node in POs:
		if node in designGraph.nodes():
			POs_reduced.append(node)

	## @var PIs
	# All external inputs of the given design netlist.
	PIs = PIs_reduced
	## @var POs
	# All external outputs of the given design netlist.
	POs = POs_reduced


	for node in list(designGraph.nodes()):
		if designGraph.degree(node) == 0:
			designGraph.remove_node(node)
			if node in POs:
				POs.remove(node)
			if node in PIs:
				PIs.remove(node)

	func.print_info("finished removing unconnected nodes")

	# add edge data such as input pin and output pin
	designGraph = parsing.add_edge_data(designGraph, nodes)

	# find control wires (fixme: currently by hand
	func.set_control_wires([])

	# add primary inputs to cellFanin Dictionary and to nodeTypes
	for PI in PIs:
		nodeTypes[PI] = "INPUT"

	# add output/input to nodetypes
	for PO in POs:
		nodeTypes[PO] = "OUTPUT"

	func.dump_digraph(str(OutputPath)+'/gephi/depth' + str(func.constants['depth_const']) + "/T1_" + str(func.constants['T1']) + '_DP_' + str(func.constants['depth_const']) + '_' + design + '_iter', designGraph)
	func.draw_digraph(str(OutputPath)+'/gephi/depth' + str(func.constants['depth_const']) + "/T1_" + str(func.constants['T1']) + '_DP_' + str(func.constants['depth_const']) +'_iter' , designGraph)

	# ~ t14=time.time()		#debugging
	# ~ print('TIME main_prep',t14-t13)		#debugging

	return designGraph, PIs, POs, nodes, cell_lib

## Extract the registers out of design netlist.
#
#  This function is searching for all nodes which are starting with 'DF' and store them in node_OI.
#  Additionally, the corresponding clock name for each identified register is determined.
def main_defineReg(nodes, PIs):

	####################################################################
	#define registers out of netlist
	####################################################################
	# ~ t11=time.time()		#debugging
	NL=myparsing.node_lookup
	NN=myparsing.nets_to_nodes
	## @var node_OI
	# A list of all registers in the given design netlist.
	node_OI=[]
	## @var all_gate_clk
	# A dictionary which shows the corresponding clock name for each register in the design netlist.
	all_gate_clk=OrderedDict()

	#find all registers !!muss noch verallgemeinert werden!!!!
	for item in nodes[1]:
		#~ if (('DFFPOSX1_' in item) or ('DFF_X1_' in item) or ('DFFSR_' in item)):
		if item.startswith('DF'):
			node_OI.append(item)
			for pin in NL[item]['src']:
				if pin[0]=='CLK':
					wire_clk=pin[1]
					if 'INPUT_'+wire_clk in PIs:
						all_gate_clk[item]='INPUT_'+wire_clk
					else:
						all_gate_clk[item]=NN[wire_clk]

	# ~ t12=time.time() 		#debugging
	# ~ print('TIME main_defineREG', t12-t11)		#debugging

	return node_OI, all_gate_clk

## Replacing asynchronous registers with a synchronous register structure.
#
#  This function is based on the assumption that the algorithm fastRELIC does not care about about asynchronous or synchronous registers.
#  Therefore, all asynchronous registers (which are assumed to start with 'DFFSR_') are rebuild to a synchronous structure .
def main_mapR(designGraph, node_OI, PIs, POs, all_gate_clk):
	####################################################################
	#replace all DFFSR with DFFPOSX1 structure
	####################################################################
	# ~ t9=time.time()	#debugging
	NL=myparsing.node_lookup
	NN=myparsing.nets_to_nodes

	node_OI2=[]
	dffsr_dict={}
	#~ counter=OrderedDict()
	for item in node_OI:
		if item.startswith('DFFSR_'):
			pprint(item)
			#find gates to S,R,D,clock
			node_OI2.append('DFFPOSX1DFFSR_'+item[6:])
			all_gate_clk['DFFPOSX1DFFSR_'+item[6:]]=all_gate_clk[item]
			del all_gate_clk[item]
			#~ counter['DFFPOSX1DFFSR_'+item[6:]]=0
			dffsr_dict[item]='DFFPOSX1DFFSR_'+item[6:]
			for pin in NL[item]['src']:
				if pin[0]=='CLK':
					wire_clk=pin[1]
					if 'INPUT_'+wire_clk in PIs:
						gate_clk='INPUT_'+wire_clk
					else:
						gate_clk=NN[wire_clk]
					if gate_clk in list(dffsr_dict.keys()):
						gate_clk=dffsr_dict[gate_clk]
						all_gate_clk['DFFPOSX1DFFSR_'+item[6:]]=dffsr_dict[gate_clk]
				elif pin[0]=='S':
					wire_S=pin[1]
					if 'INPUT_'+wire_S in PIs:
						gate_S='INPUT_'+wire_S
					else:
						gate_S=NN[wire_S]
					if gate_S in list(dffsr_dict.keys()):
						gate_S=dffsr_dict[gate_S]
				elif pin[0]=='R':
					wire_R=pin[1]
					if 'INPUT_'+wire_R in PIs:
						gate_R='INPUT_'+wire_R
					else:
						gate_R=NN[wire_R]
					if gate_R in list(dffsr_dict.keys()):
						gate_R=dffsr_dict[gate_R]
				elif pin[0]=='D':
					wire_D=pin[1]
					if 'INPUT_'+wire_D in PIs:
						gate_D='INPUT_'+wire_D
					else:
						gate_D=NN[wire_D]
					if gate_D in list(dffsr_dict.keys()):
						gate_D=dffsr_dict[gate_D]
			if gate_clk.startswith('DFF') or gate_S.startswith('DFF') or gate_R.startswith('DFF') or gate_D.startswith('DFF'):
				pprint('Register nah zusammen an input')
			#make new connections
			designGraph.add_edge(gate_clk, 'DFFPOSX1DFFSR_'+item[6:])
			designGraph.add_edge(gate_S, 'ANDX1rep1_'+item[6:])
			designGraph.add_edge(gate_S, 'INVX1rep1_'+item[6:])
			designGraph.add_edge('ANDX1rep1_'+item[6:], 'ORX1rep1_'+item[6:])
			designGraph.add_edge('ORX1rep1_'+item[6:], 'DFFPOSX1DFFSR_'+item[6:])
			designGraph.add_edge('INVX1rep1_'+item[6:], 'ANDX1rep2_'+item[6:])
			designGraph.add_edge('ANDX1rep2_'+item[6:], 'ORX1rep1_'+item[6:])
			designGraph.add_edge(gate_R, 'INVX1rep2_'+item[6:])
			designGraph.add_edge('INVX1rep2_'+item[6:], 'ANDX1rep1_'+item[6:])
			designGraph.add_edge('INVX1rep2_'+item[6:], 'ANDX1rep3_'+item[6:])
			designGraph.add_edge('ANDX1rep3_'+item[6:], 'ANDX1rep2_'+item[6:])
			designGraph.add_edge(gate_D, 'ANDX1rep3_'+item[6:])
			for gate_Q in list(designGraph.successors(item)):
				if gate_Q.startswith('DFF'):
					pprint('Register nah zusammen an input')
				designGraph.add_edge('DFFPOSX1DFFSR_'+item[6:], gate_Q)
				designGraph.remove_edge(item, gate_Q)
			designGraph.remove_edge(gate_clk, item)
			designGraph.remove_edge(gate_S, item)
			designGraph.remove_edge(gate_R, item)
			designGraph.remove_edge(gate_D, item)
			designGraph.remove_node(item)
		else:
			node_OI2.append(item)
			#~ counter[item]=0

		# ~ t10=time.time()		#debugging
		# ~ print('TIME main_mapR',t10-t9)	#debugging

	return designGraph, node_OI2, all_gate_clk

## Sort registes according to their clock region.
#
#  This function uses the information about the corresponding clock of each register and sorts the registers accordingly.
def main_differentClocks(node_OI, designGraph, all_gate_clk):
	# ~ t6=time.time()	#debugging
	####################################################################
	#check and sort different clock based registers
	####################################################################

	## @var defined_clocks
	# A list of all possible clock regions in the design.
	defined_clocks=[]
	## @var node_OI_more
	# A dictionary assigns the correct registers to each available clock region.
	node_OI_more=OrderedDict()
	gated_clocks=[]
	for item in node_OI:
		if all_gate_clk[item] not in defined_clocks:
			defined_clocks.append(all_gate_clk[item])
			node_OI_more[all_gate_clk[item]]=[item]
		else:
			node_OI_more[all_gate_clk[item]].append(item)

	pprint(defined_clocks)
	# ~ t7=time.time()	#debugging
	# ~ print('TIME main differentClocks:', t7-t6)	#debugging
	return node_OI_more, defined_clocks, all_gate_clk

## Merge some nodes of the design graph
#
#  This function preprocesses the design graph. There are two possible scenarios where a node merging is performed.
#  The first one appears if two similar nodes/gates are located next to each other without an inverter inbetween.
#  The second one appears if two different nodes/gates are located next to each other with an inverter inbetween.
#  If one of this two cases exists, the two affected nodes are merged and all double inverter structures are removed.
#  The aim is to reach a graph structure which always looks the same for the same functionality.
def mergeGraph(designGraph_merge, node_list):
	#Preprocessing (merging input wires)
	#1.case: two similar gates without inverter inbetween
	#2.case: two different gates with inverter inbetween
	# ~ t4=time.time()		#debugging
	new_count=0
	while len(node_list)!=0:
		node_list_merk=list(node_list)
		for item in sorted(node_list):
			if item not in node_list_merk:
				pass
			else:
				if str(item)[0:3]=='AND':
					stay=0
					for suc1 in sorted(list(designGraph_merge.predecessors(item))):
						if str(suc1)[0:3]=='AND':
							stay=1
							designGraph_merge.remove_edge(suc1,item)
							for suc2 in sorted(list(designGraph_merge.predecessors(suc1))):
								designGraph_merge.add_edge(suc2,item)
							if len(list(designGraph_merge.successors(suc1)))==0:
								designGraph_merge.remove_node(suc1)
								if suc1 in node_list_merk:
									node_list_merk.remove(suc1)
						elif str(suc1)[0:3]=='INV':
							suc_inv=list(designGraph_merge.predecessors(suc1))[0]
							if str(suc_inv)[0:2]=='OR':
								stay=1
								designGraph_merge.remove_edge(suc1,item)
								if len(list(designGraph_merge.successors(suc1)))==0:
									designGraph_merge.remove_node(suc1)
								for suc2 in sorted(list(designGraph_merge.predecessors(suc_inv))):
									if str(suc2)[0:3]=='INV':
										designGraph_merge.add_edge(list(designGraph_merge.predecessors(suc2))[0],item)
									else:
										designGraph_merge.add_edge(suc2, suc1 + 'new' + str(new_count))
										designGraph_merge.add_edge(suc1 + 'new' + str(new_count), item)
										new_count=new_count+1
								if len(list(designGraph_merge.successors(suc_inv)))==0:
									for suc2 in sorted(list(designGraph_merge.predecessors(suc_inv))):
										designGraph_merge.remove_edge(suc2, suc_inv)
										if len(list(designGraph_merge.successors(suc2)))==0:
											designGraph_merge.remove_node(suc2)
									designGraph_merge.remove_node(suc_inv)
									if suc_inv in node_list_merk:
										node_list_merk.remove(suc_inv)
					if stay==0:
						node_list_merk.remove(item)
				elif str(item)[0:2]=='OR':
					stay=0
					for suc1 in sorted(list(designGraph_merge.predecessors(item))):
						if str(suc1)[0:2]=='OR':
							stay=1
							designGraph_merge.remove_edge(suc1,item)
							for suc2 in sorted(list(designGraph_merge.predecessors(suc1))):
								designGraph_merge.add_edge(suc2,item)
							if len(list(designGraph_merge.successors(suc1)))==0:
								designGraph_merge.remove_node(suc1)
								if suc1 in node_list_merk:
									node_list_merk.remove(suc1)
						elif str(suc1)[0:3]=='INV':
							suc_inv=list(designGraph_merge.predecessors(suc1))[0]
							if str(suc_inv)[0:3]=='AND':
								stay=1
								designGraph_merge.remove_edge(suc1,item)
								if len(list(designGraph_merge.successors(suc1)))==0:
									designGraph_merge.remove_node(suc1)
								for suc2 in sorted(list(designGraph_merge.predecessors(suc_inv))):
									if str(suc2)[0:3]=='INV':
										designGraph_merge.add_edge(list(designGraph_merge.predecessors(suc2))[0],item)
									else:
										designGraph_merge.add_edge(suc2, suc1 + 'new' + str(new_count))
										designGraph_merge.add_edge(suc1 + 'new' + str(new_count), item)
										new_count=new_count+1
								if len(list(designGraph_merge.successors(suc_inv)))==0:
									for suc2 in sorted(list(designGraph_merge.predecessors(suc_inv))):
										designGraph_merge.remove_edge(suc2, suc_inv)
										if len(list(designGraph_merge.successors(suc2)))==0:
											designGraph_merge.remove_node(suc2)
									designGraph_merge.remove_node(suc_inv)
									if suc_inv in node_list_merk:
										node_list_merk.remove(suc_inv)
					if stay==0:
						node_list_merk.remove(item)
				else:
					node_list_merk.remove(item)
		node_list=list(node_list_merk)
		# ~ t5= time.time() 	#debugging
		# ~ print('TIME Merge_Graph', t5-t4)	#debugging
	return designGraph_merge

## Calculating a Similarity Score for two given nodes/gates
#
#  This is the code which is introduced in the paper RELIC. You recoursively check each possible children pair of two nodes/gates.
#  Return zero, if the gate type differes and a one if both gate types are an external input. Otherwise, continue the depth search until a maximum depth is reached. Then, return a normalized value.
#  If the similarity score for a child pair is higher than a special threshold 'T1', add an edge to a bipartite graph whose both sets consist of the children nodes.
#  The final similarity score for the starting node/gate pair is the maximum matching of the final bipartite graph.


def main_RELICformulti(design, designGraph, node_OI, counter, designGraph_merge, designGraph_merge_inverse, all_gate_clk):
		
	##############################################################################################################################
	#Compute Similarity Score between Registers
	#Use algorithm of Paper "Gate-Level Netlist Reverse Engineering for Hardware Security: Control Logic Register Identification"
	##############################################################################################################################
			
	MemSim=OrderedDict()
	MemSim_inv=OrderedDict()
	depth=func.constants['depth_const'] #define depth,when algorithm should stop
	
	#~ el1=0
	#~ el2=0
	#~ el3=0
	#~ el4=0
	#~ count=0
	#~ used=0
	
	starttest=time.time()
	for i in range(len(node_OI)):
		
		if ((list(designGraph_merge.predecessors(sorted(node_OI)[i]))[0] == all_gate_clk[sorted(node_OI)[i]]) and (len(list(designGraph_merge.predecessors(sorted(node_OI)[i])))==2)):
			prep1=list(designGraph_merge.predecessors(sorted(node_OI)[i]))[1]
		elif ((list(designGraph_merge.predecessors(sorted(node_OI)[i]))[1] == all_gate_clk[sorted(node_OI)[i]]) and (len(list(designGraph_merge.predecessors(sorted(node_OI)[i])))==2)):
			prep1=list(designGraph_merge.predecessors(sorted(node_OI)[i]))[0]
		else:
			pprint('Achtung, clock gate wird nicht mehr gefunden wegen merging?')
			pprint(list(designGraph_merge.predecessors(sorted(node_OI)[i])))
			pprint(all_gate_clk[sorted(node_OI)[i]])
			input()
		#comment if ignoring inverse
		if ((list(designGraph_merge_inverse.predecessors(sorted(node_OI)[i]))[0] == all_gate_clk[sorted(node_OI)[i]]) and (len(list(designGraph_merge_inverse.predecessors(sorted(node_OI)[i])))==2)):
			prep1_inv=list(designGraph_merge_inverse.predecessors(sorted(node_OI)[i]))[1]
		elif ((list(designGraph_merge_inverse.predecessors(sorted(node_OI)[i]))[1] == all_gate_clk[sorted(node_OI)[i]]) and (len(list(designGraph_merge_inverse.predecessors(sorted(node_OI)[i])))==2)):	
			prep1_inv=list(designGraph_merge_inverse.predecessors(sorted(node_OI)[i]))[0]
		else:
			pprint('Achtung, clock gate wird nicht mehr gefunden wegen merging?')
			pprint(list(designGraph_merge_inverse.predecessors(sorted(node_OI)[i])))
			pprint(all_gate_clk[sorted(node_OI)[i]])
			input()			
			
		for j in range(len(node_OI)-i-1):		
			
			#always compare similarity and inverse similarity and use highest one

			if ((list(designGraph_merge.predecessors(sorted(node_OI)[j+i+1]))[0] == all_gate_clk[sorted(node_OI)[j+i+1]]) and (len(list(designGraph_merge.predecessors(sorted(node_OI)[j+i+1])))==2)):
				prep2=list(designGraph_merge.predecessors(sorted(node_OI)[j+i+1]))[1]
			elif ((list(designGraph_merge.predecessors(sorted(node_OI)[j+i+1]))[1] == all_gate_clk[sorted(node_OI)[j+i+1]]) and (len(list(designGraph_merge.predecessors(sorted(node_OI)[j+i+1])))==2)):	
				prep2=list(designGraph_merge.predecessors(sorted(node_OI)[j+i+1]))[0]
			else:
				pprint('Achtung, clock gate wird nicht mehr gefunden wegen merging?')
				pprint(list(designGraph_merge.predecessors(sorted(node_OI)[j+i+1])))
				pprint(all_gate_clk[sorted(node_OI)[j+i+1]])
				input()			
			
			if str(prep1)[0:3]!='INV':
				regist1input=prep1
				regist1input_inverse=prep1_inv #comment if ignoring inverse
			else:
				regist1input=list(designGraph_merge.predecessors(prep1))[0]
				regist1input_inverse=list(designGraph_merge_inverse.predecessors(prep1_inv))[0]	#comment if ignoring inverse	
			if str(prep2)[0:3]!='INV':
				regist2input=prep2
			else:
				regist2input=list(designGraph_merge.predecessors(prep2))[0]	
				
			#calculate normal similarity score
			SimScore1, MemSim = GetSimilarityScore_woInfo(designGraph_merge, designGraph_merge, regist1input,regist2input, depth, MemSim)

			#use inversed subgraph & calculate constructed inverse similarity score: comment if ignoring inverse
			SimScore2, MemSim_inv = GetSimilarityScore_woInfo(designGraph_merge_inverse, designGraph_merge, regist1input_inverse, regist2input, depth, MemSim_inv)

			#choose highest score
			SimScore=max(SimScore1, SimScore2)
			#~ if SimScore2>SimScore1:
				#~ used=used+1
			#~ SimScore=SimScore1 #choose if ignoring inverse

		
			if SimScore >func.constants['T2']: #this is again a threshold, which is variable
				counter[sorted(node_OI)[i]]=counter[sorted(node_OI)[i]]+1
				counter[sorted(node_OI)[j+i+1]]=counter[sorted(node_OI)[j+i+1]]+1

			#~ count=count+1
	
	#~ elapsedtest=time.time()-starttest
	#~ pprint('whole for loops')
	#~ pprint(elapsedtest)
			
	#Register classification
	regist_OI=[]		
	count=0
	count2=0
	designGraph_prune=designGraph.copy()
	for item in sorted(node_OI):
		if counter[item]<=func.constants['T3']: #again a threshold, paper suggests 0
			regist_OI.append(item)
			
			#check if identified state registers have feedback path: comment if FPs are ignored
			fp, designGraph_temp, designGraph_prune, count, count2, one_cycle_found=find_feedbackpath(designGraph, designGraph_prune, item, sorted(node_OI), design, count, count2)
			if one_cycle_found==0:
				pprint('RELIC found a register which has no feedback path')
				pprint(item)
				regist_OI.remove(item)
	
	
	
	return regist_OI, counter

def GetChildList_seperat(merged_netlist_graph, merged_netlist_graph2, gate1,gate2,depth):
	'''
	Orders the Prededessors of two gates in a 2D Array/List depending on their depth.
	Also there are no duplicates in the same Depth.

	'''

	ChildList_seperat1 = [[] for i in range(depth)]
	ChildList_seperat2 = [[] for i in range(depth)]
	ChildList_seperat1[0]=[gate1]
	ChildList_seperat2[0]=[gate2]
	for depth_it in range(1,depth):
		helper_listy1=list()
		helper_listy2=list()
		#~ print(depth)
		for node in ChildList_seperat1[depth_it-1]:
			#~ print(node)
			helper_listy1.extend(list(merged_netlist_graph.predecessors(node)))
		ChildList_seperat1[depth_it].extend(list(OrderedDict.fromkeys(helper_listy1)))
		for node2 in ChildList_seperat2[depth_it-1]:
			#~ print(node)
			helper_listy2.extend(list(merged_netlist_graph2.predecessors(node2)))
		ChildList_seperat2[depth_it].extend(list(OrderedDict.fromkeys(helper_listy2)))

		#~ print(ChildList_seperat1[depth_it])
		#~ print(ChildList_seperat2[depth_it])
	return ChildList_seperat1, ChildList_seperat2

def GetSimilarityScore_woInfo(graph1, graph2, gate1, gate2, d, MemSim):
	#Compute Similarity Score between Registers
	#Use algorithm of Paper "Gate-Level Netlist Reverse Engineering for Hardware Security: Control Logic Register Identification"

	maxvalue=max(len(list(graph1.predecessors(gate1))), len(list(graph2.predecessors(gate2))))
	minvalue=min(len(list(graph1.predecessors(gate1))), len(list(graph2.predecessors(gate2))))
	
	if str(gate1)[0:5]=='INPUT' or str(gate2)[0:5]=='INPUT':
		if str(gate1)[0:5]=='INPUT' and str(gate2)[0:5]=='INPUT':
			return 1, MemSim #mache ich so, weiß nicht, ob richtig
		else:
			return 0, MemSim
	elif str(gate1)[0:2] == str(gate2)[0:2]:
		pass
	else:
		return 0, MemSim
	
	if d==0:
		output=minvalue/maxvalue
		return output, MemSim
	
	B = nx.Graph()
	B.add_nodes_from(list(graph1.predecessors(gate1)), bipartite=0)
	additionalN=[]

	for checkpre in list(graph2.predecessors(gate2)):
		if checkpre in list(B.nodes()):
			additionalN.append(checkpre + 'new')
		else:
			additionalN.append(checkpre)	
	B.add_nodes_from(additionalN, bipartite=1)
	
	for a in sorted(list(graph1.predecessors(gate1))):
		for b in sorted(list(graph2.predecessors(gate2))):
			#switch between dynamic and non-dynamic programming -> changes the results due to the different depth
			
			#~ if ((a,b,d) in list(MemSim.keys())):
				#~ SimScore=MemSim[(a,b,d)]
			#~ elif ((b,a,d) in list(MemSim.keys())):
				#~ SimScore=MemSim[(b,a,d)]			
			#~ else:	
				#~ SimScore, MemSim = GetSimilarityScore_woInfo(graph1, graph2, a,b,d-1, MemSim)
				#~ MemSim[(a,b,d)]=SimScore

			SimScore, MemSim = GetSimilarityScore_woInfo(graph1,graph2,a,b,d-1, MemSim)
			

			# ~ if SimScore >= func.constants['T1']: # 0 is here the threshold, which is variable
			if SimScore >= 0.5: # 0 is here the threshold, which is variable
				if b in list(graph1.predecessors(gate1)):
					B.add_edge(a, b + 'new')
				else:	
					B.add_edge(a,b)	
	
	if len(list(B.edges()))==0:
		maxmatching=0
	else:
		maxmatching=0	
		top_nodes = set(n for n,d in B.nodes(data=True) if d['bipartite']==0)
		bottom_nodes = set(B) - top_nodes
		# ~ for Bgroup in list(nx.connected_component_subgraphs(B, copy=True)):	
		for Bgroup in list(connected_component_subgraphs(B)):		
			#changed due to networkx version 2.1 instead of 2.0
			#maxmatching_new=len(nx.nx.max_weight_matching(Bgroup))/2
			maxmatching_new=len(nx.nx.max_weight_matching(Bgroup))
			maxmatching=maxmatching+maxmatching_new
			
	output2=maxmatching/maxvalue
	
	if output2>1:
		pprint('Caution: SimScore result above 1')
		input()

	return output2, MemSim

def connected_component_subgraphs(G):
    for c in nx.connected_components(G):
        yield G.subgraph(c)

def GetSimScore_seperat(merged_netlist_graph , merged_netlist_graph2  ,gate1 , gate2 , depth):
	'''
	Calculate the SimularityScores of two gates, depending on their depth and their childs
	Iterativ version of GetSimilarityScore_woInfo from original fastRELIC by Michaela Brunner

	'''
	B = nx.Graph()
	# ~ t5=time.time()
	ChildList_seperat1, ChildList_seperat2 = GetChildList_seperat(merged_netlist_graph , merged_netlist_graph2 ,gate1,gate2,depth)

	#Reverse iteration of Depth
	for depth_now in range(depth-1,-1,-1):
		#~ print(depth_now)		#debugging
		#~ print(ChildList_seperat1[depth_now]) #debugging
		#~ print(ChildList_seperat2[depth_now])	#debugging

		#Adds all gates as nodes in one Graph
		new_graph.add_nodes_from(ChildList_seperat1[depth_now])
		new_graph.add_nodes_from(ChildList_seperat2[depth_now])

		#Gates of one Tree get compared to gates of the same depth from the other Tree
		for node1 in ChildList_seperat1[depth_now]:
			for node2 in ChildList_seperat2[depth_now]:
				#~ temp=0
				#~ print(node1,node2)	#debugging
				maxvalue=max(len(list(merged_netlist_graph.predecessors(node1))), len(list(merged_netlist_graph2.predecessors(node2))))

				#Case analysis for different Inputs
				if str(node1)[0:5]=='INPUT' or str(node2)[0:5]=='INPUT':
					if str(node1)[0:5]=='INPUT' and str(node2)[0:5]=='INPUT':	#Könnte man evtl. wegrazionalisieren
						output= 1 #macht Michaela so, sie weiß nicht, ob es richtig ist. Trotzdem wurde es von mir übernommen
					else:
						output= 0
				elif str(node1)[0:2] == str(node2)[0:2]:


					if (depth-depth_now)==1: #Case depth==0
						minvalue=min(len(list(merged_netlist_graph.predecessors(node1))), len(list(merged_netlist_graph2.predecessors(node2))))
						output=minvalue/maxvalue
						

					elif new_graph.has_edge(node1,node2) and depth_now in new_graph.edges[node1,node2] :
						#~ print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
						#~ print(depth_now in new_graph.edges[node1,node2])
						#~ print(depth_now)
						#~ print('Läuft rein')
						#~ print('Output:', output)
						#~ print(node1,node2)
						#~ print(gate1,gate2)
						#~ print(new_graph.edges[node1,node2])
						#~ temp=1
						continue


					else: #Case depth!=0
						#~ B = nx.Graph()
						B.add_nodes_from(list(merged_netlist_graph.predecessors(node1)), bipartite=0)
						additionalN=[]

						for checkpre in list(merged_netlist_graph2.predecessors(node2)):
							if checkpre in list(B.nodes()):
								additionalN.append(checkpre + 'new') #necessary cause of bipartite? Better solution Possible?
							else:
								additionalN.append(checkpre)
						B.add_nodes_from(additionalN, bipartite=1)

						###########

						#optimal Solution would be, to get this Step undependent of the Threshold
						for node_help1 in merged_netlist_graph.predecessors(node1):	# brings higher complexity => O^4
							for	node_help2 in merged_netlist_graph2.predecessors(node2): # brings higher complexity => O^5
								if new_graph.edges[node_help1,node_help2][depth_now+1] >= 0.5: # 0 is here the threshold, which is variable
									if node_help2 in list(merged_netlist_graph.predecessors(node1)):
										B.add_edge(node_help1, node_help2 + 'new')
									else:
										B.add_edge(node_help1,node_help2)


						###########

						if len(list(B.edges()))==0:
							maxmatching=0
						else:

							#Maxmatching
							maxmatching=0
							top_nodes = set(n for n,d in B.nodes(data=True) if d['bipartite']==0)
							bottom_nodes = set(B) - top_nodes
							for Bgroup in [B.subgraph(c).copy() for c in nx.connected_components(B)]:
								#changed due to networkx version 2.1 instead of 2.0
								#maxmatching_new=len(nx.nx.max_weight_matching(Bgroup))/2
								maxmatching_new=len(nx.nx.max_weight_matching(Bgroup))
								maxmatching=maxmatching+maxmatching_new

						output=maxmatching/maxvalue

						B.clear()
						#~ print('B_Knoten',B.nodes())


						if output>1:
							print('Caution: SimScore result above 1')
							input()



				else:
					output=0
			
				#~ print(output)
				#~ if temp==1:
					#~ pdb.set_trace()
				new_graph.add_edge(node1, node2)					#Adds edges between two Gates
				new_graph.edges[node1,node2][depth_now] = output	#konnte es nicht ihn einem Befehl verpacken

	#~ print('EDGE_DATA',graph3.edges.data())	#debugging
	# ~ print(new_graph.edges[gate1,gate2][0])	#debugging
	# ~ t6=time.time()							#debugging
	# ~ print('TIME_iterativ__seperat',t6-t5)	#debugging

	
	return new_graph.edges[gate1,gate2][0]

def GetSimScore_seperat_INV(merged_netlist_graph , merged_netlist_graph2  ,gate1 , gate2 , depth):
	'''
	Calculate the SimularityScores of two gates, depending on their depth and their childs
	Iterativ version of GetSimilarityScore_woInfo from original fastRELIC by Michaela Brunner

	'''
	B = nx.Graph()

	# ~ t5=time.time()
	ChildList_seperat1, ChildList_seperat2 = GetChildList_seperat(merged_netlist_graph , merged_netlist_graph2 ,gate1,gate2,depth)

	#Reverse iteration of Depth
	for depth_now in range(depth-1,-1,-1):
		#~ print(depth_now)		#debugging
		#~ print(ChildList_seperat1[depth_now]) #debugging
		#~ print(ChildList_seperat2[depth_now])	#debugging

		#Adds all gates as nodes in one Graph
		new_graph_INV.add_nodes_from(ChildList_seperat1[depth_now])
		new_graph_INV.add_nodes_from(ChildList_seperat2[depth_now])

		#Gates of one Tree get compared to gates of the same depth from the other Tree
		for node1 in ChildList_seperat1[depth_now]:
			for node2 in ChildList_seperat2[depth_now]:
				#~ temp1=0
				#~ print(node1,node2)	#debugging
				maxvalue=max(len(list(merged_netlist_graph.predecessors(node1))), len(list(merged_netlist_graph2.predecessors(node2))))

				#Case analysis for different Inputs
				if str(node1)[0:5]=='INPUT' or str(node2)[0:5]=='INPUT':
					if str(node1)[0:5]=='INPUT' and str(node2)[0:5]=='INPUT':	#Könnte man evtl. wegrazionalisieren
						output_INV= 1 #macht Michaela so, sie weiß nicht, ob es richtig ist. Trotzdem wurde es von mir übernommen
						#~ print('2*INPUT')
					else:
						output_INV= 0
						#~ print('Ein INPUT')
				elif str(node1)[0:2] == str(node2)[0:2]:


					if (depth-depth_now)==1: #Case depth==0
						minvalue=min(len(list(merged_netlist_graph.predecessors(node1))), len(list(merged_netlist_graph2.predecessors(node2))))
						output_INV=minvalue/maxvalue
						#~ print('Depth=0')


					elif new_graph_INV.has_edge(node1,node2) and depth_now in new_graph_INV.edges[node1,node2] :
						#~ print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
						#~ print(depth_now in new_graph.edges[node1,node2])
						#~ print(depth_now)
						#~ print('Läuft rein')
						#~ print('Output:', output)
						#~ print(node1,node2)
						#~ print(gate1,gate2)
						#~ print(new_graph.edges[node1,node2])
						#~ temp1=1
						continue


					else: #Case depth!=0
						#~ B = nx.Graph()
						B.add_nodes_from(list(merged_netlist_graph.predecessors(node1)), bipartite=0)
						additionalN=[]

						for checkpre in list(merged_netlist_graph2.predecessors(node2)):
							if checkpre in list(B.nodes()):
								additionalN.append(checkpre + 'new') #necessary cause of bipartite? Better solution Possible?
							else:
								additionalN.append(checkpre)
						B.add_nodes_from(additionalN, bipartite=1)

						###########

						#optimal Solution would be, to get this Step undependent of the Threshold
						for node_help1 in merged_netlist_graph.predecessors(node1):	# brings higher complexity => O^4
							for	node_help2 in merged_netlist_graph2.predecessors(node2): # brings higher complexity => O^5
								if new_graph_INV.edges[node_help1,node_help2][depth_now+1] >= func.constants['T1']: # 0 is here the threshold, which is variable
									if node_help2 in list(merged_netlist_graph.predecessors(node1)):
										B.add_edge(node_help1, node_help2 + 'new')
									else:
										B.add_edge(node_help1,node_help2)


						###########

						if len(list(B.edges()))==0:
							maxmatching=0
						else:

							#Maxmatching
							maxmatching=0
							top_nodes = set(n for n,d in B.nodes(data=True) if d['bipartite']==0)
							bottom_nodes = set(B) - top_nodes
							for Bgroup in [B.subgraph(c).copy() for c in nx.connected_components(B)]:
								#changed due to networkx version 2.1 instead of 2.0
								#maxmatching_new=len(nx.nx.max_weight_matching(Bgroup))/2
								maxmatching_new=len(nx.nx.max_weight_matching(Bgroup))
								maxmatching=maxmatching+maxmatching_new


						B.clear()
						#~ print('B_Knoten_INV',B.nodes())

						output_INV=maxmatching/maxvalue


						if output_INV>1:
							print('Caution: SimScore result above 1')
							input()



				else:
					output_INV=0

				#~ print(output_INV)
				#~ if temp1==1:
					#~ pdb.set_trace()
				new_graph_INV.add_edge(node1, node2)					#Adds edges between two Gates
				new_graph_INV.edges[node1,node2][depth_now] = output_INV	#konnte es nicht ihn einem Befehl verpacken

	#~ print('EDGE_DATA',graph3.edges.data())	#debugging
	# ~ print(new_graph_INV.edges[gate1,gate2][0])	#debugging
	# ~ t6=time.time()							#debugging
	# ~ print('TIME_iterativ__seperat',t6-t5)	#debugging
	return new_graph_INV.edges[gate1,gate2][0]

## Check if a register has a feedback path.
#
#  This function checks for each potential state registers if it has a feedback path. It systematically tries to find cycles with the aid of nx.find_cycle and checks if the found cycle starts and ends at my given start node.
#  If this is not the case, the found cycle is temporally summarized to one node and the process is reapeated until a valid cycle or no other cycle can be found.
def find_feedbackpath(designGraph, designGraph_prune, node_OIi, node_OI, design, count, count2):
	# ~ t15=time.time()		#debugging
	#this function stops if a cycle is found, if this is not the cycle starting from source, the nodes in the front are deleted
	# gibt immer den ersten gefundenen cycle aus
	designGraph_temp=designGraph_prune.copy()
	found_circle_with_register=0
	one_cycle_found=0

	while found_circle_with_register==0:
		count3=1
		count4=1
		node_in_fp=[]
		fp=[]
		try:
			fp=nx.find_cycle(designGraph_temp, source=node_OIi, orientation='original')
			for item in fp:
				node_in_fp.append(item[0])

			for item in fp:
				if node_OIi in item:
					one_cycle_found=1

					#designGraph_temp=pruneGraph(fp, node_in_fp, designGraph_temp, node_OIi)
					found_circle_with_register=1
			else:
				for item in node_in_fp:
					if (item in node_OI) or ('cycle_w' in item):
						#if already_found==0:
						designGraph_temp.add_node('cycle_w_reg'+str(count))
						for n1 in node_in_fp:
							for n2 in designGraph_temp.successors(n1):
								if (n2 not in node_in_fp):
									designGraph_temp.add_edge('cycle_w_reg'+str(count), n2)
							for n2 in designGraph_temp.predecessors(n1):
								if (n2 not in node_in_fp):
									designGraph_temp.add_edge(n2, 'cycle_w_reg'+str(count))
						for item2 in node_in_fp:
							designGraph_temp.remove_node(item2)
						count=count+1
						already_found=1
						break
						#else:
							#count=count-1
							#designGraph_temp.node[circle_w_reg_count]['reg' +str(count3)]=item
							#count=count+1
							#count3=count3+1
							#pprint('again other register in circle')
				else:
					designGraph_temp.add_node('cycle_o_reg' +str(count2))
					designGraph_prune.add_node('cycle_o_reg' +str(count2))
					for n1 in node_in_fp:
						for n2 in designGraph_temp.successors(n1):
							if (n2 not in node_in_fp):
								designGraph_temp.add_edge('cycle_o_reg'+str(count2), n2)
								designGraph_prune.add_edge('cycle_o_reg'+str(count2), n2)
						for n2 in designGraph_temp.predecessors(n1):
							if (n2 not in node_in_fp):
								designGraph_temp.add_edge(n2, 'cycle_o_reg'+str(count2))
								designGraph_prune.add_edge(n2, 'cycle_o_reg'+str(count2))
					for item2 in node_in_fp:
						designGraph_temp.remove_node(item2)
						designGraph_prune.remove_node(item2)
					count2=count2+1
					pprint('only combinatorial logic in found circle')


		except:
			print('no cycle found', sys.exc_info()[0])
			break

	# ~ t16=time.time()		#debugging
	# ~ print('TIME find_feedbackpath',t16-t15)
	return fp, designGraph_prune, count, count2, one_cycle_found

## Grouping registers
#
#  This function groups the registers by calculating the required Pair Similarity Scores based on the register grouping algorithm.
def group_register(node_OI_without_start, designGraph_merge, designGraph_merge_inverse, depth, start_node, T2, all_gate_clk):
	#calculate only needed SimScores to decide what are state and what are non-state registers
	# ~ t0=time.time()		#debugging
	RELICcounter=0
	#~ el2=0
	#~ el3=0
	#~ used=0

	#first calculate PairSim scores for all registers to start node
	start_similarities=OrderedDict()
	register_groups=[]
	single_register=[]
	#~MemSim=OrderedDict()
	#~MemSim_inv=OrderedDict()

	#changed hier prep2->prep1 and vice versa
	if ((list(designGraph_merge_inverse.predecessors(start_node))[0] == all_gate_clk[start_node]) and (len(list(designGraph_merge_inverse.predecessors(start_node)))==2)):
		prep1_inv=list(designGraph_merge_inverse.predecessors(start_node))[1]
	elif ((list(designGraph_merge_inverse.predecessors(start_node))[1] == all_gate_clk[start_node]) and (len(list(designGraph_merge_inverse.predecessors(start_node)))==2)):
		prep1_inv=list(designGraph_merge_inverse.predecessors(start_node))[0]
	else:
		pprint('Achtung, clock gate wird nicht mehr gefunden wegen merging?')
		pprint(start_node)
		pprint(list(designGraph_merge_inverse.predecessors(start_node)))
		pprint(all_gate_clk[start_node])
		input()
	if ((list(designGraph_merge.predecessors(start_node))[0] == all_gate_clk[start_node]) and (len(list(designGraph_merge.predecessors(start_node)))==2)):
		prep1=list(designGraph_merge.predecessors(start_node))[1]
	elif ((list(designGraph_merge.predecessors(start_node))[1] == all_gate_clk[start_node]) and (len(list(designGraph_merge.predecessors(start_node)))==2)):
		prep1=list(designGraph_merge.predecessors(start_node))[0]
	else:
		pprint('Achtung, clock gate wird nicht mehr gefunden wegen merging?')
		pprint(start_node)
		pprint(list(designGraph_merge.predecessors(start_node)))
		pprint(all_gate_clk[start_node])
		input()

	for i in range(0, len(node_OI_without_start)):
		if ((list(designGraph_merge.predecessors(sorted(node_OI_without_start)[i]))[0] == all_gate_clk[sorted(node_OI_without_start)[i]]) and (len(list(designGraph_merge.predecessors(sorted(node_OI_without_start)[i])))==2)):
			prep2=list(designGraph_merge.predecessors(sorted(node_OI_without_start)[i]))[1]
		elif ((list(designGraph_merge.predecessors(sorted(node_OI_without_start)[i]))[1] == all_gate_clk[sorted(node_OI_without_start)[i]]) and (len(list(designGraph_merge.predecessors(sorted(node_OI_without_start)[i])))==2)):
			prep2=list(designGraph_merge.predecessors(sorted(node_OI_without_start)[i]))[0]
		else:
			pprint('Achtung, clock gate wird nicht mehr gefunden wegen merging?')
			pprint(sorted(node_OI_without_start)[i])
			pprint(list(designGraph_merge.predecessors(sorted(node_OI_without_start)[i])))
			pprint(all_gate_clk[sorted(node_OI_without_start)[i]])
			input()


		#~ SimScore, RELICcounter, MemSim, MemSim_inv, el2, el3, used=HelperRELICstructure(prep1, prep2, prep1_inv, designGraph_merge, designGraph_merge_inverse, depth, RELICcounter, MemSim, MemSim_inv, el2, el3, used)
		SimScore, RELICcounter = HelperRELICstructure(prep1, prep2, prep1_inv, designGraph_merge, designGraph_merge_inverse, depth, RELICcounter)


		start_similarities[sorted(node_OI_without_start)[i]]=SimScore

	#~ start_similarities=OrderedDict(sorted(start_similarities.items(), key=lambda item: item[1]))

	#Umsortieren von StartSimList
	start_similarities=OrderedDict(sorted(start_similarities.items(), key=lambda item: item[1]))
	start_similarities_rev=OrderedDict(sorted(start_similarities.items(), key=lambda item: item[1], reverse=True))
	index=0
	for el in list(start_similarities_rev.items()):
		if el[1]<=T2:
			break
		else:
			index=index+1

	if index==0:
		pprint('No StartSimList elements with score > T2')
		start_similarities=OrderedDict(sorted(start_similarities.items(), key=lambda item: item[1], reverse=True))
	elif index==len(list(start_similarities_rev.items())):
		pprint('All StartSimList elements with score > T2')
	else:
		pprint('Mix of StartSimList elements with and without score > T2')
		if ((index==1) and (index!=len(list(start_similarities_rev.items()))-1)):
			firstpart=sorted(list(start_similarities.items())[:-index], key=lambda item: item[1], reverse=True)
			secondpart=list(start_similarities.items())[-index:]
		elif ((index==len(list(start_similarities_rev.items()))-1) and (index!=1)):
			firstpart=list(start_similarities.items())[:-index]
			secondpart=list(start_similarities.items())[-index:]
		elif ((index==1) and (index==len(list(start_similarities_rev.items()))-1)):
			firstpart=list(start_similarities.items())[:-index]
			secondpart=list(start_similarities.items())[-index:]
		else:
			firstpart=sorted(list(start_similarities.items())[:-index], key=lambda item: item[1], reverse=True)
			secondpart=list(start_similarities.items())[-index:]
		pprint(index)
		pprint(firstpart)
		pprint(secondpart)
		start_similarities=OrderedDict(firstpart+secondpart)
		#verify
		if len(list(start_similarities.items()))!=len(list(start_similarities_rev.items())):
			pprint('Fehler bei Sortierung')
			input()

	#group them

	while list(start_similarities.items())[0][1] <= T2:

		memory=OrderedDict()
		found_group=0
		set_register=0
		farest_register=list(start_similarities.items())[0][0]

		if ((list(designGraph_merge.predecessors(farest_register))[0] == all_gate_clk[farest_register]) and (len(list(designGraph_merge.predecessors(farest_register)))==2)):
			prep1=list(designGraph_merge.predecessors(farest_register))[1]
		elif ((list(designGraph_merge.predecessors(farest_register))[1] == all_gate_clk[farest_register]) and (len(list(designGraph_merge.predecessors(farest_register)))==2)):
			prep1=list(designGraph_merge.predecessors(farest_register))[0]
		else:
			pprint('Achtung, clock gate wird nicht mehr gefunden wegen merging?')
			pprint(farest_register)
			pprint(list(designGraph_merge.predecessors(farest_register)))
			pprint(all_gate_clk[farest_register])
			input()
		#comment if ignoring inverse
		if ((list(designGraph_merge_inverse.predecessors(farest_register))[0] == all_gate_clk[farest_register]) and (len(list(designGraph_merge_inverse.predecessors(farest_register)))==2)):
			prep1_inv=list(designGraph_merge_inverse.predecessors(farest_register))[1]
		elif ((list(designGraph_merge_inverse.predecessors(farest_register))[1] == all_gate_clk[farest_register]) and (len(list(designGraph_merge_inverse.predecessors(farest_register)))==2)):
			prep1_inv=list(designGraph_merge_inverse.predecessors(farest_register))[0]
		else:
			pprint('Achtung, clock gate wird nicht mehr gefunden wegen merging?')
			pprint(farest_register)
			pprint(list(designGraph_merge_inverse.predecessors(farest_register)))
			pprint(all_gate_clk[farest_register])
			input()

		for group_item in register_groups:
			#try first "random" item of one group
			next_register=group_item[0]

			if ((list(designGraph_merge.predecessors(next_register))[0] == all_gate_clk[next_register]) and (len(list(designGraph_merge.predecessors(next_register)))==2)):
				prep2=list(designGraph_merge.predecessors(next_register))[1]
			elif ((list(designGraph_merge.predecessors(next_register))[1] == all_gate_clk[next_register]) and (len(list(designGraph_merge.predecessors(next_register)))==2)):
				prep2=list(designGraph_merge.predecessors(next_register))[0]
			else:
				pprint('Achtung, clock gate wird nicht mehr gefunden wegen merging?')
				pprint(next_register)
				pprint(list(designGraph_merge.predecessors(next_register)))
				pprint(all_gate_clk[next_register])
				input()

			#with dynamic memory programming -> possible that this is used and therefore RELICcounter changes

			#~ for mem_item in list(memory.items()):
				#~ if (((mem_item[0]==farest_register) and (mem_item[1] in group_item)) or ((mem_item[1]==farest_register) and (mem_item[0] in group_item))):
					#~ SimScore=memory[mem_item]
					#~ break
			#~ else:
				#~ SimScore, RELICcounter, MemSim, MemSim_inv=HelperRELICstructure(prep1, prep2, prep1_inv, designGraph_merge, designGraph_merge_inverse, depth, RELICcounter, MemSim, MemSim_inv)
				#~ memory[(farest_register, next_register)]=SimScore

			#~ SimScore, RELICcounter, MemSim, MemSim_inv, el2, el3, used=HelperRELICstructure(prep1, prep2, prep1_inv, designGraph_merge, designGraph_merge_inverse, depth, RELICcounter, MemSim, MemSim_inv, el2, el3, used)
			SimScore, RELICcounter =HelperRELICstructure(prep1, prep2, prep1_inv, designGraph_merge, designGraph_merge_inverse, depth, RELICcounter)


			if SimScore > T2:
				group_item2=group_item + (farest_register, )
				register_groups.remove(group_item)
				register_groups.append(group_item2)
				del start_similarities[farest_register]
				found_group=1
				break

			#try second "random" item of the same group
			#~ next_register=group_item[1]
			#~
			#~ if ((list(designGraph_merge.predecessors(next_register))[0]!= 'INPUT_clk') and (list(designGraph_merge.predecessors(next_register))[0]!= 'INPUT_clk_i') and (list(designGraph_merge.predecessors(next_register))[0]!= 'INPUT_sys_clk') and (list(designGraph_merge.predecessors(next_register))[0]!= 'INPUT_i_clk') and (list(designGraph_merge.predecessors(next_register))[0]!= 'INPUT_clock')):
				#~ prep2=list(designGraph_merge.predecessors(next_register))[0]
			#~ else:
				#~ prep2=list(designGraph_merge.predecessors(next_register))[1]
					#~
			#~ SimScore, RELICcounter=HelperRELICstructure(prep1, prep2, prep1_inv, designGraph_merge, designGraph_merge_inverse, depth, RELICcounter)
			#~
			#~ if SimScore > T2:
				#~ pprint('group2')
				#~ group_item2=group_item + (farest_register, )
				#~ register_groups.remove(group_item)
				#~ register_groups.append(group_item2)
				#~ del start_similarities[farest_register]
				#~ found_group=1
				#~ break

		if found_group==0:
			for j in range(1, len(list(start_similarities.items()))):
				next_register=list(start_similarities.items())[j][0]

				if list(start_similarities.items())[j][1] <= T2:

					if ((list(designGraph_merge.predecessors(next_register))[0] == all_gate_clk[next_register]) and (len(list(designGraph_merge.predecessors(next_register)))==2)):
						prep2=list(designGraph_merge.predecessors(next_register))[1]
					elif ((list(designGraph_merge.predecessors(next_register))[1] == all_gate_clk[next_register]) and (len(list(designGraph_merge.predecessors(next_register)))==2)):
						prep2=list(designGraph_merge.predecessors(next_register))[0]
					else:
						pprint('Achtung, clock gate wird nicht mehr gefunden wegen merging?')
						pprint(next_register)
						pprint(list(designGraph_merge.predecessors(next_register)))
						pprint(all_gate_clk[next_register])
						input()


					#~ SimScore, RELICcounter, MemSim, MemSim_inv, el2, el3, used = HelperRELICstructure(prep1, prep2, prep1_inv, designGraph_merge, designGraph_merge_inverse, depth, RELICcounter, MemSim, MemSim_inv, el2, el3, used)
					SimScore, RELICcounter = HelperRELICstructure(prep1, prep2, prep1_inv, designGraph_merge, designGraph_merge_inverse, depth, RELICcounter)


					#with dynamic memory programming -> possible that this is used and therefore RELICcounter changes
					#~ memory[(farest_register, next_register)]=SimScore

					if SimScore > T2:
						register_groups.append((farest_register, next_register))
						del start_similarities[farest_register]
						del start_similarities[next_register]
						set_register=1
						break

				else:
					single_register.append(farest_register)
					del start_similarities[farest_register]
					set_register=1
					break

			if set_register==0:
				single_register.append(farest_register)
				del start_similarities[farest_register]

		if len(list(start_similarities.items()))==0:
			break

	if len(list(start_similarities.items()))==0:
		single_register.append(start_node)
	else:
		item_g=(start_node, )
		for item in list(start_similarities.items()):
			item_g = item_g + (item[0], )
		register_groups.append(item_g)
		#register_groups.append(set(start_similarities.items()))

	#~ pprint('SimScore1')
	#~ pprint(el2)
	#~ pprint(float(el2)/float(RELICcounter))
	#~ pprint('SimScore2')
	#~ pprint(el3)
	#~ pprint(float(el3)/float(RELICcounter))
	#~ pprint('used')
	#~ pprint(used)
	# ~ t1=time.time()		#debugging
	# ~ print('TIME group_register',t1-t0)		#debugging
	return register_groups, single_register, RELICcounter

## Helper function for Pair Similarity Score calculation
#
#  This function calculates the original and the inverse Pair Similarity Score and chooses the maximum of them.
def HelperRELICstructure(prep1, prep2,  designGraph_merge, depth, RELICcounter):
	#calculates SimScore1 and SimScore2 and chooses the maximum of them
	

	

	if str(prep1)[0:3]!='INV':
		regist1input=prep1
	else:
		regist1input=list(designGraph_merge.predecessors(prep1))[0]

	if str(prep2)[0:3]!='INV':
		regist2input=prep2
	else:
		regist2input=list(designGraph_merge.predecessors(prep2))[0]

	#calculate normal similarity score
	# ~ print('registerInput:	',regist1input,regist2input, depth)	#debugging
	SimScore1 = GetSimScore_seperat(designGraph_merge, designGraph_merge, regist1input,regist2input, int(depth+1))
	# ~ print('TotalSimScore:', SimScore1) 	#debugging
	#~ elapsed2=time.time()-start2
	#~ el2=el2+elapsed2
	#~ start3=time.time()
	#use inversed subgraph & calculate constructed inverse similarity score: comment if ignoring inverse
	# ~ SimScore2 = GetSimScore_seperat_INV(designGraph_merge_inverse, designGraph_merge, regist1input_inverse, regist2input, int(depth+1))
	#~ elapsed3=time.time()-start3
	#~ el3=el3+elapsed3

	#choose highest score
	# ~ SimScore=max(SimScore1, SimScore2)
	#~ if SimScore2>SimScore1:
		#~ used=used+1
	#~ SimScore=SimScore1 #use if ignoring inverse

	RELICcounter=RELICcounter+1

	# ~ t18=time.time()		#debugging
	# ~ print('TIME HelperRELICstructure', t18-t17)		#debugging

	return SimScore1, RELICcounter

## fastRELIC structure
#
#  This function starts the design graph preprocessing, calculates the inverse merged graph, starts the Pair Similarity Score calcuation and defines the identified state registers in the end.
#  Therefore, those registers are chosen as state registers which are successfully identified by fastRELIC (meaning are in groups with an item number smaller than or equal to a threshold T3 and have a feedback path.
def main_fastRELIC(design, designGraph, node_OI, all_gate_clk, defined_clocks):

	####################################################################
	#detect registers with fastRELIC
	####################################################################

	#Preprocessing (merging input wires)
	#1.case: two similar gates without inverter inbetween
	#2.case: two different gates with inverter inbetween

	# ~ t25=time.time() 		#debugging

	## @var designGraph_merge
	# The merged version of the original design netlist.
	designGraph_merge=nx.DiGraph()
	designGraph_merge=designGraph.copy()
	# ~ print('Merge_Graph:		',designGraph_merge.nodes())	#debugging
	node_list=list(designGraph.nodes())
	pprint(len(node_list))
	# ~ print(node_list)	#debugging
	# ~ print(designGraph.number_of_edges())	#debugging
	# ~ print(designGraph.edges())				#debugging

	designGraph_merge=mergeGraph(designGraph_merge, node_list)
	# ~ print('Merge_Graph2:	', designGraph_merge.nodes()) 	#debugging
	# ~ print('Merge_Graph2:	', designGraph_merge.edges()) 	#debugging

	#func.draw_digraph(design + 'merge_debugging', designGraph_merge)

	pprint('merged')
	pprint(len(list(designGraph.nodes())))
	pprint(len(list(designGraph_merge.nodes())))

	#Construct inverse graph by swapping OR and AND and invert inputs (except clock)
	mapping={}
	## @var designGraph_merge_inverse
	# The inverse merged version of the original design netlist.
	designGraph_merge_inverse=nx.DiGraph()
	designGraph_merge_inverse=designGraph_merge.copy()
	count_sub=0
	for node_bev in list(designGraph_merge.nodes()):
		if str(node_bev)[0:3]=='AND':
			mapping[node_bev]='OR' + str(node_bev)[4:]
		elif str(node_bev)[0:2]=='OR':
			mapping[node_bev]='AND' + str(node_bev)[3:]
		#Achtung: hier wurden jetzt die clock inputs ausgeschlossen
		elif ((str(node_bev)[0:5]=='INPUT') and (str(node_bev) not in defined_clocks)):
			for suc_sub in list(designGraph_merge.successors(node_bev)):
				if suc_sub[0:3]=='INV':
					for suc_sub2 in list(designGraph_merge.successors(suc_sub)):
						designGraph_merge_inverse.add_edge(node_bev,suc_sub2)
					designGraph_merge_inverse.remove_node(suc_sub)
				else:
					designGraph_merge_inverse.add_edge('INVX1_sub' + str(count_sub), suc_sub)
					designGraph_merge_inverse.remove_edge(node_bev,suc_sub)
					designGraph_merge_inverse.add_edge(node_bev, 'INVX1_sub' + str(count_sub))
			count_sub=count_sub+1
	nx.relabel_nodes(designGraph_merge_inverse, mapping, copy=False)

	#~ func.draw_digraph(design + 'merge_inverse', designGraph_merge_inverse)
	pprint('swapped')
	#~ pprint(len(list(designGraph.nodes())))
	#~ pprint(len(list(designGraph_merge.nodes())))

	#~ elapsed=time.time()-start
	#~ pprint(elapsed)

	##############################################################################################################################
	#Group Register regarding their similarity
	##############################################################################################################################

	start_node=sorted(node_OI)[func.constants['start_const']] #chosen random
	pprint(start_node)

	node_OI_without_start=node_OI.copy()
	node_OI_without_start.remove(start_node)

	register_groups, single_register, RELICcounter = group_register(node_OI_without_start, designGraph_merge, designGraph_merge_inverse, func.constants['depth_const'], start_node, func.constants['Textra_const'], all_gate_clk)

	regist_OIs=list(single_register)
	pprint(single_register)

	for regist_groups_item in register_groups:
		if func.constants['Textra2_const'] != 0:
			if len(regist_groups_item) <= func.constants['Textra2_const']:
				regist_OIs.extend(regist_groups_item)

	#check if identified state registers have feedback path: comment if FPs are ignored
	count=0
	count2=0
	designGraph_prune=designGraph.copy()
	regist_OI_it=list(regist_OIs)
	for item in regist_OI_it:
		fp, designGraph_prune, count, count2, one_cycle_found=find_feedbackpath(designGraph, designGraph_prune, item, sorted(node_OI), design, count, count2)
		if one_cycle_found==0:
			pprint('fastRELIC found a register which has no feedback path')
			pprint(item)
			regist_OIs.remove(item)

	# ~ t26=time.time() 		#debugging
	# ~ print('TIME main_fastRELIC',t26-t25)		#debugging

	return regist_OIs, single_register, register_groups, RELICcounter, start_node

## fastRELIC preparation structure if more than one clock region exists
#
#  This function starts the design graph preprocessing and calculates the inverse merged graph.
def main_fastRELICprepformulti(design, designGraph, defined_clocks):

	####################################################################
	#detect registers with fastRELIC
	####################################################################

	#Preprocessing (merging input wires)
	#1.case: two similar gates without inverter inbetween
	#2.case: two different gates with inverter inbetween

	#~ start=time.time()

	# ~ t23=time.time() 		#debugging

	designGraph_merge=nx.DiGraph()
	designGraph_merge=designGraph.copy()
	node_list=list(designGraph.nodes())
	pprint(len(node_list))

	designGraph_merge=mergeGraph(designGraph_merge, node_list)

	#~ func.draw_digraph(design + 'merge', designGraph_merge)

	pprint('merged')
	pprint(len(list(designGraph.nodes())))
	pprint(len(list(designGraph_merge.nodes())))

	#Construct inverse graph by swapping OR and AND and invert inputs (except clock)
	mapping={}
	designGraph_merge_inverse=nx.DiGraph()
	designGraph_merge_inverse=designGraph_merge.copy()
	count_sub=0
	for node_bev in list(designGraph_merge.nodes()):
		if str(node_bev)[0:3]=='AND':
			mapping[node_bev]='OR' + str(node_bev)[4:]
		elif str(node_bev)[0:2]=='OR':
			mapping[node_bev]='AND' + str(node_bev)[3:]
		#Achtung: hier wurden jetzt die clock inputs ausgeschlossen
		elif ((str(node_bev)[0:5]=='INPUT') and (str(node_bev) not in defined_clocks)):
			for suc_sub in list(designGraph_merge.successors(node_bev)):
				if suc_sub[0:3]=='INV':
					for suc_sub2 in list(designGraph_merge.successors(suc_sub)):
						designGraph_merge_inverse.add_edge(node_bev,suc_sub2)
					designGraph_merge_inverse.remove_node(suc_sub)
				else:
					designGraph_merge_inverse.add_edge('INVX1_sub' + str(count_sub), suc_sub)
					designGraph_merge_inverse.remove_edge(node_bev,suc_sub)
					designGraph_merge_inverse.add_edge(node_bev, 'INVX1_sub' + str(count_sub))
			count_sub=count_sub+1
	nx.relabel_nodes(designGraph_merge_inverse, mapping, copy=False)

	#~ func.draw_digraph(design + 'merge_inverse', designGraph_merge_inverse)
	pprint('swapped')
	#~ pprint(len(list(designGraph.nodes())))
	#~ pprint(len(list(designGraph_merge.nodes())))

	#~ elapsed=time.time()-start
	#~ pprint(elapsed)

	# ~ t24=time.time() 		#debugging
	# ~ print('TIME main_fastRELICprepformulti',t24-t23)		#debugging

	return designGraph_merge, designGraph_merge_inverse

## fastRELIC structure if more than one clock region exists
#
#  This function starts the Pair Similarity Score calcuation and defines the identified state registers in the end.
#  Therefore, those registers are chosen as state registers which are successfully identified by fastRELIC (meaning are in groups with an item number smaller than or equal to a threshold T3 and have a feedback path.
def main_fastRELICformulti(design, designGraph, node_OI, designGraph_merge, designGraph_merge_inverese, all_gate_clk):
	##############################################################################################################################
	#Group Register regarding their similarity
	##############################################################################################################################
	# ~ t21=time.time() 		#debugging
	start_node=sorted(node_OI)[func.constants['start_const']] #chosen random
	pprint(start_node)

	node_OI_without_start=node_OI.copy()
	node_OI_without_start.remove(start_node)

	register_groups, single_register, RELICcounter = group_register(node_OI_without_start, designGraph_merge, designGraph_merge_inverse, func.constants['depth_const'], start_node, func.constants['Textra_const'], all_gate_clk)

	regist_OIs=list(single_register)
	pprint(single_register)

	for regist_groups_item in register_groups:
		if func.constants['Textra2_const'] != 0:
			if len(regist_groups_item) <= func.constants['Textra2_const']:
				regist_OIs.extend(regist_groups_item)

	#check if identified state registers have feedback path: comment if FPs are ignored
	count=0
	count2=0
	designGraph_prune=designGraph.copy()
	regist_OI_it=list(regist_OIs)
	for item in regist_OI_it:
		fp, designGraph_prune, count, count2, one_cycle_found=find_feedbackpath(designGraph, designGraph_prune, item, sorted(node_OI), design, count, count2)
		if one_cycle_found==0:
			pprint('fastRELIC found a register which has no feedback path')
			pprint(item)
			regist_OIs.remove(item)

	# ~ t22=time.time() 		#debugging
	# ~ print('TIME main_fastRELICformulti',t22-t21)		#debugging

	return regist_OIs, single_register, register_groups, RELICcounter, start_node



if __name__ == '__main__':

	## @var designPath
	# Defines path for used design.
	designPath = "../Designs/"

	## @var libPath
	# Defines path for cell library.
	libPath = "./cell_library/osu035_stdcells.lib"

	## @var design
	# Defines name of design file (without the file ending .v).
	#~ design = "automated_osu035_altered_sdr_ctrl_latest_final_sdrc_top"
	#~ design="b08_reset"

	designGraph, PIs, POs, nodes, cell_lib = main_prep(designPath, libPath, design)
	## @var designGraph_rep
	# A copy of the designGraph to consider the asynchronous registers in the netlist.
	designGraph_rep=nx.DiGraph()
	designGraph_rep=designGraph.copy()
	node_OI, all_gate_clk = main_defineReg(nodes, PIs)
	designGraph_rep, node_OI, all_gate_clk = main_mapR(designGraph_rep, node_OI, PIs, POs, all_gate_clk)
	node_OI_more, defined_clocks, all_gate_clk = main_differentClocks(node_OI, designGraph_rep, all_gate_clk)
	if len(defined_clocks)>1:
		#case that there exist more than one clock
		## @var time_start
		# Starting time measurement for fastRELIC. Excluding the register identification, the asynchronous register mapping and the identification of different clock regions. Including the merging and determination of the single registers and groups.
		time_start=time.time()
		## @var regist_OI
		# A dictionary of the identified state registers for each clock region.
		regist_OI=OrderedDict()
		## @var single_register
		# A dictionary of the identified single registers for each clock region.
		single_register=OrderedDict()
		## @var register_groups
		# A dictionary of the identified register groups for each clock region.
		register_groups=OrderedDict()
		## @var calcNumber
		# A dictionary of the needed Pair Similarity Score calculations (only the numbers of maximum values, so actual number should be doubled) for each clock region.
		calcNumber=OrderedDict()
		## @var start_node
		# A dictionary of the start node for each clock region.
		start_node=OrderedDict()
		designGraph_merge, designGraph_merge_inverse = main_fastRELICprepformulti(design, designGraph_rep, defined_clocks)
		for k in sorted(defined_clocks):
			regist_OI[k], single_register[k], register_groups[k], calcNumber[k], start_node[k]= main_fastRELICformulti(design, designGraph_rep, node_OI_more[k], designGraph_merge, designGraph_merge_inverse, all_gate_clk)
		## @var time_end
		# Ending time measurement for fastRELIC.
		time_end=time.time()
		## @var elapsed_time
		# Measured time for fastRELIC.
		elapsed_time=time_end-time_start
	else:
		#original case
		time_start=time.time()
		regist_OI=OrderedDict()
		single_register=OrderedDict()
		register_groups=OrderedDict()
		calcNumber=OrderedDict()
		start_node=OrderedDict()
		regist_OI[defined_clocks[0]], single_register[defined_clocks[0]], register_groups[defined_clocks[0]], calcNumber[defined_clocks[0]], start_node[defined_clocks[0]] = main_fastRELIC(design, designGraph_rep, node_OI_more[defined_clocks[0]], all_gate_clk, defined_clocks)
		time_end=time.time()
		elapsed_time=time_end-time_start

	pprint(elapsed_time)
	pprint('state registers with fastRELIC')
	pprint(regist_OI)
	for c in sorted(defined_clocks):
		pprint(c)
		pprint(len(regist_OI[c]))
		pprint('# Pair SimScore calculations')
		pprint(calcNumber[c])

	## @var rp
	# Number of Pair Similarity Score calculations (only the numbers of maximum values, so actual number should be doubled) which would be needed for RELIC.
	rp=float(len(node_OI)*(len(node_OI)-1))/float(2)


	## @var dicFile
	# Path for file which stores the identified state registers.# '/fastRelic_dp'+ str(func.constants['depth_const'])+'_th_' + str(func.constants['T1']) + '.txt
	dicFile=open(func.OutputPath + "/SR_dicts/depth" + str(func.constants['depth_const'])+ '/fastRelic_dp'+ str(func.constants['depth_const'])+'_th_' + str(func.constants['T1']) + '_iter.txt' , 'w+')
	dicFile.write(str(regist_OI))

	## @var inputFile
	# Path for file which stores fastRELIC information.
	inputFile = open(func.OutputPath + "/counter/depth"+ str(func.constants['depth_const']) + '/fastRelic_dp'+ str(func.constants['depth_const'])+'_th_' + str(func.constants['T1']) + '_iter.txt', 'w+')
	for c in sorted(defined_clocks):
		inputFile.write(c)
		inputFile.write("\n")
		inputFile.write("\n")
		inputFile.write("RELIC chosen state registers\n")
		inputFile.write("\n")
		inputFile.write("\n")
		for item in sorted(regist_OI[c]):
			inputFile.write(str(item))
			inputFile.write("\n")
		inputFile.write("\n")
		inputFile.write("\n")
		inputFile.write("fastRELIC group length and items")
		inputFile.write("\n")
		inputFile.write("\n")
		for item in sorted(register_groups[c]):
			inputFile.write(str(len(item)))
			inputFile.write("\n")
			for item2 in item:
				inputFile.write(str(item2))
				inputFile.write("\n")
			inputFile.write("\n")
		inputFile.write("single register")
		inputFile.write("\n")
		for item in sorted(single_register[c]):
			inputFile.write(str(item))
			inputFile.write("\n")
		inputFile.write("\n")
		inputFile.write("\n")
		inputFile.write("Number of Registers:")
		inputFile.write(str(len(node_OI_more[c])))
		inputFile.write("\n")
		inputFile.write("Number of Register Pairs:")
		rp=float(len(node_OI_more[c])*(len(node_OI_more[c])-1))/float(2)
		inputFile.write(str(rp))
		inputFile.write("\n")
		inputFile.write("Number of Pair SimScore calculations:")
		inputFile.write(str(calcNumber[c]))
		inputFile.write("\n")
		inputFile.write("Start Node")
		inputFile.write(start_node[c])
		inputFile.write("\n")
		inputFile.write("\n")
	inputFile.write("\n")
	inputFile.write("\n")
	inputFile.write("Number of Registers:")
	inputFile.write(str(len(node_OI)))
	inputFile.write("\n")
	inputFile.write("Number of Register Pairs:")
	rp=float(len(node_OI)*(len(node_OI)-1))/float(2)
	inputFile.write(str(rp))
	inputFile.write("Time:")
	inputFile.write(str(elapsed_time))
	inputFile.close()

	singleFile = open(func.OutputPath + "/single/depth"+ str(func.constants['depth_const']) + '/fastRelic_dp'+ str(func.constants['depth_const'])+'_th_' + str(func.constants['T1']) + '_iter.txt', 'w+')
	for c in sorted(defined_clocks):
		for item in sorted(single_register[c]):
			singleFile.write(str(item))
			singleFile.write("\n")
		singleFile.write("RELIC chosen state registers\n")
		for item in sorted(regist_OI[c]):
			singleFile.write(str(item))
			singleFile.write("\n")
	singleFile.close()
