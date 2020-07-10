#!usr/bin/ipython3
#func.py


"""!
 Set of smaller, helpful functions

 alphabetically sorted functions required to set globals, write and read, dump and load different formats to and from file, and write info file
"""



# imports
import sys
import timing
import json
from datetime import date
from datetime import datetime
from pyeda.inter import expr
import networkx as nx
import matplotlib.pyplot as plt
from pprint import pprint
from ast import literal_eval
import os
import pickle
from collections import Counter


## global variable design_nameFread_s
design_name = None
## global variable control_wires
control_wires = None
## global variable info_dictionary used to write info File of design
info_dict = None
## global variable constants used for storing all required constants as dictionary
constants = None
# node types
node_types = None
# current step we are in for printing
current_step = None

def create_paths():
	"""!
	Function checks if all required paths exist, and if not, creates them
	"""
	# create output dir for files, if not already existant
	if not os.path.exists("outputFiles/"):
		os.makedirs("outputFiles/")
	if not os.path.exists("outputFiles/infoDump/"):
		os.makedirs("outputFiles/infoDump/")
	if not os.path.exists("outputFiles/" + design_name  ):
		os.makedirs("outputFiles/" + design_name )
	if not os.path.exists("outputFiles/" + design_name + "/subcircuits/"):
		os.makedirs("outputFiles/" + design_name + "/subcircuits")
	if not os.path.exists("outputFiles/" + design_name + "/qbfFiles/"):
		os.makedirs("outputFiles/" + design_name + "/qbfFiles/")
	if not os.path.exists("outputFiles/" + design_name + "/dataDump/"):
		os.makedirs("outputFiles/" + design_name + "/dataDump/")
	if not os.path.exists("outputFiles/" + design_name + "/results/"):
		os.makedirs("outputFiles/" + design_name + "/results/")
	if not os.path.exists("outputFiles/" + design_name + "/smtFiles/"):
		os.makedirs("outputFiles/" + design_name + "/smtFiles/")
	if not os.path.exists("outputFiles/" + design_name + "/clusters/"):
		os.makedirs("outputFiles/" + design_name + "/clusters/")
	return

def draw_digraph(filename, G):
	"""!
	Generic function to save DiGraph as pydot file
	@param [in] filename as string of name of dotfile
	@param [in] G as networkX DiGraph to be drawn

	"""

	P = nx.drawing.nx_pydot.to_pydot(G)
	P.write_raw(filename + ".dot")

	return




def draw_progress_bar(percent, barLen=60):
	"""!
	Function draws progress bar

	@param [in] percent as float
	@param [in] barLen as int of lenght of progress bar

	"""
	sys.stdout.write("\r")
	progress = ""
	for i in range(barLen):
		if i < int(barLen * percent):
			progress += "="
		else:
			progress += " "
	sys.stdout.write("[ %s ] %.2f%%" % (progress, percent * 100))
	sys.stdout.flush()

def dump_digraph(filename, G):
	"""!
	Functions writes networkx Digraph as gpickle, adjacency list, gexf and gml file
	@param [in] filename as string
	@param [in] G as networkX DiGraph
	"""
	print_info("Dumping digraph to " + filename)
	nx.write_gpickle(G, filename + design_name + ".gpickle")
	nx.write_adjlist(G, filename + design_name )
	nx.write_yaml(G, filename+ ".yaml")
	nx.write_gml(G, filename + design_name + ".gml")
	nx.write_gexf(G, filename + design_name +  ".gexf" , prettyprint=True)
	nx.write_gml(G, filename + design_name + ".gml")



	return

def print_info(text):
	"""!
	Function outputs timing data, design and step info, as well as a text
	@param[in] text as string of text to be output
	"""
	try:
		print(str(datetime.now().replace(microsecond=0)) + " - " + design_name + " - " + current_step + " : " + text)
	except:
		print(str(datetime.now().replace(microsecond=0)) + " - "  + text)
	return

def read_circuit_functions(filename):
	"""!
	Function reads circuit functions from file to dictionary
	"""
	print_info("Reading circuit functions from file")

	circuitFunctions_str = read_dict(filename)

	circuitFunctions = {}

	for circuit in circuitFunctions_str:
		circuitFunctions[circuit] = {}
		for func in circuitFunctions_str[circuit].keys():
			circuitFunctions[circuit][func] = {}
			if func.startswith("INPUTS"):
				circuitFunctions[circuit][func] = literal_eval(circuitFunctions_str[circuit][func])
			else:
				circuitFunctions[circuit][func] = expr(circuitFunctions_str[circuit][func])

	#~ print(str(datetime.now().replace(microsecond=0)) + " " + design_name +" -Finished reading circuit functions from file")

	return circuitFunctions

def read_cuts(filename):
	"""!
	Function reads dictionary of pyeda cuts from file

	"""
	print_info("Reading cuts from file")
	cuts_str = read_dict(filename)
	cuts = {}

	counter = 0
	for node in cuts_str:
		cuts[node] = []
		for cut in cuts_str[node]:
			cuts[node].append(expr(cut))
		counter = counter +1
		draw_progress_bar(counter / len(cuts_str))

	print()
	return cuts

def read_dict(filename):
	"""!
	Generic function to read Dict dumped in json format
	@param [in] filename as string of name of file to be read
	@retval read_Dict as dictionary
	"""

	print_info("Reading dictionary from file " + filename)


	with open(filename, 'r') as handle:
		try:
			read_dict = json.load(handle)
		except ValueError:
			read_dict = {}

	#~ print(str(datetime.now().replace(microsecond=0)) + " " + design_name +" - Finished reading dictionary from file")

	return read_dict

def read_digraph(filename):
	"""!
	Function reads graph from file as networkx Digraph
	"""
	print_info("Reading digraph from file " + filename)
	#~ G = nx.read_gpickle(filename + ".gpickle")
	#~ G = nx.read_adjlist(filename)
	G = nx.read_gexf(filename + ".gexf")
	#~ G = nx.read_gml(filename + ".gml")


	return G

def read_named_circuits(filename):
	"""!
	Functions reads circuits from file and saves in dictionary namedCircuits
	"""
	print_info("Reading circuits from file")
	namedCircuits = read_string(filename)
	namedCircuits = literal_eval(namedCircuits)


	for i in range(len(namedCircuits)):

		graph = read_digraph("outputFiles/" + design_name + "/dataDump/" + str(namedCircuits[i][2]))
		namedCircuits[i].insert( 0, [graph])

	#~ print(str(datetime.now().replace(microsecond=0)) + " " + design_name +" -Finished reading circuits from file")

	return namedCircuits

def read_process(arg):
	"""!
	Function reads process variables from command line arguments
	"""
	if len(arg) > 1:
		if arg[1] == "all":
			process_word_id = True
			process_data_path = True
			process_subcircuit = True
			process_circuit_id = True
			process_postP = True
			process_cluster = False
		elif arg[1] == 'wordid':
			process_word_id = True
			process_data_path = False
			process_subcircuit = False
			process_circuit_id = False
			process_postP = False
			process_cluster = False
		elif arg[1] == "datapath":
			process_word_id = False
			process_data_path = True
			process_subcircuit = False
			process_circuit_id = False
			process_postP = False
			process_cluster = False
		elif arg[1] == "subcircuit":
			process_word_id = False
			process_data_path = False
			process_subcircuit = True
			process_circuit_id = False
			process_postP = False
			process_cluster = False
		elif arg[1] == "subcircuit+":
			process_word_id = False
			process_data_path = False
			process_subcircuit = True
			process_circuit_id = True
			process_postP = True
			process_cluster = False
		elif arg[1] == "circuitid":
			process_word_id = False
			process_data_path = False
			process_subcircuit = False
			process_circuit_id = True
			process_postP = False
			process_cluster = False
		elif arg[1] == "postP":
			process_word_id = False
			process_data_path = False
			process_subcircuit = False
			process_circuit_id = False
			process_postP = True
			process_cluster = False
		elif arg[1] == "cluster":
			process_word_id = False
			process_data_path = False
			process_subcircuit = False
			process_circuit_id = False
			process_postP = False
			process_cluster = True
		else:
			raise ValueError("Unknown keyword " + sys.argv[1] + ". Please check allowed keywords (all, wordid, datapath, subcircuit, subcircuit+, circuitid, postP) and retry")
	else:
		process_word_id = True
		process_data_path = True
		process_subcircuit = True
		process_circuit_id = True
		process_postP = True
		process_cluster = True

	info_dict["process_word"]= process_word_id
	info_dict["process_data"] = process_data_path
	info_dict["process_subc"] = process_subcircuit
	info_dict["process_cirid"] = process_circuit_id
	info_dict["process_PP"] = process_postP
	info_dict["process_cluster"] = process_cluster
	return process_word_id, process_data_path, process_subcircuit, process_circuit_id, process_postP, process_cluster

def read_string(filename):
	"""!
	Generic function to read string from file

	@param [in] filename as string of name of file to be read
	"""
	with open(filename, "r+") as handle:
		read = handle.read()

	return read

def set_constants(constants_dictionary):
	"""!
	Function writes global variable constants

	Function writes global dictionary constants, containing data regarding all constants used

	@param [in] constants_dictionary as dictionary containing all constants, not global
	@param [out] contants as global dictionary of constants
	"""

	global constants
	constants = constants_dictionary

	return

def set_control_wires(controlWireList):
	"""!
	Function sets global varibale control_wires
	@param [in] controlWireList as list of control Wires
	@retval control_wires as list
	"""
	global control_wires
	control_wires = controlWireList

	return control_wires

def set_current_step(text):
	"""!
	Function sets global varibale current step
	@param [in] text as string of current step

	"""

	global current_step
	current_step = text

	return

def set_design_name(name):
	"""!
	Function sets global variable desing_name

	@param [in] name as string of design Name
	@retval desing_name as string
	"""
	global design_name
	design_name = name

	return design_name

def set_info_dict():
	"""!
	Function writes global variable info_dict which is later used to write Info file

	"""

	print(str(datetime.now().replace(microsecond=0)) + " - Now setting the info dict")

	global info_dict
	info_dict = {}
	# initalise info_dict in case some things aren't written

	info_dict["elapsed"] = 0
	info_dict["elapsedPars"] = 0
	info_dict["elapsedword_id"] = 0
	info_dict["elapsedCutFind"] = 0
	info_dict["elapsedCutWords"] = 0
	info_dict["elapsedShape"] = 0
	info_dict["elapsedCheckWord"] = 0
	info_dict["elapseddata_path"] = 0
	info_dict["elapsedDataPath"] = 0
	info_dict["elapsedClean"] =0
	info_dict["elapsedDraw"] =0
	info_dict["elapsedsubID"] = 0
	info_dict["elapsedBound"] = 0
	info_dict["elapsedsubC"] = 0
	info_dict["elapsedCheckSubC"] = 0
	info_dict["elapsedPrint"] = 0
	info_dict["elapsedFunc"] = 0
	info_dict["elapsedSAT"] = 0
	info_dict["elapsedMatch"] = 0
	info_dict["gates"] = "no Info"
	info_dict["avrgCutLen"] = "no Info"
	info_dict["maxCutLen"] = "no Info"
	info_dict["wordsCut"] = "no Info"
	info_dict["bitsliceWords"] = "no Info"
	info_dict["inpsWordCut"] = "no Info"
	info_dict["wordsShape"] = "no Info"
	info_dict["uncheckedWords"] = "no Info"
	info_dict["sourceWords"] = "no Info"
	info_dict["chainWords"] = "no Info"
	info_dict["control_wires"] = "no Info"
	info_dict["propWords"] = "no Info"
	info_dict["candWords"] = "no Info"
	info_dict["dataPath"] = "no Info"
	info_dict["boundWords"] = "no Info"
	info_dict["checkedCircuits"] = "no Info"
	info_dict["circuits"] = []
	info_dict["cov"] = "no Info"
	info_dict["uniqueWords"] = 0
	info_dict["process_word"] = "no Info"
	info_dict["process_data"] = "no Info"
	info_dict["process_subc"] = "no Info"
	info_dict["process_cirid"] ="no Info"
	info_dict["process_cluster"] ="no Info"
	info_dict["pRepDictLen"] = "no Info"

	return

def set_node_types(nodeTypes):
	"""!
	Function sets global varibale node_types
	@param [in] text as string of node_types

	"""
	global node_types
	node_types = nodeTypes

	return

def show_digraph(G):
	"""!
	Generic function to save DiGraph as pydot file
	@param [in] filename as string of name of dotfile
	@param [in] G as networkX DiGraph to be drawn

	"""
	nx.draw(G)
	plt.show()

	return




def top_sort(graph):


	if nx.is_directed_acyclic_graph(graph):
		sort =  nx.topological_sort(graph)
	else:
		# find strongly connected components
		# do top sort, except when in strongly connected compoents

		## find cycles and preprocess
		print("len simple cylces is")
		pprint(len(list(nx.simple_cycles(graph))))

		cycles = [x for x in nx.simple_cycles(graph) if len(x) >= 2] # fixme: really large and slow JB 17.08.2018
		cycle_dict = {}
		node_dict = {}
		#~ pprint(cycles)
		#~ print("#######################")
		all_nodes = [x for cycle in cycles for x in cycle]
		cnt = Counter(all_nodes)
		n = [ k for k,v in cnt.items() if v > 1]
		for node in n:
			# get cylces with n, remove, merge and readd
			cycles_node = [cycle for cycle in cycles if node in cycle]
			merged = []
			for cycle in cycles_node:
				cycles.remove(cycle)
				merged.extend(cycle)
			cycles.append(list(set(merged)))

		H = nx.DiGraph(graph)
		counter = 0
		# replace cycles with bigger node
		for cycle in cycles:
			cycle_dict["cycle" + str(counter)] = cycle

			for node in cycle:
				node_dict[str(node)] = "cycle" + str(counter)

				for pred in list(H.predecessors(node)):
					if pred not in cycle:
						if H.has_node(pred):
							H.add_edge(pred, "cycle" + str(counter) )
						else:
							H.add_edge(node_dict[str(pred)], "cycle" + str(counter))

				for suc in list(H.successors(node)):
					if suc not in cycle:
						if H.has_node(suc):
							H.add_edge("cycle" + str(counter) , suc)
						else:
							H.add_edge("cycle" + str(counter), node_dict[str(suc)])
				H.remove_node(node)
			counter = counter +1

		top_sort = nx.topological_sort(H)

		pprint("now sorting cycles")
		sort = []
		for val in list(top_sort):
			if val.startswith("cycle"):
				nodes = cycle_dict[val]
				cycle_sort_graph = graph.subgraph(nodes).copy()
				for node in cycle_sort_graph.nodes():
					if node.startswith("DFF"):
						#todo: config for break before or after dff
						cycle_sort_graph.remove_edge(list(graph.predecessors(node))[0], node) # fixme: aug 17, not quite sure it this is correct
						# ~ cycle_sort_graph.remove_edge( list(cycle_sort_graph.predecessors(node))[0], node )
				cycle_sort = nx.topological_sort(cycle_sort_graph)
				sort.extend(cycle_sort)

			else:
				sort.append(val)




	return sort


def write_circuit_functions(filename, circuitFunctions):
	"""!
	Function writes dictionary of circuit functions to file
	"""
	print_info("Writing circuit functions to file")
	circuitFunctions_str = {}
	for circuit in circuitFunctions:
		circuitFunctions_str[circuit] = {}

		for func in circuitFunctions[circuit].keys():
			circuitFunctions_str[circuit][func] = str(circuitFunctions[circuit][func])

	write_dict(filename, circuitFunctions_str)

	#~ print(str(datetime.now().replace(microsecond=0)) + " " + design_name +" -Finished writing circuit functions to file")

	return

def write_cuts(filename, cuts):
	"""!
	Function writes dictionary of pyeda cuts to file
	"""
	print_info("Writing cuts to file")

	cuts_str = {}
	for node in cuts:
		cuts_str[node] = []
		for cut in cuts[node]:
			cuts_str[node].append(str(cut))

	write_dict(filename, cuts_str)

	#~ print(str(datetime.now().replace(microsecond=0)) + " " + design_name +" - Finished writing cuts to file")

	return

def write_dict(filename, write):
	"""!
	Generic function to dump Dict to file in json format
	@param [in] filename as string of name of file to be dumped
	@param [in] write_dict as dictionary to be written
	"""

	print_info("Writing dictionary to file " + filename)

	with open(filename, 'w+') as handle:
		json.dump(write, handle)

	#~ print(str(datetime.now().replace(microsecond=0)) +" - Finished writing dictionary to file")


	return

def write_info_file(matches):
	"""!
	Function writes Info File, from info_dict

	The Info File contains information regarding the design, such as amount of words found, amount of subcircuits found, timing information and results. It can easily be copied to excel, allowing for simple comparison.

	@param [in] constants as dictionary of constants
	@param [in] matches as list of subcurcuit matches
	@param [out] info_file is written to disk
	"""

	print_info("Writing info file")

	# open info file
	infoFile = open("outputFiles/infoDump/" + design_name + str(datetime.now().replace(microsecond=0)) + ".txt", 'w+')

	# write general info to File
	infoFile.write('Date; ' + str(date.today()) + "\n")
	infoFile.write('design_name; ' + design_name + "\n")
	infoFile.write("\n")
	infoFile.write("Constants\n")
	infoFile.write("Cut Size;" + str(constants["cutSize"]) + "\n")
	infoFile.write("Shape Size;" + str(constants["shapeSize"]) + "\n")
	infoFile.write("Allowed Word Size;" + str(constants["wordLenght"]) + "\n")
	infoFile.write("partWords;" + str(constants["partWord"]) + "\n")
	infoFile.write("Word Distance;" + str(constants["wordDistance"]) + "\n")
	infoFile.write("Search Depth;" + str(constants["searchDepth"]) + "\n")
	infoFile.write("Do word id?;" + str(info_dict["process_word"]) + "\n")
	infoFile.write("Do data_path?;" + str(info_dict["process_data"]) + "\n")
	infoFile.write("Do find subcircuits?;" + str(info_dict["process_subc"]) + "\n")
	infoFile.write("Do id circuits?;" + str(info_dict["process_cirid"]) + "\n")



	# write Timing info to file
	infoFile.write("\n")
	infoFile.write("Timing;" + timing.secondsToStr(info_dict["elapsed"]) + "\n")
	infoFile.write("Parsing;" + timing.secondsToStr(info_dict["elapsedPars"]) + "\n")
	infoFile.write("WordID Total;" + timing.secondsToStr(info_dict["elapsedword_id"]) + "\n")
	infoFile.write("   Cut ID;" + timing.secondsToStr(info_dict["elapsedCutFind"]) + "\n")
	infoFile.write("   Words from Cut;" + timing.secondsToStr(info_dict["elapsedCutWords"]) + "\n")
	infoFile.write("   Shape ID & Words from Shape;" + timing.secondsToStr(info_dict["elapsedShape"]) + "\n")
	infoFile.write("   Word Check;" + timing.secondsToStr(info_dict["elapsedCheckWord"]) + "\n")
	infoFile.write("\n")
	infoFile.write("WordProp Total;" + timing.secondsToStr(info_dict["elapseddata_path"]) + "\n")
	infoFile.write("   data  calc;" + timing.secondsToStr(info_dict["elapsedDataPath"]) + "\n")
	infoFile.write("   data path cleaning;" + timing.secondsToStr(info_dict["elapsedClean"]) + "\n")
	infoFile.write("   data path drawring;" + timing.secondsToStr(info_dict["elapsedDraw"]) + "\n")
	infoFile.write("\n")
	infoFile.write("Subcircuit ID Total;" + timing.secondsToStr(info_dict["elapsedsubID"]) + "\n")
	infoFile.write("   finding Boundary words;" + timing.secondsToStr(info_dict["elapsedBound"]) + "\n")
	infoFile.write("   finding subcircuits;" + timing.secondsToStr(info_dict["elapsedsubC"]) + "\n")
	infoFile.write("   checking subcircuits;" + timing.secondsToStr(info_dict["elapsedCheckSubC"]) + "\n")
	infoFile.write("   printing subcircuits;" + timing.secondsToStr(info_dict["elapsedPrint"]) + "\n")
	infoFile.write("   finding subcircuit functions;" + timing.secondsToStr(info_dict["elapsedFunc"]) + "\n")
	infoFile.write("Matching ;" + timing.secondsToStr(info_dict["elapsedMatch"]) + "\n")
	infoFile.write("   SAT;" + timing.secondsToStr(info_dict["elapsedSAT"]) + "\n")
	infoFile.write("\n")

	# write Stats to info
	infoFile.write("Stats\n")
	infoFile.write("Nr of gates;" + str(info_dict["gates"]) + "\n")
	infoFile.write("Average Nr of Cuts;" + str(info_dict["avrgCutLen"])+ "\n")
	infoFile.write("Max cuts for one Node;" + str(info_dict["maxCutLen"]) + "\n")
	infoFile.write("Nr of words found from Cut;" + str(info_dict["wordsCut"]) + "\n")
	infoFile.write("Size of pRep dictionary from file;" + str(info_dict["pRepDictLen"]) + "\n")
	infoFile.write("Nr of words found from bitslice;" + str(info_dict["bitsliceWords"]) + "\n")
	infoFile.write("Nr of words found from bitslice input;" + str(info_dict["inpsWordCut"]) + "\n")
	infoFile.write("Nr of words found from shape;" + str(info_dict["wordsShape"]) + "\n")
	infoFile.write("Unchecked source Words (without PIs/POs);" + str(info_dict["uncheckedWords"]) + "\n")
	infoFile.write("Checked source Words;" + str(info_dict["sourceWords"]) + "\n")
	infoFile.write("Chain words;" + str(info_dict["chainWords"]) + "\n")
	infoFile.write("Nr of control Wires;" + str(info_dict["control_wires"]) + "\n")
	infoFile.write("Nr of propergated Words;" + str(info_dict["propWords"]) + "\n")
	infoFile.write("Nr of candidate Words;" + str(info_dict["candWords"]) + "\n")
	infoFile.write("Lenght of DataPath;" + str(info_dict["dataPath"]) + "\n")
	infoFile.write("Nr of boundary word sets;" + str(info_dict["boundWords"]) + "\n")
	infoFile.write("Nr of checked subcircuits;" + str(info_dict["checkedCircuits"]) + "\n")

	# write Matches to info File
	infoFile.write("Nr of matches;" + str(len(matches)) + "\n")
	infoFile.write("Matches are; \n")
	for match in matches:
		infoFile.write("\t  "+ str(match) + "\n")
	infoFile.write("Coverage of matches;" + str(info_dict["cov"]) + "\n")
	infoFile.write("\n")

	infoFile.write("subcircuit Info\n")
	for circuit in info_dict["circuits"]:
		infoFile.write(circuit + " contains " + str(info_dict["circuits"][circuit]) + " nodes. \n")
		matchList = [x.values() for x in matches if list(x.keys())[0] == circuit]
		if matchList:
			for match in matchList:
				infoFile.write(circuit + " is a match to " + str(list(match)[0]) + "\n")
	infoFile.write("\n")

	# close info file
	infoFile.close()

	return

def write_named_circuits(filename, namedCircuits):
	"""!
	Function writes dictionary namedCircuits to file
	"""

	print_info("Writing circuits to file")
	namedCircuits_str = []

	for i  in range(len(namedCircuits)):

		dump_digraph("outputFiles/" + design_name + "/dataDump/" + str(namedCircuits[i][3]), namedCircuits[i][0][0])
		namedCircuits_str.extend([[namedCircuits[i][1], namedCircuits[i][2], namedCircuits[i][3]]])


	write_string(filename, str(namedCircuits_str))

	return

def write_string(filename, write):
	"""!
	Generic function to write string to file

	@param [in] filename as string of name of file to be written
	@param [in] write_string as string to be written
	"""
	with open(filename, "w+") as handle:
		handle.write(write)

	return
