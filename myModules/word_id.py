#!usr/bin/ipython3
#word_id.py

"""!
Functions to identifiy words by analysing structural and functional characteristics

Function finds set of feasible cuts and shape of each node, and groups nodes with similar shape or feasible cut into potential words.
"""

# imports
# import myparsing
# import myfunctions
from pprint import pprint
import timing
from ast import literal_eval
from datetime import datetime
from pyeda.inter import exprvar, expr2truthtable
from pyeda.boolalg.table import PC_ZERO, PC_ONE, PC_DC
from pyeda.boolalg import boolfunc
import networkx as nx
import func
# import word_check
# import subcircuit_gen
import itertools
import math
import pydotplus

from networkx.algorithms import bipartite


########################################################################
	# Structural word identification
########################################################################

def find_shapes(G, nodeTypes):
	"""!
	Function finds Shape of each node in design

	Function performs recursive DFS on all nodes in design, up to depth constant["shapeSize"], to find "shape" of node
	runtime = O(|V|+|E|)
	@param [in] G as Digraph of design
	@parem [in] nodeTypes as dictionary nodeName : nodeType
	@retval shape as dictionary of node: shape
	"""
	# timing and printing
	startShape = timing.clock()
	func.print_info("Step 1: Finding shape of all nodes in design")
	# definitions and instantiations
	# shape is a dictionary of {node: shape}
	# (without full name of cells in the shape, ie only AND2X1)
	shape = {}

	# for each node find shape
	shapeSize = 4 	#the depth 
	for node in G:

		# list of nodes that have been visited, contains full cell name
		visited = []
		# list of nodes to add to shape, does not contain full cell name
		nodesToAdd = []
		result = []
		# counts nr of iterations, should never be larger than k (depth)
		counter = 0
		result, shape[node] = shape_hash_node(node, G, shapeSize, visited, nodesToAdd, counter, nodeTypes)


	# timing and printing
	endShape = timing.clock()
	# func.info_dict["elapsedShape"] = endShape - startShape
	print()
	# returns shape dictionary without full cell names
	return shape

def shape_hash_node(node, G, k, visited, nodesToAdd, counter, nodeTypes):
	"""!
	Function performs recursive DFS on node up to depth given in constants

	runtime = O(|V|+|E|)
	@param [in] node as networkx node of G
	@param [in] G as DiGraph of design under test
	@param [in] k as int of depth of shape to be calcualted
	@param [in] vistited as list of visited nodes
	@param [in] nodesToAdd as list of nodes to be added to shape
	@param [in] counter as int, once k<counter don't go any deeper
	@parem [in] nodeTypes as dictionary nodeName : nodeType

	@param [out] visited as list of visited nodes
	@param [out] nodesToAdd as list of visited nodes, as nodeTypes, not nodeName

	"""
	# if node has been visited, do not include it in shape again
	if node in visited:
		pass

	# if node is a primary input, node has now been visited, and node
	# should be added to shape
	elif not list(G.predecessors(node)):
		visited.append(node)
		# fixme: fix control wire handeling.
		#~ if node not in func.control_wires:
		nodesToAdd.append(node.split("_")[0])
		#~ else:
			#~ nodesToAdd.append(node)

	# if node is not a primary input, and has not yet been visisted,
	# and is no more than k depth from source node
	else:
		if counter <= k:
			# increase counter by one,
			counter = counter + 1
			# add node to visited and nodesType to nodesToAdd
			visited.append(node)
			nodesToAdd.append(nodeTypes[node])
			# if more than 1 child
			if len(list(G.predecessors(node))) >= 2:
				# for each child (sorted alphabetically)
				for child in sort_children(G, node, k, counter, nodeTypes):
					# call function for child node
					if child == "GND":
						visited.append(node)
						nodesToAdd.append(nodeTypes[node])
					else:
						visited, nodesToAdd = shape_hash_node(child, G, k, visited, nodesToAdd, counter, nodeTypes)
			# if only 1 child, call function on child node
			else:
				child = list(G.predecessors(node))[0]
				if child == "GND":
					visited.append(node)
					nodesToAdd.append(nodeTypes[node])
				else:
					visited, nodesToAdd = shape_hash_node(child, G, k, visited, nodesToAdd, counter, nodeTypes)

	# return update on visited and nodesToAdd
	return visited, nodesToAdd

def sort_children(G, node, k, counter, nodeTypes):
	"""!
	function which alphabetically sorts children of node. if two or more nodes of same nodetype exist, will try to sort based on lenght of all predecessors up to k-counter, and if lenght is the same, according to alphabetical split of immediate predecessors
	"""

	children = sorted(list(G.predecessors(node)))
	type_list = []
	for child in children:
		if nodeTypes[child] is not "INPUT":
			type_list.append(nodeTypes[child])
	# if there are one or more identical elements in type list, and counter is smaller than k (otherwise it doesn't really matter in which order the end pieces are added)
	if len(set(type_list)) is not len(type_list) and k > counter:

		# do the whole thing a couple of times to make sure that children are being swapped properly
		degree_dict =dict(G.in_degree(children))
		degree = max(degree_dict.values())
		while degree > 1:

			done = []

			for child1 in children:
				for child2 in children:
					# if not already done, if the children are not identical, but nodeTypes are:
					if [child1,child2] not in done:
						if  child1 != child2:
							if nodeTypes[child1] == nodeTypes[child2]:
								# if above coniditions are fullfilled, there is a chance that nodes may have to be swapped
								# find predecessors and index in sorted list of predecessors
								pred1 = [child1]
								pred2 = [child2]
								index1 = children.index(child1)
								index2 = children.index(child2)
								# find depth of search for which child has the smaller amounts of
								steps = k-counter

								# find all predecessors up to nr of steps back and compare size

								pred1_done = []
								pred2_done = []

								while steps > 0:
									for child in [x for x in pred1 if x not in pred1_done]:
										pred1.extend(list(G.predecessors(child)))
										pred1_done.append(child)
									for child in [x for x in pred2 if x not in pred2_done]:
										pred2.extend(list(G.predecessors(child)))
										pred2_done.append(child)
									steps = steps -1
								pred1 = list(set(pred1))
								pred2 = list(set(pred2))
								# shortest goes first, so if pred1 longer than pred2, indexes need to be swapped
								if len(pred2) < len(pred1):
									children[index1], children[index2] = children[index2], children[index1]
								# if both have the same lenght - problem
								elif len(pred2) == len(pred1):

									# compare immediate predecessors regarding nodetypes:
									pred1 = sorted([nodeTypes[x] for x in list(G.predecessors(child1))])
									pred2 = sorted([nodeTypes[x] for x in list(G.predecessors(child2))])
									if pred1 != pred2:
										# check which one has the different one...
										pred = pred1 + pred2
										pred = sorted(list(set(pred)))
										for element in pred:
											if element not in pred1:
												#swap
												children[index1], children[index2] = children[index2], children[index1]
												break
						done.extend([[child1,child2], [child2, child1]])
			degree = degree -1

	return children

def find_words_from_shape(shape, G):
	"""!
	Function groups nodes with similar shapes into words

	Function creates hastable of shapes of nodes in design, and groups alike nodes

	@param [in] shape as dict of nodeName:shape
	@param [out] words as list of potential words
	"""

	func.print_info("Step 1: Grouping nodes with similar shape, to identify words from shape.")

	# hashtable is dictionary of {shape: ["node1", "node2]}
	hashTable = {}
	# for each node in shape
	for node in iter(shape.keys()):
		if len(shape[node]) > 7:
			# if string of shape is not yet in hashTable
			if str(shape[node]) not in hashTable:
				# create new dictionary item {item: node}
				hashTable[str(shape[node])] = [node]
			# if already contained in hashTable
			else:
				# append node to shape key
				hashTable[str(shape[node])].append(node)

	# possible words is list of lists, grouping nodes which have
	# the same shape
	shape_words = []
	# for each shape in hashTable
	append_words = shape_words.append
	for words in hashTable.values():
		word_noIO = [x for x in words if not (x.startswith("OUTPUT") or x.startswith("INPUT"))]
		if word_noIO and len(word_noIO) > 2:
			append_words(sorted(word_noIO))

	pprint("shape words")
	pprint(shape_words)


	shape_words = word_check.check_words(shape_words, G)


	pprint("checked words")
	pprint(shape_words)

	# func.info_dict["wordsShape"] = len(shape_words)
	return shape_words


########################################################################
	# Functional word identification
########################################################################
def find_cuts(G, cell_lib, POs):
	"""!
	Finds set of Cuts for all nodes

	iterative function which finds set of cuts for each node, and writes to result dictionary
	@param [in] G as DiGraph of design under test
	@param [in] cell_lib as dictionary containing library information
	@param [in] POs as list of primary outputs

	@param [out] result as dictionary containing list of cuts {nodeName : [list of cuts (type:pyeda)]}

	"""
	# timing and prints
	startCutFind = timing.clock()
	func.print_info("Step 1: Searching for feasible cut of size " + str(func.constants["cutSize"]) + " for each node in design")

	# result is dictionary of {node: [cuts]}
	result = {}
	#~ result["INPUT_gnd"] = ["INPUT_gnd"]
	#~ result["INPUT_gnd"] = [0]
	# list of inputs and outputs for which cuts have been calculated
	doneIO = []

	# while not every node is defined
	while len(result) < len(G):
		# nodes which do not yet have a entry in result dictionary
		#~ incompleteNodes = [x for x in nx.topological_sort(G) if x not in result and x not in doneIO]
		incompleteNodes = (x for x in func.top_sort(G) if x not in result and x not in doneIO)
		pprint("done topsort")
		for node in incompleteNodes:
			# for output nodes, set of cuts is set of cuts of predecessors
			if node.startswith("OUTPUT"):
				if list(G.predecessors(node))[0] in result:
					try:
						result[node.replace("\\", "").replace(" ", "")] = result[list(G.predecessors(node))[0]]
					except:
						result[node] = result[list(G.predecessors(node))[0]]
					doneIO.append(node)
			# if node is already done, pass
			elif node in result:
				pass
			elif node.startswith("INPUT"):
				result[node.replace("\\", "").replace(" ", "")] = [node.replace("\\", "").replace(" ", "")]
				doneIO.append(node)
			elif node.startswith("DFF"):
				result[node] = [node]
			else:
			# if node is not output, not already defined, and not GND
			# find predecessors
				#~ predecessors = G.predecessors(node)

				# add trivial cut to cutset
				cutSet = [exprvar(node)]

				# get function
				function = get_function_single(node, cell_lib, POs)
				# for all different sizes of gates, process is the same:
				# 1. result must be defined for all inputs
				# 2. for each cut of result for each input, compose dictionary is written, and new cut is composed using function of node
				# 3. all new cuts are appeneded to cutset if smaller than k_cutSize
				# 4. if all inputs are defined (ie result exists) cutSet is written to result - node is now defined

				# size 2
				if len(function.inputs) == 2:

					inp1 = function.inputs[0]
					inp2 = function.inputs[1]
					if str(inp1) in result and str(inp2) in result:
						composeDict = {}
						for cut1 in result[str(inp1)]:
							for cut2 in result[str(inp2)]:
								composeDict = {inp1: cut1, inp2: cut2}
								newCut = function.compose(composeDict)
								if newCut.degree <= func.constants["cutSize"]:
									cutSet.append(newCut)
						if cutSet:
							result[node] = cutSet


				# size 1
				elif len(function.inputs) == 1:
					inp1 = function.inputs[0]

					if str(inp1) in result:
						composeDict = {}
						for cut1 in result[str(inp1)]:
							composeDict = {inp1: cut1}
							newCut = function.compose(composeDict)
							if newCut.degree <= func.constants["cutSize"]:
								cutSet.append(newCut)
						if cutSet:
							result[node] = cutSet

				# size 3
				elif len(function.inputs) == 3:
					inp1 = function.inputs[0]
					inp2 = function.inputs[1]
					inp3 = function.inputs[2]
					if str(inp1) in result and str(inp2) in result and str(inp3) in result:
						composeDict = {}
						for cut1 in result[str(inp1)]:
							for cut2 in result[str(inp2)]:
								for cut3 in result[str(inp3)]:
									composeDict = {inp1: cut1, inp2: cut2, inp3:cut3}
									newCut = function.compose(composeDict)
									if newCut.degree <= func.constants["cutSize"]:
										cutSet.append(newCut)
						if cutSet:
							result[node] = cutSet
				# size 4
				elif len(function.inputs) == 4:
					inp1 = function.inputs[0]
					inp2 = function.inputs[1]
					inp3 = function.inputs[2]
					inp4 = function.inputs[3]
					if str(inp1) in result and str(inp2) in result and str(inp3) in result and str(inp4) in result:
						for cut1 in result[str(inp1)]:
							for cut2 in result[str(inp2)]:
								for cut3 in result[str(inp3)]:
									for cut4 in result[str(inp4)]:
										newCut = function.compose({inp1: cut1, inp2: cut2, inp3:cut3, inp4:cut4})
										if newCut.degree <= func.constants["cutSize"]:
											cutSet.append(newCut)
						if cutSet:
							result[node] = cutSet
				# size 5
				elif len(function.inputs) == 5:
					inp1 = function.inputs[0]
					inp2 = function.inputs[1]
					inp3 = function.inputs[2]
					inp4 = function.inputs[3]
					inp5 = function.inputs[4]
					if str(inp1) in result and str(inp2) in result and str(inp3) in result and str(inp4) in result and str(inp5) in result:
						composeDict = {}
						for cut1 in result[str(inp1)]:
							for cut2 in result[str(inp2)]:
								for cut3 in result[str(inp3)]:
									for cut4 in result[str(inp4)]:
										for cut5 in result[str(inp5)]:
											composeDict = {inp1: cut1, inp2: cut2, inp3:cut3, inp4:cut4, inp5: cut5}
											newCut = function.compose(composeDict)
											if newCut.degree <= func.constants["cutSize"]:
												cutSet.append(newCut)
						if cutSet:
							result[node] = cutSet

				else:
					raise ValueError("too many inputs for cut")

			func.draw_progress_bar(len(result)/len(G))

	# find average amount of cuts per node, and max amount of cut
	avrgLen = 0
	maxLen = []
	for node in result:
		avrgLen = avrgLen + len(result[node])
		maxLen.append(len(result[node]))
	avrgLen = avrgLen / len(result)
	maxLen = sorted(maxLen, reverse=True)[0]

	# timing, prints and func.info_dict
	print()
	endCutFind = timing.clock()
	# func.info_dict["elapsedCutFind"] = endCutFind-startCutFind
	# func.info_dict["avrgCutLen"] = avrgLen
	# func.info_dict["maxCutLen"] = maxLen

	# return result dictionary
	return result

def find_words_from_cut(cuts, PIs, POs, G, nodeTypes, InputGrouping, words_shape):
	"""!
	Function groups nodes by analysing structural functionality

	Function finds P-representative classes from set of feasible cuts
	for each node, and groups nodes with similar p-representative
	operating on same set of inputs.

	First reads all permutation graphs required to calculate p-representative.
	Then for each cut of each node, calculated p-representaive, and inputs operated upon

	@param [in] cuts as dictionary of nodes: list of cuts
	@param [in] PIs as list of primary inputs

	@param [out] wordsCut as list of possible words
	"""
	# timing and prints
	startCutWords = timing.clock()


	# read in all permutation graphs up to size 6
	PList = [nx.read_edgelist("permutations/permEdgeList" + str(1), create_using=nx.DiGraph()), nx.read_edgelist("permutations/permEdgeList" + str(2), create_using=nx.DiGraph()), nx.read_edgelist("permutations/permEdgeList" + str(3), create_using=nx.DiGraph()), nx.read_edgelist("permutations/permEdgeList" + str(4), create_using=nx.DiGraph()), nx.read_edgelist("permutations/permEdgeList" + str(5), create_using=nx.DiGraph()), nx.read_edgelist("permutations/permEdgeList" + str(6), create_using=nx.DiGraph())]

	PVarList = {"perm1" : func.read_dict("permutations/perm1.txt"), "perm2" : func.read_dict("permutations/perm2.txt"), "perm3" : func.read_dict("permutations/perm3.txt"), "perm4" : func.read_dict("permutations/perm4.txt"), "perm5" : func.read_dict("permutations/perm5.txt"), "perm6" : func.read_dict("permutations/perm6.txt")}

	func.print_info("Grouping nodes into equivalence classes")
	pClass = {}
	counter = 0
	# write pClass dictionary, grouping cuts and nodes together
	# for each node in cuts dictionary
	for node in sorted(cuts.keys()):
		for singleCut in cuts[node][1:]:
			if len(singleCut.inputs) > 4:
				# find bistlice
				bitslice = cut_to_bitlice(G, node, singleCut, PIs, POs)
				# replace primary inputs with name INPUT[nr] i.e.
				inputs = str(singleCut.inputs)
				group_counter = 0
				for inpGroup in InputGrouping:
					for inp in inpGroup:
						inp = inp.replace("\\", "").replace(" ", "")
						#~ if inp not in func.control_wires:
						inputs = inputs.replace(inp, "INPUT" + str(group_counter))
					group_counter = group_counter + 1

				# find input independant representation of cut
				cutInputFree = str(singleCut.to_cnf())
				# convert inputs to list of strings, and replace in cut with "x" (longest first, as to not replace only part of input)
				inp_strs = [str(x) for x in singleCut.inputs]
				for inp in sorted(inp_strs, key=len, reverse=True):
					cutInputFree = cutInputFree.replace(str(inp), "x")

				function, permutation = calc_prep(singleCut, PList[len(singleCut.inputs)-1], PVarList)
				function = str(function)

				nodeDict = { "node": node,  "inputs":[str(x) for x in singleCut.inputs], "bitslice":bitslice, "perm": permutation, "cut": singleCut}
				if function not in pClass:
					pClass[function] = [nodeDict]
				elif nodeDict not in pClass[function]:
					pClass[function].append(nodeDict)

		# draw progress bar
		counter = counter+1
		func.draw_progress_bar(counter/len(cuts))

	print()

	func.print_info("Finding words from cuts")

	# find words from pRep , if more than 1 value (ie.node) for 1 given pClass,
	# group all values and declare as word
	wordsCut = []
	chainWords = []
	chainWordsList = []
	###FIMXE: control wire permutations!!!
	word_match = {}
	for function, nodes in pClass.items():
		word = [x["node"] for x in nodes if not (x["node"].startswith("OUTPUT") or x["node"].startswith("INPUT"))]
		word = sorted(list(set(word)))

		# write to wordsCut, to be checked later
		if word and len(word) > min([int(x) for x in func.constants["wordLenght"]]):
			wordsCut.append(sorted(word))

			word_match.update({str(word):function})


			# find chain words
			word_inputs = sorted([y for x in nodes for y in x["inputs"]])
			intersection = list(set(word).intersection(set(word_inputs)))

			if intersection and len(intersection)+1 == len(word):
				chainTree = nx.DiGraph()
				bitslices = [x["bitslice"] for x in nodes]
				for bitslice in bitslices:
					chainTree = nx.compose(chainTree, bitslice)

				if [word, chainTree] not in chainWords:
					chainWords.append([word, chainTree])
					if word not in chainWordsList:
						chainWordsList.append(word)


	func.print_info("Finished finding words from cuts")

	wordsCut = word_check.check_words(wordsCut, G, words_shape)




	cutInputWords = []

	print()
	endCutWords = timing.clock()
	# func.info_dict["elapsedCutWords"] = endCutWords-startCutWords
	# func.info_dict["wordsCut"] = len(wordsCut)
	# #~ func.info_dict["inpsWordCut"] = len(inpsWordCut)
	# func.info_dict["chainWords"] = len(chainWordsList)


	return wordsCut, cutInputWords, chainWordsList

def cut_to_bitlice(G, node, cut, PIs, POs):
	"""!
	Function finds subgraph / bitslice from cut of node
	@param [in] G as DiGraph of design under test
	@param [in] node as str of node in DiGraph
	@param [in] cut as pyeda function
	@param [in] POs as list of primary outputs
	@param [in] PIs as list of primary inputs

	@param [out] bitslice as DiGraph
	"""
	# if node is not a input or output
	if not node.startswith("OUTPUT") and not node.startswith("INPUT"):
		# find inputs of cut in str format
		# if more than 1 input
		if len(cut.inputs) > 1:
			inpsCut = str(cut.inputs).replace("(","").replace(")","").replace(" ", "").split(",")
		# if only 1 input
		else:
			inpsCut = [str(cut.inputs).replace("(","").replace(")","").replace(" ", "").replace(",", "")]

		# find inputs for bitslice/graph
		inpsGraph = []
		for inp in inpsCut:
			if inp + " " in PIs:
				inpsGraph.append(inp + " ")
			elif inp not in PIs:
				inpsGraph.append(inp.replace("INPUT_", "INPUT_\\").replace("]", "] "))
			else:
				inpsGraph.append(inp)
		# find bitslice using subcircuit_id function find_subgraph (using input and output word)

		bitslice = subcircuit_gen.find_subgraph(inpsGraph, [node], G)

	# if node is input or output
	else:
		if node not in POs:
			node = node.replace("OUTPUT_", "OUTPUT_\\").replace("]", "] ")
		# find inputs of each cut in str format
		if len(cut.inputs) > 1:
			inpsCut = str(cut.inputs).replace("(","").replace(")","").replace(" ", "").split(",")
		else:
			inpsCut = [str(cut.inputs).replace("(","").replace(")","").replace(" ", "").replace(",", "")]

		# find inputs to bitslice / graph
		inpsGraph = []
		for inp in inpsCut:
			if inp + " " in PIs:
				inpsGraph.append(inp + " ")
			elif inp not in PIs:
				inpsGraph.append(inp.replace("INPUT_", "INPUT_\\").replace("]", "] "))
			else:
				inpsGraph.append(inp)

		# find bitslice using subcircuit_id function find_subgraph (using input and output word)
		bitslice = subcircuit_gen.find_subgraph(inpsGraph, [node], G)

	return bitslice

def get_function_single(node, cell_lib, POs):
	"""!
	Function returns function of cell, based only on immidiate predecessors

	@param [in] node as string of name of node under test
	@param [in] cell_lib as dictionary containing library information
	@param [in] POs as list of primary outputs
	@param [out] function of node as pyeda function
	"""

	# find new node_lookup and nets_to_nodes dictionaries, to be only
	# based on immidiate predecessors
	node_lookup_new = {}
	nets_to_nodes_new = {}

	node_lookup_new[node] = myparsing.node_lookup[node]
	nets_to_nodes_new[node_lookup_new[node]["dest"][0][1]] = myparsing.nets_to_nodes[node_lookup_new[node]["dest"][0][1]]

	# get function

	function = myfunctions.get_function(node, node_lookup_new, cell_lib, nets_to_nodes_new)
	# parse to pyeda
	function = myfunctions.stringToPyeda(function)



	# swap net name to actual node name by updating compose dictionary
	# for each input, and then composing function
	composeDict = {}
	for inp in function.inputs:
		if str("OUTPUT_\\" + str(inp)[6:] + " ") in POs:
			composeDict.update({inp: "OUTPUT_" + str(inp)[6:]})
		elif str(inp)[6:] in myparsing.nets_to_nodes: # ie not primary input
			composeDict.update({inp: myparsing.nets_to_nodes[str(inp)[6:]]})
	# compose function
	function = function.compose(composeDict)
	# return function
	return function

def calc_prep(cut, P, PVarList):
	"""!
	Function calculates representative function for given cut

	Function converts truthtable outvector to smallest p-representative using permutation trees . runtime = O(|V|+|E|) reduced due to using only smallest path in search tree
	@param [in] cut as pyeda formula
	@param [in] P as DiGraph of permutation tree of size (len(cut.inputs))

	@param [out] pRep as list containg the p-representative
	"""
	# create truthtable of cut
	T = expr2truthtable(cut)
	inputs = T.inputs

	# calculate the outvector of the truth table of cut
	outvec = list()

	for i, point in enumerate(boolfunc.iter_points(inputs)):
		#~ invec = [2 if point[v] else 1 for v in inputs]
		val = T.pcdata[i]
		if val == PC_ZERO:
			outvec.append(0)
		elif val == PC_ONE:
			outvec.append(1)
		elif val == PC_DC:
			outvec.append(2)

	# search permutation tree, to find smallest pRepresentative
	coefPerm = search_permut_tree(P, [nx.topological_sort(P)[0]], outvec)

	coefPerm = coefPerm.split("_")[1:]

	# find pRep from coefficent permutation and outvector of truth table
	pRep = []
	for i in range(len(outvec)):
		pRep.append(outvec[int(coefPerm[i])])

	for x in range(len(coefPerm)):
		coefPerm[x] = int(coefPerm[x])

	permDict = PVarList["perm" + str(int(math.log(len(coefPerm))/math.log(2)))]
	permut = permDict[str(coefPerm)]

	return pRep, permut

def search_permut_tree(P, candidates, outvec):
	"""!
	Function searches permutation tree for smallest path in order to calculate p-representative

	"""

	finished = False
	while finished == False:
		sucs = []
		for candidate in candidates:
			sucs.extend(P.successors(candidate))
		sucs = sorted(sucs)

		if not sucs:
			coef = candidates
			break
		candidates = []

		for child in sucs:
			if outvec[int(child.split("_")[-1])] == 0:
				candidates.append(child)

		# if none of the values is a 0 - ie. all are equally bad
		if not candidates:
			candidates.extend(sucs)

	# if two canonical representation exist, return sorted first one
	return sorted(coef)[0]

########################################################################
	# Post-processing
########################################################################
def draw_words(G, words, chainWords,  filename):

	func.print_info("drawing words onto graph")

	colours = ["blue", "yellow", "seagreen", "crimson", "magenta", "cyan","darkorchid4", "peru",   "cadetblue1", "magenta",   "indianred2", "plum", "brown", "skyblue", "darkgreen", "darkolivegreen2", "darkslategray1",  "dodgerblue", "gold", "green", "hotpink1", "indianred2", "maroon3","plum",  "yellow", "tan1", "steelblue1", "violetred", "blueviolet", "darkgoldenrod", "darkorange", "olivedrab3", "palevioletred",  "skyblue", "yellowgreen", "seagreen",  "peru", "moccasin",  "cadetblue","burlywood1", "chartreuse3", "darksalmon", "blue", "aquamarine", "brown", "blueviolet", "cadetblue1", "chocolate", "crimson",  "darkgreen", "darkolivegreen2",  "darkslategray1", "deeppink", "dodgerblue", "gold", "green", "hotpink1",  "maroon3",  "yellow", "tan1", "steelblue1", "violetred", "blueviolet", "darkgoldenrod", "darkorange", "olivedrab3", "palevioletred",   "yellowgreen", "seagreen", "magenta", "peru", "moccasin", "lightpink", "darkslategray1","cadetblue","burlywood1", "chartreuse3", "darksalmon","blue", "aquamarine", "brown", "blueviolet", "cadetblue1", "chocolate", "crimson", "cyan", "darkgreen", "darkolivegreen2", "darkorchid4",  "deeppink", "dodgerblue", "gold", "green", "hotpink1", "cyan","indianred2", "maroon3","plum",  "yellow", "tan1", "steelblue1", "violetred", "blueviolet", "darkgoldenrod", "darkorange", "olivedrab3", "palevioletred",  "skyblue", "yellowgreen",    "moccasin", "lightpink", "cadetblue","burlywood1", "chartreuse3", "blueviolet", "darksalmon","blue",  "brown", "blueviolet", "cadetblue1", "chocolate","chocolate", "darkorchid4", "crimson", "cyan", "darkgreen", "darkolivegreen2", "darkorchid4", "darkslategray1", "deeppink", "dodgerblue", "gold", "green", "hotpink1", "indianred2", "maroon3","plum",    "aquamarine","tan1", "steelblue1", "violetred", "blueviolet", "darkgoldenrod", "darkorange", "olivedrab3", "palevioletred",  "skyblue", "yellowgreen", "seagreen", "magenta", "peru", "moccasin", "lightpink", "cadetblue","burlywood1", "chartreuse3", "darksalmon","blue", "aquamarine", "brown", "blueviolet", "cadetblue1", "chocolate", "crimson", "cyan", "darkgreen", "darkolivegreen2", "darkorchid4", "darkslategray1", "deeppink", "deeppink", "lightpink",  "dodgerblue", "gold", "green", "hotpink1", "indianred2", "maroon3","plum",  "yellow", "tan1", "steelblue1", "violetred", "blueviolet", "darkgoldenrod", "darkorange", "olivedrab3", "palevioletred",  "skyblue", "yellowgreen", "seagreen", "magenta", "peru", "moccasin", "lightpink", "cadetblue","burlywood1", "chartreuse3", "darksalmon",]



	subsets = [x for x in words for y in words if y is not x and set(x).issubset(y)]

	words_clean = []
	for word in words:
		if word not in subsets:
			words_clean.append(word)


	words_clean.append(func.control_wires)
	if len(words_clean) <= len(colours):
		H = nx.nx_pydot.to_pydot(G)
		P = nx.nx_pydot.to_pydot(G)
		counter = 0
		I = pydotplus.Cluster("Inputs")
		O = pydotplus.Cluster("Outputs")
		for word in words_clean:
			S = pydotplus.Cluster("word" + str(counter), style="filled", color=colours[counter])

			for node in word:
				S.add_node(pydotplus.Node(node))
				if node.startswith("INPUT"):
					I.add_node(pydotplus.Node(node))
				if node.startswith("OUTPUT"):
					O.add_node(pydotplus.Node(node))


				if node not in [x.get_name() for x in P.get_nodes()]:
					n = P.get_node("\"" + node + "\"")[0]
				else:
					n = P.get_node(node)[0]
				n.set_style("filled")
				n.set_color(colours[counter])

			H.add_subgraph(S)
			#~ H.add_subgraph(O)
			#~ H.add_subgraph(I)
			#~ P.add_subgraph(I)
			#~ P.add_subgraph(O)
			#@todo: fix clustering of primary inputs and outputs...
			counter = counter +1

		for word in chainWords:

			for node in word:
				if node not in [x.get_name() for x in P.get_nodes()]:
					n = P.get_node("\"" + node + "\"")[0]
				else:
					n = P.get_node(node)[0]
				#~ n.set_style("filled")
				n.set_color(colours[counter])
				counter = counter +1

		H.write_raw("wordGraphs/" + filename + ".dot")
		P.write_raw("wordGraphs/" +filename + "2.dot")

	else:
		print("Not enough colors to properly draw word Graph")
		input()

	func.print_info("Finished drawring words onto graph")
	return

### for control wire stuff

def preprocess_cuts(cuts):
	"""!
	"""
	## optimised cuts dictionary with removed control wires
	cuts_opt = {}
	for node in sorted(cuts.keys()):

		# if node is not a input or output
		if not node.startswith("INPUT"):
			# nodes without a cut are not included in the optimised cuts
			if cuts[node] == 'n.a.':
				pass
			else:
				cuts_opt[node] = []
				# for each single cut, which is not first trivial cut, or a cut of only x inputs (as this represents general functions like NAND, XNOR etc, and allows for way to many matches - these are found with structural shape hashing)
				for singleCut in cuts[node][1:]:
					#todo: make this value inputtable
					if len(singleCut.inputs) > 3:

						# find if cut contains a control wire,
						wireList = []
						for wire in func.control_wires:
							wire = exprvar(wire)
							if wire in singleCut.inputs:
								wireList.append(wire)

						# if wire list is empty, no control wires are contained within cut, cut is written to optimised cuts as is
						if not wireList:
							cuts_opt[node].append([singleCut, {}])


						# if wirelist is not empty, cut contains a control wire, several optimised cuts are written to optimised cuts dictionary
						else:
							counter = 0
							for cofactor in singleCut.cofactors(sorted(wireList)):
								if str(cofactor) != "1" and str(cofactor) != "0":

									truthTable = list(boolfunc.iter_points(wireList))
									permut = truthTable[counter]
									if [cofactor, permut] not in cuts_opt[node] and [cofactor, {}] not in cuts_opt[node]:
										cuts_opt[node].append([cofactor, permut])
								counter = counter +1




	# fixme: some cuts are still in twice after cofactoring - problem with pyeda expressions


	# cuts_opt = {nodeName : [[cut1 , {perm}], [cuts2, {perm2}]...]}
	return cuts_opt



def find_cut_words_new(cuts, PIs, POs, G, nodeTypes, InputGrouping, words_shape):

	"""!
	Function groups nodes by analysing structural functionality

	Function finds P-representative classes from set of feasible cuts
	for each node, and groups nodes with similar p-representative
	operating on same set of inputs.

	First reads all permutation graphs required to calculate p-representative.
	Then for each cut of each node, calculated p-representaive, and inputs operated upon

	@param [in] cuts as dictionary of nodes: list of cuts
	@param [in] PIs as list of primary inputs

	@param [out] wordsCut as list of possible words
	"""
	# timing and prints
	startCutWords = timing.clock()

	cuts_opt = preprocess_cuts(cuts)

	# read in all permutation graphs up to size 6
	PList = [nx.read_edgelist("permutations/permEdgeList" + str(1), create_using=nx.DiGraph()), nx.read_edgelist("permutations/permEdgeList" + str(2), create_using=nx.DiGraph()), nx.read_edgelist("permutations/permEdgeList" + str(3), create_using=nx.DiGraph()), nx.read_edgelist("permutations/permEdgeList" + str(4), create_using=nx.DiGraph()), nx.read_edgelist("permutations/permEdgeList" + str(5), create_using=nx.DiGraph()), nx.read_edgelist("permutations/permEdgeList" + str(6), create_using=nx.DiGraph())]

	PVarList = {"perm1" : func.read_dict("permutations/perm1.txt"), "perm2" : func.read_dict("permutations/perm2.txt"), "perm3" : func.read_dict("permutations/perm3.txt"), "perm4" : func.read_dict("permutations/perm4.txt"), "perm5" : func.read_dict("permutations/perm5.txt"), "perm6" : func.read_dict("permutations/perm6.txt")}

	func.print_info("Grouping nodes into equivalence classes")
	pClass = {}
	counter = 0
	# write pClass dictionary, grouping cuts and nodes together
	# for each node in cuts dictionary
	for node in sorted(cuts_opt.keys()):
		for cut in cuts_opt[node]:

			singleCut = cut[0]
			if len(singleCut.inputs) > 2:
				cw_perm = cut[1]
				# find bistlice
				bitslice = cut_to_bitlice(G, node, singleCut, PIs, POs)
				# replace primary inputs with name INPUT[nr] i.e.
				inputs = str(singleCut.inputs)
				group_counter = 0
				for inpGroup in InputGrouping:
					for inp in inpGroup:
						inp = inp.replace("\\", "").replace(" ", "")
						#~ if inp not in func.control_wires:
						inputs = inputs.replace(inp, "INPUT" + str(group_counter))
					group_counter = group_counter + 1

				# find input independant representation of cut
				cutInputFree = str(singleCut.to_cnf())
				# convert inputs to list of strings, and replace in cut with "x" (longest first, as to not replace only part of input)
				inp_strs = [str(x) for x in singleCut.inputs]
				for inp in sorted(inp_strs, key=len, reverse=True):
					cutInputFree = cutInputFree.replace(str(inp), "x")

				function, permutation = calc_prep(singleCut, PList[len(singleCut.inputs)-1], PVarList)
				function = str(function)

				nodeDict = { "node": node,  "inputs":[str(x) for x in singleCut.inputs], "bitslice":bitslice, "perm": permutation, "cut": singleCut, "cw_perm": cw_perm}
				if function not in pClass:
					pClass[function] = [nodeDict]
				elif nodeDict not in pClass[function]:
					pClass[function].append(nodeDict)

		# draw progress bar
		counter = counter+1
		func.draw_progress_bar(counter/len(cuts))

	print()

	func.print_info("Finding words from cuts")

	#### for optimised cuts:
	# find words from pRep , if more than 1 value (ie.node) for 1 given pClass,
	# group all values and declare as word
	wordsCut = []
	chainWords = []
	chainWordsList = []


	word_match = {}
	for function, nodes in pClass.items():
		wordsCut.extend(extract_words(nodes))

	pprint("words_pre_check")
	pprint(wordsCut)

	wordsCut = word_check.check_words(wordsCut, G, words_shape)
	cutInputWords = []

	print()
	endCutWords = timing.clock()
	# func.info_dict["elapsedCutWords"] = endCutWords-startCutWords
	# func.info_dict["wordsCut"] = len(wordsCut)
	# #~ func.info_dict["inpsWordCut"] = len(inpsWordCut)
	# func.info_dict["chainWords"] = len(chainWordsList)


	return wordsCut, cutInputWords, chainWordsList


def extract_words(nodes):

	words = []
	if len(nodes) > 2:
		print("###################")
		cw_free_nodes = [x for x in nodes if not x["node"].startswith("OUTPUT") and not x["cw_perm"]]
		cw_nodes = [x for x in nodes if not x["node"].startswith("OUTPUT") and x["cw_perm"]]
		print()

		### if no nodes with control wires, return word as is


		if cw_free_nodes and cw_nodes:
			pprint("both")
			input()
		if not cw_nodes:
			return set([x["node"] for x in cw_nodes])
		else:
			hashTable = {}
			for node1 in cw_nodes:
				for node2 in cw_nodes:
					check = "ok"
					perm1 = node1["cw_perm"]
					perm2 = node2["cw_perm"]
					keys1 = set(perm1.keys())
					keys2 = set(perm2.keys())
					intersection = keys1.intersection(keys2)
					if intersection:
						for permut in intersection:
							if perm1[permut] != perm2[permut]:
								check = "not ok"
								break


					hashTable[(node1["node"] + "+" + node2["node"])] = check

			G = nx.Graph()
			for k,v in hashTable.items():
				if v is "ok":
					G.add_edge(k.split("+")[0], k.split("+")[1])
			new_words = []
			cli = list(nx.find_cliques(G))
			words.extend(cli)
			if cw_free_nodes:
				for c in cli:
					words.append(c.extend(cw_free_nodes))
			return words
	return []
