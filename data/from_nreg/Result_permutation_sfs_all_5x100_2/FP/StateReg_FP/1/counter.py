import os
import pandas as pd

folder_name = "unmodified"
FOLDER_PATH = "./{}/".format(folder_name)

feature_names ={
"average_neighbour_degree": 0,
"betweenness_centrality": 0,
"closeness_centrality": 0,
"clustering": 0,
"degree": 0,
"degree_centrality": 0,
"has_feedback_path": 0,
"katz": 0,
"load_centrality": 0,
"outdegree": 0,
"pagerank": 0
}
if folder_name == "euclidean" or folder_name == "fastRELIC": 
    feature_names["similarity__score"] = 0

if folder_name == "mixed":
    feature_names["euclidean_score"] = 0
    feature_names["fastRELIC_score"] = 0


for folder in os.listdir(FOLDER_PATH):
    folder = FOLDER_PATH + folder + "/"
    print(folder)

    # feature_names ={
    # "average_neighbour_degree": 0,
    # "betweenness_centrality": 0,
    # "closeness_centrality": 0,
    # "clustering": 0,
    # "degree": 0,
    # "degree_centrality": 0,
    # "has_feedback_path": 0,
    # "katz": 0,
    # "load_centrality": 0,
    # "outdegree": 0,
    # "pagerank": 0
    # }
    # if folder_name == "euclidean" or folder_name == "fastRELIC": 
    #     feature_names["similarity__score"] = 0

    # if folder_name == "mixed":
    #     feature_names["euclidean_score"] = 0
    #     feature_names["fastRELIC_score"] = 0

    mean_files = [f for f in os.listdir(folder) if 'feature_names_permuted' in f]
    
    
    for mf in mean_files:
        if os.path.getsize(folder+mf)>1:
            df = pd.read_csv(folder+mf)
            for index, row in df.iterrows():
                feature_names[row[0]] += 1
    print(feature_names)

for key in feature_names:
    feature_names[key] = feature_names[key]/65

if folder_name == "euclidean" or folder_name == "fastRELIC":
    feature_names[folder_name]=feature_names["similarity__score"]
    del feature_names["similarity__score"]

print(feature_names)
feature_names = pd.DataFrame.from_dict(feature_names, orient='index')
print (feature_names)
feature_names.to_csv("./{}_counter.csv".format(folder_name))
             
        
