import os
import pandas as pd
import numpy as np

folder_name = "unmodified"
FOLDER_PATH = "./{}/".format(folder_name)

feature_average ={
"average_neighbour_degree": [],
"betweenness_centrality": [],
"closeness_centrality": [],
"clustering": [],
"degree": [],
"degree_centrality": [],
"has_feedback_path": [],
"katz": [],
"load_centrality": [],
"outdegree": [],
"pagerank": []
}
if folder_name == "euclidean" or folder_name == "fastRELIC": 
    feature_average["similarity__score"] = []

if folder_name == "mixed":
    feature_average["euclidean_score"] = []
    feature_average["fastRELIC_score"] = []

for folder in os.listdir(FOLDER_PATH):
    folder = FOLDER_PATH + folder + "/"
    mean_files = [f for f in os.listdir(folder) if 'StateReg' in f]
    print(folder)
    for mf in mean_files:
        print(mf)
        df = pd.read_csv(folder+mf)
        df.insert(0, "feature_names", feature_average, True)
        for index, row in df.iterrows():
            print(df.iloc[0,2],df.iloc[index,1])
            ratio = df.iloc[0,2]/df.iloc[index,1]

            feature_average[row["feature_names"]].append(ratio)


for key in feature_average:
    feature_average[key] = np.average(np.array(feature_average[key]))

if folder_name == "euclidean" or folder_name == "fastRELIC":
    feature_average[folder_name]=feature_average["similarity__score"]
    del feature_average["similarity__score"]

feature_average = pd.DataFrame.from_dict(feature_average, orient='index')
feature_average.to_csv("./{}_ratio_mean.csv".format(folder_name))
             
        
