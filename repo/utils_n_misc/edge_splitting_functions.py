import csv
import time
from collections import defaultdict
import json

st = time.time()

def ExistedUnweighted():
    edge_dict = defaultdict(list)
    new_edge_dict = {}

    with open("./dataset/ogbl_collab/processed/edges.csv", "r") as f:
        reader = csv.reader(f)

        for row in reader:
            edge_dict[row[0]].append(row[1])

        for key in edge_dict.keys():
            edges = []

            for value in edge_dict[key]:
                if value not in edges:
                    edges.append(value)
            new_edge_dict[key] = edges

    with open("./dataset/ogbl_collab/processed/edges_existed_unweighted.json", "w") as edges_out:
        json.dump(new_edge_dict, edges_out, indent=4)


def ExistedSumWeight():
    edge_dict = defaultdict(lambda: defaultdict(int))

    with open("./dataset/ogbl_collab/processed/edges.csv", "r") as f:
        reader = csv.reader(f)

        for src, dst, weight, year in reader:
            edge_dict[src][dst] += int(weight)

    # convert to desired format
    result = {
        src: [[dst, weight] for dst, weight in dsts.items()]
        for src, dsts in edge_dict.items()
    }

    with open("./dataset/ogbl_collab/processed/edges_existed_sumweight.json", "w") as edges_out:
        json.dump(result, edges_out, indent=4)
  
    
def ExistedAvgWeight():
    edge_dict = defaultdict(lambda: defaultdict(lambda: [0, 0]))

    with open("./dataset/ogbl_collab/processed/edges.csv", "r") as f:
        reader = csv.reader(f)

        for src, dst, weight, year in reader:
            weight = int(weight)
            edge_dict[src][dst][0] += weight  # sum
            edge_dict[src][dst][1] += 1       # count
            
    result = {
        src: [
            [dst, total / count]   # average
            for dst, (total, count) in dsts.items()
        ]
        for src, dsts in edge_dict.items()
    }

    with open("./dataset/ogbl_collab/processed/edges_existed_avgweight.json", "w") as edges_out:
        json.dump(result, edges_out, indent=4)


def SplitUnweighted():
    edge_dict = defaultdict(list)

    with open("./dataset/ogbl_collab/processed/edges.csv", "r") as f:
        reader = csv.reader(f)

        for src, dst, weight, year in reader:
            edge_dict[year].append([src, dst])

    result = dict(edge_dict)

    with open("./dataset/ogbl_collab/processed/edges_split_unweighted.json", "w") as edges_out:
        json.dump(result, edges_out, indent=4)


def SplitWeighted():
    edge_dict = defaultdict(list)

    with open("./dataset/ogbl_collab/processed/edges.csv", "r") as f:
        reader = csv.reader(f)

        for src, dst, weight, year in reader:
            edge_dict[year].append([src, dst, int(weight)])

    result = dict(edge_dict)

    with open("./dataset/ogbl_collab/processed/edges_split_weighted.json", "w") as edges_out:
        json.dump(result, edges_out, indent=4)


def main(function):
    if function == "EU":
        ExistedUnweighted()
    elif function == "ES":
        ExistedSumWeight()
    elif function == "EA":
        ExistedAvgWeight()
    elif function == "SU":
        SplitUnweighted()
    elif function == "SW":
        SplitWeighted()
    else:
        print("Invalid")

#main("SU")

et = time.time() - st
print(f"Done in {et} seconds")