import time
import csv
import json

st = time.time()

def CombineData():
    with open("./dataset/ogbl_collab/raw/edge.csv", "r") as f_edge, \
        open("./dataset/ogbl_collab/raw/edge_weight.csv", "r") as f_weight, \
        open("./dataset/ogbl_collab/raw/edge_year.csv", "r") as f_year, \
        open("./dataset/ogbl_collab/raw/num-edge-list.csv", "r") as f_len, \
        open("./dataset/ogbl_collab/processed/edges.csv", "w", newline="") as f_out:

        edge_reader = csv.reader(f_edge)
        weight_reader = csv.reader(f_weight)
        year_reader = csv.reader(f_year)

        edges = int(next(csv.reader(f_len))[0])
        edge_writer = csv.writer(f_out)

        for _ in range(edges):
            edge = next(edge_reader)
            new_entry = [
                edge[0],
                edge[1],
                next(weight_reader)[0],
                next(year_reader)[0]
            ]
            #print(new_entry)
            edge_writer.writerow(new_entry)


def CombineData2():
    with open("edges_-2017.json", "r") as f, \
        open("edges_2018.json", "r") as f2, \
        open("edges_2019.json", "r") as f3:
            data1 = json.load(f)
            data2 = json.load(f2)
            data3 = json.load(f3)

    combined = {}

    for data in [data1, data2, data3]:
        for year, edges in data.items():
             
            # Create year key if missing
            if year not in combined:
                combined[year] = []

            # Append edges
            combined[year].extend(edges)

    # Save merged file
    with open("combined.json", "w") as f:
        json.dump(combined, f, indent=4)
    
    print("Merged successfully.")

#CombineData()
CombineData2()

et = time.time() - st
print(f"Done in {et} seconds")