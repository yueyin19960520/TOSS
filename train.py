import os
import sys

if __name__ == "__main__":
	os.chdir("toss_GNN")
	#os.system("python node_classification.py -p pyg -m gcn -s s")
	os.system("python link_prediction.py -p pyg -m gcn -s s -g Hetero")
