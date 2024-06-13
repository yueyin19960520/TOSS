import os
import sys
import multiprocessing

if __name__ == "__main__":
	os.chdir("toss")
	cores = multiprocessing.cpu_count()
	os.system("python extract_structures.py -n %s"%cores)
	os.system("python Get_MOT.py -n %s"%cores)
	os.system("python Get_Initial_Guess.py -n %s"%cores)
	os.system("python Get_TOS.py -n %s"%cores)
	os.system("python BVS.py -n %s"%cores)