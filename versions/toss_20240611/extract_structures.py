import re
import os
import multiprocessing
import argparse


def extract_structure(line, path):
    mid = re.findall(r'@Y@(.*)@Y@',line)[0]
    with open(path+mid, "a+") as f1:
        f1.write(re.sub("YUEYIN","\n", line))
    return None

if __name__ == "__main__":
    path = os.path.split(os.path.abspath(os.path.dirname(__file__)))[0]
    structure_path = path + "/structures/"
    os.mkdir(structure_path)

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--ncore', type=int, required=True, help='Number of Parallel')
    args = parser.parse_args()
    n = args.ncore

    with open(path+"/all.cif") as f:
        str_structures = f.readlines()

    pool = multiprocessing.Pool(n)

    for str_structure in str_structures:
        pool.apply_async(extract_structure, args=(str_structure, structure_path,))
    pool.close()
    pool.join()
    