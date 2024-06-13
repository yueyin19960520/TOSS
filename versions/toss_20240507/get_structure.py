from pymatgen.core.structure import IStructure
from pymatgen.ext.matproj import MPRester
import re
import os

path = os.path.split(os.path.abspath(os.path.dirname(__file__)))[0]

def self_image(sites, i):
    image_list = []
    a = -1
    while a in [-1,0,1]:
        b = -1
        while b in [-1,0,1]:
            c = -1
            while c in [-1,0,1]:
                image_list.append([a,b,c])
                c += 1
            b += 1
        a += 1

    image_list.remove([0,0,0])
    self_distance = sorted([sites[i].distance(sites[i],jimage=image) for image in image_list])[0]            
    return self_distance


class GET_STRUCTURE():
    """You will got:
        8 variables: struct, sites, idx, matrix_of_length, elements_list, valence_list
                     min_oxi_list, max_oxi_list
        1 function: get_ele_from_sites()
    """

    def __init__(self,m_id, specific_path=None):
        try:
            if specific_path == None:
                file = path + "/structures/" + str(m_id)
                self.struct = IStructure.from_file(file)
            else:
                file = specific_path + str(m_id)
                self.struct = IStructure.from_file(file)
        except:
            raise NameError("Cannot find the structure! Check the name of the structure.")

        #self.struct = supercell(self.struct.as_dict(), direction = ["X","Y","Z","-X","-Y","-Z"])

        self.sites = self.struct.sites
        self.idx = [i for i in range(len(self.sites))]
        self.matrix_of_length = self.struct.distance_matrix 
        for i in self.idx:
            self.matrix_of_length[i][i] = self_image(self.sites, i)

        self.valence_list = []
        for i in self.idx:
            self.valence_list.append(0)

        self.elements_list = []
        for i in self.idx:
            ele = self.sites[i].specie.name
            self.elements_list.append(ele)
"""END HERE"""