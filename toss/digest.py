import copy 
from pre_set import CounterSubset
from itertools import combinations, permutations


def get_ele_from_sites(i,res):
    ele = res.sites[i].specie.name
    return ele


def get_propoteries_of_atom(e,vl,res):
    ele = get_ele_from_sites(e,res)
    r = res.dict_ele[ele]["covalent_radius"]
    X = res.dict_ele[ele]["X"]
    min_oxi = res.min_oxi_list[e]
    max_oxi = res.max_oxi_list[e]   
    v = vl[e]
    return ele, r, X, min_oxi, max_oxi, v


def prior_threshold(ele_c, ele_n,res):
    #get the self-consistent threshold value of the elements pair.
    atomic_number_c = res.periodic_table.elements_list.index(ele_c)
    atomic_number_n = res.periodic_table.elements_list.index(ele_n)
    threshold = res.matrix_of_threshold[atomic_number_c, atomic_number_n]
    return threshold


def whether_organic_structure(res):

    """Generally, this program will count the numbers of organic elements,
       If the proportion of organic atoms exceed boundary, the structure will be considered as a organic one.
       organic_density = 0.5 and organic_elements = C,H work as default settings. """

    organic_density = 0.5
    organic_elements = ["C","H"]

    count_CH = 0
    if CounterSubset(res.elements_list, organic_elements):

        for ele in res.elements_list:
            count_CH += 1 if ele in organic_elements else 0

        if count_CH/len(res.elements_list) > organic_density:
            organic_patch = True

        else:
            organic_patch = False

    else:
        organic_patch = False

    return organic_patch


def find_crystal_water(tolerance,res):  

    """the undirectional ligand will effect the coordination environment
       expecially the H2O molecule. (NH3 could be considered later!)"""

    excluded_H_atom = []
    crystal_water_O = []
    for i in res.idx:
        if get_ele_from_sites(i,res) in ["C", "N", "O"]: #if the NH3 will be considered, the ["O"] can be changed to ["O","N"]

            list_of_length = res.matrix_of_length[i]
            temp_exclude_list = [] if round(min(list_of_length), 5) == round(list_of_length[i], 5) else [i]

            visa_ele = get_ele_from_sites(sorted([(l,e) for l, e in zip(list_of_length,res.idx) if e not in temp_exclude_list],
                                                 key = lambda x:x[0])[0][1],res)

            for j in res.idx:
                ele_c = get_ele_from_sites(i,res)
                ele_n = get_ele_from_sites(j,res)
                if ele_n != visa_ele:
                    double_check = prior_threshold(ele_c, ele_n,res)
                    if list_of_length[j] >= double_check:
                        temp_exclude_list.append(j)

            fine_list_of_length = []
            for j,l in enumerate(list_of_length):
                if j not in temp_exclude_list:
                    fine_list_of_length.append(l)  # BUT this include the length between the center and "H"

            #only the length of the real bond can be added in the min!!!!!
            threshold = tolerance * min(fine_list_of_length) if len(fine_list_of_length) != 0 else 0 

            #check all the length, get rid off those longger than 5 Angtrom.
            temp_list_of_idx = [i for i,l in enumerate(list_of_length) if l <= min(threshold,5)]  
            
            list_of_idx = list(set(temp_list_of_idx).difference(set(temp_exclude_list)))

            list_of_ele = [get_ele_from_sites(i,res) for i in list_of_idx]
            if list_of_ele.count("H") >= 1: #if =1. it is the hydroxy.
                for j in list_of_idx:
                    if get_ele_from_sites(j,res) == "H":
                        excluded_H_atom.append(j) 
                        crystal_water_O.append(i)
    return excluded_H_atom, crystal_water_O


#There are two hyperparameters, one is the amplified for the threshold (1.2), another one is the for H is (1.8).
def bonded_radius(i, organic_patch, tolerance, excluded_H_atom, crystal_water_O, offset,res):
    list_of_length = res.matrix_of_length[i]
    ele_c = get_ele_from_sites(i,res)

    #exclude the center atom itself unless there is no other atoms in the sphere with radius 5 A.
    #maybe we should not exclude the atom itself.
    temp_exclude_list = [] #if round(min(list_of_length), 5) == round(list_of_length[i], 5) else [i]
    Exclude_idx = copy.deepcopy(temp_exclude_list)

    #excluede the H atoms unless the center atom is Oxygen, and the list can be appended like "N","B".
    #it must be consistent with atom list in the find crystall water.
    Exclude_idx += excluded_H_atom if get_ele_from_sites(i,res) not in ["C", "N", "O"] else []
    Exclude_idx += crystal_water_O if get_ele_from_sites(i,res) not in ["H"] else []

    #the nearest atom cannot be excluded no matter what happened!
    ele_with_distance = [(get_ele_from_sites(j,res),l) for j,l in zip(res.idx, list_of_length) if j not in Exclude_idx]
    sorted_ele_with_distance = sorted(ele_with_distance, key = lambda x:x[1])
    min_length = sorted_ele_with_distance[0][1]
    refined_sorted_ele_with_distance = [j_l for j_l in sorted_ele_with_distance if j_l[0] != ele_c and j_l[1] <= tolerance * min_length]

    visa_ele = sorted_ele_with_distance[0][0] if refined_sorted_ele_with_distance == [] else refined_sorted_ele_with_distance[0][0]

    for j in res.idx:  
        ele_n = get_ele_from_sites(j,res)
        if ele_n != visa_ele:
            double_check = prior_threshold(ele_c, ele_n,res)
            if list_of_length[j] >= double_check:
                Exclude_idx.append(j)

    #Exclude the atoms whose distance to center atom longer than the threshold.
    fine_list_of_length = []
    for j,l in enumerate(list_of_length):
        #if j not in temp_exclude_list:
        if j not in Exclude_idx:
            fine_list_of_length.append(l)  # BUT this include the length between the center and "H"

      
    #the uniqueness of carbon in organic structure.              
    #if organic_patch and ele_c == "C":
        #tolerance = 1.2 * tolerance
    #else:
        #tolerance = tolerance

    for i in range(offset):
        fine_list_of_length.remove(min(fine_list_of_length))

    #only the length of the real bond can be added in the min!!!!!
    threshold = tolerance * min(fine_list_of_length) if len(fine_list_of_length) != 0 else 0 


    if len(fine_list_of_length) > 1:
        ratio_list = [i/min(fine_list_of_length) for i in fine_list_of_length]
        ratio_list.remove(1)
        min_ratio = 10 if ratio_list == [] else min(ratio_list)
    else:
        min_ratio = 10 #10 is nonsense, just from out of the loop.

    temp_list_of_idx = [i for i,l in enumerate(list_of_length) if l <= min(threshold, 8)]
    #list_of_idx = list(set(temp_list_of_idx).difference(set(temp_exclude_list)))    
    list_of_idx = list(set(temp_list_of_idx).difference(set(Exclude_idx)))   
    list_of_ele = [get_ele_from_sites(i,res) for i in list_of_idx]

    #check whether the coordinate atoms are only hydrogen, which will cause the atomic sphere too small.
    if list(set(list_of_ele)) == ["H"] and len(list_of_ele) < 5: 
        #tolerance += 0.3  //  rolerance *= 1.5
        threshold = (tolerance * 1.5) * min(fine_list_of_length) if len(fine_list_of_length) != 0 else 0 
        #only the length of the real bond can be added in the min!!!!!
        new_temp_list_of_idx = [i for i,l in enumerate(list_of_length) if l <= min(threshold, 8)]
        #list_of_idx = list(set(new_temp_list_of_idx).difference(set(temp_exclude_list)))
        list_of_idx = list(set(new_temp_list_of_idx).difference(set(Exclude_idx)))
    else:
        None

    return list_of_idx, min_ratio, threshold, Exclude_idx


def check_double_connectivity(res, SHELL_idx_list):
    #The forces are reciprocal, so as chemical bond.
    ori_SHELL_idx_list = copy.deepcopy(SHELL_idx_list)
    for i in res.idx:
        for j in SHELL_idx_list[i]:
            if i not in SHELL_idx_list[j]:
                SHELL_idx_list[j].append(i)

    return SHELL_idx_list, ori_SHELL_idx_list


def check_double_connectivity_imaged(res, SHELL_idx_list_with_images):
    #The forces are reciprocal, so as chemical bond.
    #this function considers the images of atoms.
    for i in res.idx:
        for j in list(SHELL_idx_list_with_images[i].keys()):
            ij_images = SHELL_idx_list_with_images[i][j]
            for ij_image in ij_images:
                ji_image = [-s for s in ij_image]
                if ji_image not in SHELL_idx_list_with_images[j][i]:
                    SHELL_idx_list_with_images[j][i].append(ji_image)
    return SHELL_idx_list_with_images


def get_the_bg_of_atom(i,res):
    #only for acclerating the program.
    certain_distance_list = res.matrix_of_length[i]
    backgroud_atom_idx_list = [j for j in res.idx if 0 < certain_distance_list[j] <= 5]
    return backgroud_atom_idx_list


def break_fake_H_bond(fake_H_bond,res):
    for b in fake_H_bond:
        i = b[0]
        j = b[1]
        if i in SHELL_idx_list[j] and j in SHELL_idx_list[i]:
            SHELL_idx_list[i].remove(j)
            SHELL_idx_list[j].remove(i)
    return SHELL_idx_list


def get_image(i,j,res,threshold_list):
    """In this program, not only the real atom will be considered as the coordination atom,
       but also all the possible images of the coordinated atoms can be considered."""
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
        
    IMAGE = []
    for image in image_list:
        if round(res.sites[i].distance(res.sites[j],jimage = image),4) <= round(threshold_list[i], 4):
            """only if the distance from the image to the center less than the threshold, 
               the image will be added, no matter if the image has the same distance with the real one."""
            IMAGE.append(image)

    J_coords = copy.deepcopy(IMAGE) #Ex:[0,1,1]
    j_coords = [res.sites[j].a, res.sites[j].b, res.sites[j].c] #get the fractional coordinates.

    for image in J_coords:
        for c in [0,1,2]:
            image[c] += j_coords[c] 
            
    #Convert all the images' coordinates from fractional to cartesional.
    J_coords = [res.struct.lattice.get_cartesian_coords(image) for image in J_coords]
                    
    return IMAGE,J_coords


def apply_image(i, SHELL_idx_list, res, threshold_list, ori_SHELL_idx_list): #ori_SHELL_idx_list determine the choose of the threshold
    temp_list = copy.deepcopy(SHELL_idx_list)
    SHELL_idx_list_with_images = []

    for i in res.idx:
        sub_temp_list = temp_list[i]
        sub_list_of_idx = {}
        sub_idx_list_imaged = []

        for j in sub_temp_list:
            if j not in ori_SHELL_idx_list[i]:
                #while means the threshold of smaller one should be changed to larger one.
                temp_threshold_list = copy.deepcopy(threshold_list)
                temp_threshold_list[i] = temp_threshold_list[j]
                IMAGE,J_coords = get_image(i,j,res,temp_threshold_list)
            else:
                IMAGE,J_coords = get_image(i,j,res,threshold_list)

            #only one atom in the whole structure.
            if j == i:
                IMAGE.remove([0,0,0])

            sub_list_of_idx[j] = []
            for image in IMAGE:
                sub_list_of_idx[j].append(image)
                sub_idx_list_imaged.append(j)

        SHELL_idx_list_with_images.append(sub_list_of_idx)

    return SHELL_idx_list_with_images


def benchmark(operate_idx, organic_patch, tolerance, excluded_H_atom, crystal_water_O, offset, res, SHELL_idx_list,threshold_list):

    min_ratio_list = [0 for i in range(len(res.idx))]
    for i in operate_idx:
        list_of_idx, min_ratio, threshold, Exclude_idx = bonded_radius(i,organic_patch,tolerance,excluded_H_atom,crystal_water_O,offset,res)
        SHELL_idx_list[i] = list_of_idx
        min_ratio_list[i] = min_ratio
        threshold_list[i] = threshold

    #keep reciprocal, apply image fucntion, can keep the reciprocal for images!
    SHELL_idx_list, ori_SHELL_idx_list = check_double_connectivity(res,SHELL_idx_list)
    SHELL_idx_list_with_images = apply_image(i, SHELL_idx_list, res, threshold_list, ori_SHELL_idx_list)
    SHELL_idx_list_with_images = check_double_connectivity_imaged(res, SHELL_idx_list_with_images)

    SHELL_idx_list_imaged = []
    for sub_dict in SHELL_idx_list_with_images:
        sub_list = []
        for k,image_list in sub_dict.items():
            for image in image_list:
                sub_list.append(k) 
        SHELL_idx_list_imaged.append(sub_list)

    #actually, in the periodic structure, the coordination number equals to one is rare!
    temp_operate_idx = []
    for i in operate_idx:
        if len(SHELL_idx_list_imaged[i]) == 1 and min_ratio_list[i] < tolerance + 0.1: #get_ele_from_sites(i,res) not in ["H","F","Br","Cl","I"]:
            temp_operate_idx.append(i)

    #digest the structures like the uranxyl (O=U=O). Hint from Xiaokun Zhao.
    for i in operate_idx:
        if res.elements_list[i] in res.periodic_table.lanthanide_and_actinide:
            if offset == 0:
                if len(SHELL_idx_list_imaged[i]) == 2:
                    temp_operate_idx.append(i)
            else:
                if len(SHELL_idx_list_imaged[i]) == 2 and min_ratio_list[i] < tolerance + 0.1:
                    temp_operate_idx.append(i)

    return temp_operate_idx, SHELL_idx_list_imaged,SHELL_idx_list_with_images


def erase_the_second_neighbors(benchmark_idx_imaged, new_idx_imaged, res):
    for i in res.idx:
        benchmark_list = []
        for k,v in benchmark_idx_imaged[i].items():
            for image in v:
                benchmark_list.append((k,tuple(image)))

        new_list = []
        for k,v in new_idx_imaged[i].items():
            for image in v:
                new_list.append((k,tuple(image)))

        added_list = list(set(new_list).difference(set(benchmark_list))) #type:(idx, [0,0,0])

        deleted = []

        for added_one in added_list:
            if added_one not in deleted:
                j = added_one[0]
                ij_image = list(added_one[1])
                break_flag = False

                for ori_one in benchmark_list:
                    k = ori_one[0]
                    ik_image = list(ori_one[1])

                    if j in benchmark_idx_imaged[k]:
                        subtense = [n-m for n,m in zip(ik_image, ij_image)]
                        ele_triple = [get_ele_from_sites(ijk, res) for ijk in (i,j,k)]

                        uni_flag = True if len(set(ele_triple)) == 1 else False
                        H_flag = True if "H" in ele_triple else False

                        if subtense in benchmark_idx_imaged[j][k] and not (uni_flag or H_flag):
                            #until now, it is undoubtful, the j is the second neighbor of i, bridged by k.
                            new_idx_imaged[i][j].remove(ij_image)
                            new_idx_imaged[j][i].remove([-s for s in ij_image])
                            deleted.append((j, tuple([-s for s in ij_image])))

                            break
                            break_flag = True
    return new_idx_imaged

                        
def contrast(operate_idx, organic_patch, tolerance, excluded_H_atom, crystal_water_O, offset, res, SHELL_idx_list,threshold_list,
             benchmark_SHELL_idx_list_with_images, Exclude_list): #new added parameters contrast with the benchmark.

    min_ratio_list = [0 for i in range(len(res.idx))]
    for i in operate_idx:
        list_of_idx, min_ratio, threshold, Exclude_idx = bonded_radius(i,
                                                                       organic_patch,
                                                                       tolerance,
                                                                       excluded_H_atom,
                                                                       crystal_water_O,
                                                                       offset,res)
        SHELL_idx_list[i] = list_of_idx
        min_ratio_list[i] = min_ratio
        threshold_list[i] = threshold
        Exclude_list[i] = Exclude_idx

    SHELL_idx_list, ori_SHELL_idx_list = check_double_connectivity(res,SHELL_idx_list)
    SHELL_idx_list_with_images = apply_image(i, SHELL_idx_list, res, threshold_list, ori_SHELL_idx_list)
    SHELL_idx_list_with_images = check_double_connectivity_imaged(res, SHELL_idx_list_with_images)

    #this this the only difference with the function body constast from the benchmark.
    SHELL_idx_list_with_images = erase_the_second_neighbors(benchmark_SHELL_idx_list_with_images,
                                                            SHELL_idx_list_with_images, 
                                                            res)

    SHELL_idx_list_imaged = []
    for sub_dict in SHELL_idx_list_with_images:
        sub_list = []
        for k,image_list in sub_dict.items():
            for image in image_list:
                sub_list.append(k) 
        SHELL_idx_list_imaged.append(sub_list)

    temp_operate_idx = []
    for i in operate_idx:
        if len(SHELL_idx_list_imaged[i]) == 1 and min_ratio_list[i] < tolerance + 0.1: #get_ele_from_sites(i,res) not in ["H","F","Br","Cl","I"]:
            temp_operate_idx.append(i)

    for i in operate_idx:
        if res.elements_list[i] in res.periodic_table.lanthanide_and_actinide:
            if offset == 0:
                if len(SHELL_idx_list_imaged[i]) == 2:
                    temp_operate_idx.append(i)
            else:
                if len(SHELL_idx_list_imaged[i]) == 2 and min_ratio_list[i] < tolerance + 0.1:
                    temp_operate_idx.append(i)

    return temp_operate_idx, SHELL_idx_list_imaged,SHELL_idx_list_with_images,Exclude_list


class DIGEST():

    def __init__(self, valid_t, tolerance, m_id, res):

        self.alloy_flag = True if CounterSubset(res.periodic_table.metals, list(set(res.elements_list))) else False
        [res.dict_ele[ele].update({"X": res.dict_ele[ele]["X"] - 0.3}) for ele in ["Tc", "Os", "Pt"] if not self.alloy_flag]

        self.organic_patch = whether_organic_structure(res)
        self.max_oxi_list = [res.dict_ele[get_ele_from_sites(a,res)]["max_oxi"] for a in res.idx]
        self.min_oxi_list = [res.dict_ele[get_ele_from_sites(a,res)]["min_oxi"] for a in res.idx]

        tolerance_list = copy.deepcopy(valid_t)
        benchmark_tolerance = 1.1

        ################################################1.1 work as the benchmark########################################
        excluded_H_atom,crystal_water_O = find_crystal_water(benchmark_tolerance,res)
        SHELL_idx_list = [[] for eml in range(len(res.idx))]
        operate_idx = res.idx #the first run of the benchmark is always all the atoms.
        offset = 0

        self.threshold_list = [0 for i in range(len(res.idx))]
        Exclude_list = [[] for eml in range(len(res.idx))]
        while len(operate_idx) != 0:
            operate_idx, SHELL_idx_list,SHELL_idx_list_with_images = benchmark(operate_idx, 
                                                                               self.organic_patch, 
                                                                               benchmark_tolerance, 
                                                                               excluded_H_atom,
                                                                               crystal_water_O, 
                                                                               offset, 
                                                                               res, 
                                                                               SHELL_idx_list, 
                                                                               self.threshold_list)
            offset += 1
            #print("453")

        benchmark_SHELL_idx_list_with_images = copy.deepcopy(SHELL_idx_list_with_images)
        #################################################################################################################

        #do it again with the real tolerance, let's check the difference!
        while True:
            try: 
                fake_tolerance = tolerance_list.pop(0)
                if fake_tolerance == 1.1:
                    continue  #actually,jump the tolerance = 1.1 for saving time.
                    #jump to the position of the while True, as same as refresh the fake_tolerance.

                #######################################loop the tolerance list###########################################
                excluded_H_atom,crystal_water_O = find_crystal_water(fake_tolerance,res)
                SHELL_idx_list = [[] for eml in range(len(res.idx))]
                operate_idx = res.idx
                offset = 0

                self.threshold_list = [0 for i in range(len(res.idx))]
                Exclude_list = [[] for eml in range(len(res.idx))]
                while len(operate_idx) != 0:
                    operate_idx, SHELL_idx_list,SHELL_idx_list_with_images,Exclude_list = contrast(operate_idx, 
                                                                                                   self.organic_patch, 
                                                                                                   fake_tolerance, 
                                                                                                   excluded_H_atom, 
                                                                                                   crystal_water_O,
                                                                                                   offset, 
                                                                                                   res, 
                                                                                                   SHELL_idx_list, 
                                                                                                   self.threshold_list,
                                                                                                   benchmark_SHELL_idx_list_with_images,
                                                                                                   Exclude_list)
                    offset += 1
                    #print("487")
                benchmark_SHELL_idx_list_with_images = copy.deepcopy(SHELL_idx_list_with_images)
                ########################################################################################################
            except:
                break
            #print("492")
        ##################################################real tolerance works##########################################
        excluded_H_atom, crystal_water_O = find_crystal_water(tolerance,res)
        SHELL_idx_list = [[] for eml in range(len(res.idx))]
        operate_idx = res.idx
        offset = 0

        self.threshold_list = [0 for i in range(len(res.idx))]
        Exclude_list = [[] for eml in range(len(res.idx))]
        while len(operate_idx) != 0:
            operate_idx, SHELL_idx_list,SHELL_idx_list_with_images,Exclude_list = contrast(operate_idx, 
                                                                            self.organic_patch, 
                                                                            tolerance, 
                                                                            excluded_H_atom, 
                                                                            crystal_water_O,
                                                                            offset, 
                                                                            res, 
                                                                            SHELL_idx_list, 
                                                                            self.threshold_list,
                                                                            benchmark_SHELL_idx_list_with_images,
                                                                            Exclude_list)
            offset += 1
            #print("514")

        SHELL_idx_list, SHELL_idx_list_with_images = self.excluede_coordinates(SHELL_idx_list, 
                                                                               SHELL_idx_list_with_images, 
                                                                               Exclude_list,
                                                                               excluded_H_atom, 
                                                                               crystal_water_O,
                                                                               res)
        self.SHELL_idx_list = SHELL_idx_list #actually it with images.
        self.SHELL_idx_list_with_images = SHELL_idx_list_with_images

        ################################################################################################################
        return None


    def digest_structure_with_image(self,res):
        self.shell_idx_list = []
        self.shell_ele_list = []
        self.shell_X_list = []
        self.shell_CN_list = []
        self.shell_env_list = []

        for list_of_idx in res.SHELL_idx_list:
            self.shell_idx_list.append(list_of_idx)
            self.shell_ele_list.append([get_ele_from_sites(s,res) for s in list_of_idx])
            self.shell_X_list.append([res.dict_ele[get_ele_from_sites(s,res)]["X"] for s in list_of_idx])
            self.shell_CN_list.append(len(list_of_idx))
            self.shell_env_list.append(sum([res.dict_ele[get_ele_from_sites(s,res)]["X"] for s in list_of_idx]))
        self.shell_env_list = [round(env,2) for env in self.shell_env_list]


    def uniformity(self,vl,res):
        ele_list = [get_ele_from_sites(i,res) for i in res.idx]
        species_list = [(ele,env,v) for ele,env,v in zip(ele_list, res.shell_env_list, vl)]
        species_uni_list = list(set(species_list))
        return species_uni_list


    def excluede_coordinates(self, SHELL_idx_list, SHELL_idx_list_with_images, Exclude_list, excluded_H_atom, crystal_water_O, res):
        break_pair = []
        water_atoms = excluded_H_atom + crystal_water_O

        for i in res.idx:
            if i not in crystal_water_O:
                exclude_atom_list = Exclude_list[i]
                exclude_atom_list = set(exclude_atom_list).difference(set(exclude_atom_list).intersection(set(crystal_water_O)))
                for j in exclude_atom_list:
                    if j in SHELL_idx_list[i]:
                        break_pair.append((i,j))
                        if j in excluded_H_atom:
                            break_pair.append((j,i))
            else:
                exclude_atom_list = Exclude_list[i]
                for j in exclude_atom_list:
                    if j in SHELL_idx_list[i]:
                        break_pair.append((i,j))
                        break_pair.append((j,i))


        for bp in set(break_pair):
            i = bp[0]
            j = bp[1]
            if (j,i) in break_pair:
                while i in SHELL_idx_list[j]:
                    SHELL_idx_list[j].remove(i)
                while j in SHELL_idx_list[i]:
                    SHELL_idx_list[i].remove(j)
            if i in SHELL_idx_list_with_images[j]:
                del(SHELL_idx_list_with_images[j][i])
            if j in SHELL_idx_list_with_images[i]:
                del(SHELL_idx_list_with_images[i][j])
        return SHELL_idx_list, SHELL_idx_list_with_images
"""END HERE"""

"""
def analyze_bridges(bridge_list, res):
    None


def erase_the_brigde(SHELL_idx_list_with_images):
    ready_break = []
    for i, coord_info in enumerate(SHELL_idx_list_with_images):
        for j, images in coord_info.items():
            for image in images:
                shared_idx = set(SHELL_idx_list_with_images[i].keys()).intersection(set(SHELL_idx_list_with_images[j].keys()))
                if len(shared_idx) > 1:
                    shared_atoms = 0
                    for k in shared_idx:
                        for shared_candi in SHELL_idx_list_with_images[i][k]:
                            shared_candi_imaged = [shared_candi[s] - image[s] for s in range(3)]
                            if shared_candi_imaged in SHELL_idx_list_with_images[j][k]:
                                shared_atoms += 1
                    if shared_atoms > 1:
                        ready_break.append((i,j,image))
    for brk in ready_break:
        SHELL_idx_list_with_images[brk[0]][brk[1]].remove(brk[2])
    return SHELL_idx_list_with_images
"""