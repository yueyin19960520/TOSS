class Periodic_Table():
    def __init__(self):
        self.elements_list = ['H' ,                                                                                                 'He', 
                              'Li', 'Be',                                                             'B' , 'C' , 'N' , 'O' , 'F' , 'Ne', 
                              'Na', 'Mg',                                                             'Al', 'Si', 'P' , 'S' , 'Cl', 'Ar', 
                              'K' , 'Ca', 'Sc', 'Ti', 'V' , 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 
                              'Rb', 'Sr', 'Y' , 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I' , 'Xe', 
                              'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 
                                                'Hf', 'Ta', 'W' , 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn',
                              'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U' , 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
                                                'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']

        self.metals = ['Li', 'Be',                                                             
                       'Na', 'Mg',                                                             'Al', 
                       'K' , 'Ca', 'Sc', 'Ti', 'V' , 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 
                       'Rb', 'Sr', 'Y' , 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 
                       'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 
                                         'Hf', 'Ta', 'W' , 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 
                       'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U' , 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
                                         'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts']


        self.transition_metals = ['Sc', 'Ti', 'V' , 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
                                  'Y' , 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
                                  'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
                                        'Hf', 'Ta', 'W' , 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
                                  'Ac', 'Th', 'Pa', 'U' , 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
                                        'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn']
        
        self.nonmetals = ['H',                          'He',
                               'B' ,'C' ,'N' ,'O' ,'F' ,'Ne',
                                    'Si','P' ,'S' ,'Cl','Ar',
                                         'As','Se','Br','Kr',
                                              'Te','I' ,'Xe',
                                                        'Rn',
                                                        'Og']
        
        self.alkali = {"Li":1,"K":1,"Na":1,"Rb":1,"Cs":1,"Fr":1}
        self.earth_alkali = {"Be":2, "Mg":2,"Ca":2,"Sr":2,"Ba":2,"Ra":2}

        self.lanthanide_and_actinide = ['La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
        								'Ac', 'Th', 'Pa', 'U',  'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr']

class RESULT(Periodic_Table):
	def __init__(self):
		self.mid = None
		self.dict_ele = None
		pt = Periodic_Table()
		self.periodic_table = pt
		self.inorganic_group = None
		self.Forced_transfer_group = None
		#self.transit_metals = None
		#self.metals = None
		self.matrix_of_threshold = None

		self.matrix_of_length = None
		self.struct = None
		self.sites = None
		self.elements_list = None
		self.valence_list = None
		self.min_oxi_list = None
		self.max_oxi_list = None
		self.organic_patch = None
		self.alloy_flag = None

		self.threshold_list = None
		self.shell_idx_list = None
		self.shell_ele_list = None
		self.shell_X_list = None
		self.shell_CN_list = None
		self.shell_env_list = None
		self.SHELL_idx_list = None
		self.SHELL_idx_list_with_images = None

		self.bo_matrix = None
		self.ori_bo_matrix = None
		self.print_covalent_status = None
		self.covalent_pair = None
		self.original_min_oxi_list = None
		self.original_max_oxi_list = None

		self.inorganic_acid_flag = None
		self.inorganic_acid_center_idx = None
		self.first_algorithm_flag = None

		self.sum_of_valence = None
		self.ori_n = None
		self.ori_super_atom_idx_list = None
		self.ori_sum_of_valence = None
		self.super_atom_idx_list = None
		self.link_atom_idx_list = None
		self.single_atom_idx = None

		self.fake_n = None
		self.exclude_super_atom_list = None
		self.perfect_valence_list = None
		self.plus_n = None
		self.resonance_order = None

		self.species_uni_list = None
		self.initial_vl = None
		self.final_vl = None



def raise_error(error_name, error_info):
    #error_info = Error_class[error_name]
    class NewError(Exception):
        def __init__(self,error_info):
            super().__init__(self)
            self.errorinfo = error_info
            self.__class__.__name__ = error_name
        def __str__(self):
            return self.errorinfo
    raise NewError(error_info)
"""END HERE"""