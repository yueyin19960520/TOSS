
# TOSS

Welcome to the TOSS repository! This repository contains the core codebase for TOSS, including traditional methods and Graph Neural Network (GNN) implementations.

## Installation Instructions

To get started with TOSS, follow the steps below:

### 1. Set Up the Python Environment

First, create a Python environment. We recommend using Python version 3.9.0 for compatibility. You can use `venv` or `conda` to create a new environment.

```bash
# Using conda
conda create -n toss_env python=3.9.0
conda activate toss_env
```

### 2. Install Required Dependencies

After activating your environment, install the necessary dependencies listed in the `requirements.txt` file. 

```bash
pip install -r requirements.txt
```

### 3. Run the Scripts

Once the dependencies are installed, you can proceed to run the main scripts:

- `run.py`: This script handles the initial setup and execution of the traditional TOSS methods.
- `train.py`: This script is used to train the GNN models.

```bash
# Run traditional TOSS methods
python TOSS/toss/run.py

# Train GNN models
python TOSS/toss_gnn/train.py
```

## Repository Structure

- `TOSS/toss/`: Contains the traditional TOSS methods and related scripts.
- `TOSS/toss_gnn/`: Contains the code for training and evaluating Graph Neural Networks.

### 4. Use FeTiO3 as a Sample to Show Results

### Import Necessary Modules

Show results using the pre-trained model and display them in a 3D plot.

```python
import sys
sys.path.append("./toss_GNN")
from data_utils import *
from dataset_utils_pyg import *
from model_utils_pyg import *
```

### Load Pre-trained Models

Load pre-trained models for link prediction (LP) and node classification (NC).

```python
LP_model = pyg_Hetero_GCNPredictor(atom_feats=13, bond_feats=13, hidden_feats=[256,256,256,256], 
                                   predictor_hidden_feats=64, n_tasks=2, predictor_dropout=0.3)

NC_model = pyg_GCNPredictor(in_feats=15, hidden_feats=[256, 256, 256, 256], 
                            predictor_hidden_feats=64, n_tasks=12, predictor_dropout=0.3) 

LP_model.load_state_dict(torch.load("./models/pyg_Hetero_GCN_s_0608.pth"))
NC_model.load_state_dict(torch.load("./models/pyg_GCN_s_0609.pth"))
```

All keys matched successfully.

### 3D Plotting for the Result

```python
from LP_NC_Vis import vis_LP_from_cif
vis = vis_LP_from_cif("FeTiO3.cif", LP_model, NC_model)
vis.draw()
vis.show_fig()
vis.save_fig("pred_TiFeO3.html")
```

[Visit the Predicted 3D plot on our webpage.](https://www.toss.science/example/pred_TiFeO3.html)

### Import Necessary Packages

```python
import pandas as pd
import numpy as np
import sys

# Append TOSS path to system path
sys.path.append("./toss")

# Import packages from TOSS
from result import RESULT
from pre_set import PRE_SET
from Get_Initial_Guess import get_the_valid_t
from get_fos import GET_FOS
from Get_TOS import get_Oxidation_States
```

### Use FeTiO3.cif as an Example (Fe3O4 can also be used as an example with mixed valence)

```python
# Example CIF file
mid = "FeTiO3.cif"  # "Fe3O4.cif" or "Modified_Prussian_Blue.cif" can be used similarly
# If you only want to check the sample, please make a directory, and move the cif files in it.
```

### Get the Valid Tolerance List (Load and Digest the Structure)

The modules GET_STRUCTURE and DIGEST are wrapped in the function `get_the_valid_t`, which returns the valid tolerances for the given structure. In this example, only one tolerance is valid, i.e., 1.1.

```python
valid_t = get_the_valid_t(m_id=mid)
valid_t
```

This is the 0th structure with mid FeTiO3.cif, and we got 3 different valid tolerances:
[1.1, 1.12, 1.14]

### Initial Guess for the Oxidation States (OS)

Perform the initial guess for the OS and CN and display the results in a DataFrame.

```python
GFOS = GET_FOS()
res = RESULT()
GFOS.initial_guess(m_id=mid, delta_X=0.1, tolerance=1.1, tolerance_list=valid_t, res=res)
pd.DataFrame([res.elements_list, res.sum_of_valence, res.shell_CN_list], index=["Elements", "Valence", "Coordination Number"])
```

### Tune the OS Result by Maximum A Posteriori (MAP)

Perform the final result for OS and display the results in a DataFrame.

```python
RES = get_Oxidation_States(m_id=mid, input_tolerance_list=valid_t)[-1]
pd.DataFrame([RES.elements_list, RES.sum_of_valence, RES.shell_CN_list], index=["Elements", "Valence", "Coordination Number"])
```

Got the Formal Oxidation State of the 0th structure FeTiO3.cif in 1.350132942199707 seconds.

### 3D Plotting for the Result

It will show one 3D plot (no change) or two 3D plots for better visualization of the changes during the MAP process. The opacity of the spheres represents the loss values projected onto each atom.

```python
from visualization import VS
vs = VS(RES,res,loss_ratio=10)
vs.show_fig()
vs.save_fig("true_TiFeO3.html")
```

CNs are DIFFERENT! USE two figs!
[Visit the TOSS 3D plot on our webpage.](https://www.toss.science/example/true_TiFeO3.html)
