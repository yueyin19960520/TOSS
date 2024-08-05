### Import Necessary Modules

Show result by the pre-trained model, and show it in 3D plot.


```python
import sys
sys.path.append("D:/share/TOSS/toss_GNN")
from data_utils import *
from dataset_utils_pyg import *
from model_utils_pyg import *
```

### Load Pre-trained Models

Loads pre-trained models for link prediction (LP) and node classification (NC).


```python
LP_model = pyg_Hetero_GCNPredictor(atom_feats=13, bond_feats=13, hidden_feats=[256,256,256,256], 
                                   predictor_hidden_feats=64, n_tasks=2,predictor_dropout=0.3)

NC_model = pyg_GCNPredictor(in_feats=15, hidden_feats=[256, 256, 256, 256], 
                            predictor_hidden_feats=64, n_tasks=12, predictor_dropout=0.3) 

LP_model.load_state_dict(torch.load("../models/pyg_Hetero_GCN_s_0608.pth"))
NC_model.load_state_dict(torch.load("../models/pyg_GCN_s_0609.pth"))
```
All keys matched successfully



### 3D Plotting for the Result


```python
from LP_NC_Vis import vis_LP_from_cif
vis = vis_LP_from_cif("FeTiO3.cif", LP_model, NC_model)
vis.draw()
vis.show_fig()
vis.save_fig("pred_TiFeO3.html")
```
[Visit the Predicted 3D plot at our webpage.](https://www.toss.science/example/pred_TiFeO3.html)



### Import Necessary Packages


```python
import pandas as pd
import numpy as np
import sys

# Append TOSS path to system path
sys.path.append("D:/share/TOSS/toss")

# Import package from TOSS
from result import RESULT
from pre_set import PRE_SET
from Get_Initial_Guess import get_the_valid_t
from get_fos import GET_FOS
from Get_TOS import get_Oxidation_States
```

### Use FeTiO3.cif  as Example (Fe3O4 can be am example with mixed-valence)


```python
# Example CIF file
mid = "FeTiO3.cif"#"TiFeO3.cif"#"Fe3O4.cif"  # "Modified_Prussian_Blue.cif" can be used similarly
```

### Get the valid tolerance list (Load and Digest the Structure)

The module GET_STRUCTURE and DIGEST are wrapped in the function get_the_valid_t, which returns the valid tolerances for the given structure. In this example, only one tolerance is valid, i.e., 1.1.


```python
valid_t = get_the_valid_t(m_id=mid)
valid_t
```

This is the 0th structure with mid FeTiO3.cif and we got 3 different valid tolerance(s).
[1.1, 1.12, 1.14]



### Initial Guess for the Oxidation States (OS)

Perform the initial guess for the OS and CN and display the results in a DataFrame.


```python
GFOS = GET_FOS()
res = RESULT()
GFOS.initial_guess(m_id=mid, delta_X=0.1, tolerance=1.1, tolerance_list=valid_t, res=res)
pd.DataFrame([res.elements_list, res.sum_of_valence, res.shell_CN_list], index=["Elements", "Valence", "Coordination Number"])
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
      <th>12</th>
      <th>13</th>
      <th>14</th>
      <th>15</th>
      <th>16</th>
      <th>17</th>
      <th>18</th>
      <th>19</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Elements</th>
      <td>Ti</td>
      <td>Ti</td>
      <td>Ti</td>
      <td>Ti</td>
      <td>Fe</td>
      <td>Fe</td>
      <td>Fe</td>
      <td>Fe</td>
      <td>O</td>
      <td>O</td>
      <td>O</td>
      <td>O</td>
      <td>O</td>
      <td>O</td>
      <td>O</td>
      <td>O</td>
      <td>O</td>
      <td>O</td>
      <td>O</td>
      <td>O</td>
    </tr>
    <tr>
      <th>Valence</th>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>-2</td>
      <td>-2</td>
      <td>-2</td>
      <td>-2</td>
      <td>-2</td>
      <td>-2</td>
      <td>-2</td>
      <td>-2</td>
      <td>-2</td>
      <td>-2</td>
      <td>-2</td>
      <td>-2</td>
    </tr>
    <tr>
      <th>Coordination Number</th>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>3</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>



### Tune the OS Result by Maximum A Posteriori (MAP)

Perform the final result for OS and display the results in a DataFrame.


```python
RES = get_Oxidation_States(m_id=mid, input_tolerance_list=valid_t)[-1]
pd.DataFrame([RES.elements_list, RES.sum_of_valence, RES.shell_CN_list], index=["Elements", "Valence", "Coordination Number"])
```

Got the Formal Oxidation State of the 0th structure FeTiO3.cif in 1.350132942199707 seconds.



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
      <th>12</th>
      <th>13</th>
      <th>14</th>
      <th>15</th>
      <th>16</th>
      <th>17</th>
      <th>18</th>
      <th>19</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Elements</th>
      <td>Ti</td>
      <td>Ti</td>
      <td>Ti</td>
      <td>Ti</td>
      <td>Fe</td>
      <td>Fe</td>
      <td>Fe</td>
      <td>Fe</td>
      <td>O</td>
      <td>O</td>
      <td>O</td>
      <td>O</td>
      <td>O</td>
      <td>O</td>
      <td>O</td>
      <td>O</td>
      <td>O</td>
      <td>O</td>
      <td>O</td>
      <td>O</td>
    </tr>
    <tr>
      <th>Valence</th>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>-2</td>
      <td>-2</td>
      <td>-2</td>
      <td>-2</td>
      <td>-2</td>
      <td>-2</td>
      <td>-2</td>
      <td>-2</td>
      <td>-2</td>
      <td>-2</td>
      <td>-2</td>
      <td>-2</td>
    </tr>
    <tr>
      <th>Coordination Number</th>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>



### 3D Ploting for the result

It will show one 3d plot (no change) or two 3d plots for better performing the change between the MAP process.Opaucity spehere is the loss values projected to each atom.


```python
from visualization import VS
vs = VS(RES,res,loss_ratio=10)
vs.show_fig()
vs.save_fig("true_TiFeO3.html")
```

CNs are DIFFERENT! USE two figs !
[Visit the TOSS 3D plot at our webpage.](https://www.toss.science/example/true_TiFeO3.html)