### Import Necessary Packages


```python
import pandas as pd
import numpy as np
import sys

# Append TOSS path to system path
sys.path.append("D:/share/TOSS/toss")
sys.path.append("D:/share/TOSS/toss_VIS")

# Import package from TOSS
from result import RESULT
from pre_set import PRE_SET
from Get_Initial_Guess import get_the_valid_t
from get_fos import GET_FOS
from Get_TOS import get_Oxidation_States
```

### Use Fe3O4 with Mixed-Valences as Example


```python
# Example CIF file
mid = "TiFeO3.cif"#"TiFeO3.cif"#"Fe3O4.cif"  # "Modified_Prussian_Blue.cif" can be used similarly
```

### Get the valid tolerance list (Load and Digest the Structure)

The module GET_STRUCTURE and DIGEST are wrapped in the function get_the_valid_t, which returns the valid tolerances for the given structure. In this example, only one tolerance is valid, i.e., 1.1.


```python
valid_t = get_the_valid_t(m_id=mid)
valid_t
```

This is the 0th structure with mid TiFeO3.cif and we got 3 different valid tolerance(s).
valid_t = [1.1, 1.12, 1.14]


### Initial Guess for the Oxidation States (OS)

Perform the initial guess for the OS and CN and display the results in a DataFrame.


```python
GFOS = GET_FOS()
res = RESULT()
GFOS.initial_guess(m_id=mid, delta_X=0.1, tolerance=1.14, tolerance_list=valid_t, res=res)
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Elements</th>
      <td>Ti</td>
      <td>Ti</td>
      <td>Fe</td>
      <td>Fe</td>
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



### Tune the OS Result by Maximum A Posteriori (MAP)

Perform the final result for OS and display the results in a DataFrame.


```python
RES = get_Oxidation_States(m_id=mid, input_tolerance_list=valid_t)[-1]
pd.DataFrame([RES.elements_list, RES.sum_of_valence, RES.shell_CN_list], index=["Elements", "Valence", "Coordination Number"])
```

    Got the Formal Oxidation State of the 0th structure TiFeO3.cif in 0.9564187526702881 seconds.
    2024-08-02 09:26:18
    




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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Elements</th>
      <td>Ti</td>
      <td>Ti</td>
      <td>Fe</td>
      <td>Fe</td>
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
      <td>2</td>
      <td>2</td>
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
```

CNs are SAME! OSs are DIFFERENT! USE two figs !
    