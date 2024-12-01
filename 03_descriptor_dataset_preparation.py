
! wget https://github.com/dataprofessor/bioinformatics/raw/master/padel.zip
! wget https://github.com/dataprofessor/bioinformatics/raw/master/padel.sh

! unzip padel.zip

import pandas as pd

df3 = pd.read_csv('merged_data_04_bioactivity_data_3class_pIC50.csv')

df3

selection = ['canonical_smiles','molecule_chembl_id']
df3_selection = df3[selection]
df3_selection.to_csv('molecule.smi', sep='\t', index=False, header=False)

! cat molecule.smi | head -5

! cat molecule.smi | wc -l

"""## **Calculate fingerprint descriptors**

### **Calculate PaDEL descriptors**
"""

! cat padel.sh

! bash padel.sh

! ls -l

"""## **Preparing the X and Y Data Matrices**

### **X data matrix**
"""

df3_X = pd.read_csv('descriptors_output.csv')

df3_X

df3_X = df3_X.drop(columns=['Name'])
df3_X

"""## **Y variable**

### **Convert IC50 to pIC50**
"""

df3_Y = df3['pIC50']
df3_Y

"""## **Combining X and Y variable**"""

dataset3 = pd.concat([df3_X,df3_Y], axis=1)
dataset3

dataset3.to_csv('Merged_data_all_06_bioactivity_data_3class_pIC50_pubchem_fp.csv', index=False)

"""# **Let's download the CSV file to your local computer for the Part 3B (Model Building).**"""
