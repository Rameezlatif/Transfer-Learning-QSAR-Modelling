
! pip install chembl_webresource_client

"""## **Importing libraries**"""

# Import necessary libraries
import pandas as pd
from chembl_webresource_client.new_client import new_client

from google.colab import files
import pandas as pd

# Upload CSV files
uploaded = files.upload()

# List uploaded files
for filename in uploaded.keys():
    print(f'File {filename} uploaded')

# Merge CSV files
merged_data = pd.DataFrame()
for filename in uploaded.keys():
    df = pd.read_csv(filename)
    merged_data = pd.concat([merged_data, df], ignore_index=True, sort=False)

# Save merged data to CSV
merged_data.to_csv('merged_data.csv', index=False)
print('Merged data saved to merged_data.csv')

"""## **Search for Target protein**

### **Target search **
"""

# Target search

target = new_client.target
target_query = target.search('CHEMBL3038499')
targets=pd.DataFrame.from_dict(target_query)
targets

"""### **Select and retrieve bioactivity data for 

We will assign the fifth entry (which corresponds to the target protein, *Human Acetylcholinesterase*) to the ***selected_target*** variable
"""

#selected_target =targets.target_chembl_id[0]
#selected_target
selected_target= 'merged_data.csv'

selected_target

"""Here, we will retrieve only bioactivity data for target that are reported as pChEMBL values."""

#activity= new_client.activity
res= activity.filter(target_chembl_id=selected_target).filter(standard_type="IC50")

df = pd.DataFrame.from_dict(res)

df

df='merged_data_all.csv'

"""Finally we will save the resulting bioactivity data to a CSV file **bioactivity_data.csv**."""

df.to_csv('merged_data.csv', index=False)

"""## **Handling missing data**
If any compounds has missing value for the **standard_value** and **canonical_smiles** column then drop it.
"""

df2= df[df.standard_value.notna()]
df2 = df2[df.canonical_smiles.notna()]
df2

len(df2.canonical_smiles.unique())

df2_nr = df2.drop_duplicates(['canonical_smiles'])
df2_nr

"""## **Data pre-processing of the bioactivity data**

### **Combine the 3 columns (molecule_chembl_id,canonical_smiles,standard_value) and bioactivity_class into a DataFrame**
"""

selection = ['molecule_chembl_id','canonical_smiles','standard_value']
df3 = df2_nr[selection]
df3

"""Saves dataframe to CSV file"""

df3.to_csv('TAK1TAB1_CHEMBL3038499_02_bioactivity_data_preprocessed.csv', index=False)

"""### **Labeling compounds as either being active, inactive or intermediate**
The bioactivity data is in the IC50 unit. Compounds having values of less than 1000 nM will be considered to be **active** while those greater than 10,000 nM will be considered to be **inactive**. As for those values in between 1,000 and 10,000 nM will be referred to as **intermediate**.
"""

df4 = pd.read_csv('TAK1TAB1_CHEMBL3038499_02_bioactivity_data_preprocessed.csv')

bioactivity_threshold = []
for i in df4.standard_value:
  if float(i) >= 1000:
    bioactivity_threshold.append("inactive")
  elif float(i) <= 1000:
    bioactivity_threshold.append("active")
  else:
    bioactivity_threshold.append("intermediate")

bioactivity_class = pd.Series(bioactivity_threshold, name='class')
df5 = pd.concat([df4, bioactivity_class], axis=1)
df5

"""Saves dataframe to CSV file"""

df5.to_csv('TAK1TAB1_CHEMBL3038499_03_bioactivity_data_curated.csv', index=False)

! zip TAK1TAB1_CHEMBL3038499-all.zip *.csv

! ls -l

"""---"""
