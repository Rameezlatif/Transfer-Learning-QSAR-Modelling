---

## **Install conda and rdkit**
"""

! wget https://repo.anaconda.com/miniconda/Miniconda3-py37_4.8.2-Linux-x86_64.sh
! chmod +x Miniconda3-py37_4.8.2-Linux-x86_64.sh
! bash ./Miniconda3-py37_4.8.2-Linux-x86_64.sh -b -f -p /usr/local
! conda install -c rdkit rdkit -y
import sys
sys.path.append('/usr/local/lib/python3.7/site-packages/')

"""## **Load bioactivity data**"""



import pandas as pd

df = pd.read_csv('merged_data_03_bioactivity_data_curated.csv')
df

df_no_smiles = df.drop(columns='canonical_smiles')

smiles = []

for i in df.canonical_smiles.tolist():
  cpd = str(i).split('.')
  cpd_longest = max(cpd, key = len)
  smiles.append(cpd_longest)

smiles = pd.Series(smiles, name = 'canonical_smiles')

df_clean_smiles = pd.concat([df_no_smiles,smiles], axis=1)
df_clean_smiles

"""## **Calculate Lipinski descriptors**
Christopher Lipinski, a scientist at Pfizer, came up with a set of rule-of-thumb for evaluating the **druglikeness** of compounds. Such druglikeness is based on the Absorption, Distribution, Metabolism and Excretion (ADME) that is also known as the pharmacokinetic profile. Lipinski analyzed all orally active FDA-approved drugs in the formulation of what is to be known as the **Rule-of-Five** or **Lipinski's Rule**.

The Lipinski's Rule stated the following:
* Molecular weight < 500 Dalton
* Octanol-water partition coefficient (LogP) < 5
* Hydrogen bond donors < 5
* Hydrogen bond acceptors < 10

### **Import libraries**
"""

!pip install rdkit

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski

"""### **Calculate descriptors**"""

# Inspired by: https://codeocean.com/explore/capsules?query=tag:data-curation

def lipinski(smiles, verbose=False):

    moldata= []
    for elem in smiles:
        mol=Chem.MolFromSmiles(elem)
        moldata.append(mol)

    baseData= np.arange(1,1)
    i=0
    for mol in moldata:

        desc_MolWt = Descriptors.MolWt(mol)
        desc_MolLogP = Descriptors.MolLogP(mol)
        desc_NumHDonors = Lipinski.NumHDonors(mol)
        desc_NumHAcceptors = Lipinski.NumHAcceptors(mol)

        row = np.array([desc_MolWt,
                        desc_MolLogP,
                        desc_NumHDonors,
                        desc_NumHAcceptors])

        if(i==0):
            baseData=row
        else:
            baseData=np.vstack([baseData, row])
        i=i+1

    columnNames=["MW","LogP","NumHDonors","NumHAcceptors"]
    descriptors = pd.DataFrame(data=baseData,columns=columnNames)

    return descriptors

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
import numpy as np
import pandas as pd

def lipinski(smiles, verbose=False):
    """
    Calculate Lipinski descriptors for a list of SMILES strings.

    Parameters:
    smiles (list): List of SMILES strings representing chemical compounds.
    verbose (bool): Whether to print verbose output.

    Returns:
    pd.DataFrame: DataFrame containing Lipinski descriptors for each compound.
    """
    moldata = []
    for elem in smiles:
        mol = Chem.MolFromSmiles(elem)
        if mol is not None:
            moldata.append(mol)
        else:
            print(f"Invalid SMILES: {elem}")

    if not moldata:
        raise ValueError("No valid molecules found in the input SMILES list.")

    baseData = np.zeros((len(moldata), 4))
    for i, mol in enumerate(moldata):
        baseData[i, 0] = Descriptors.MolWt(mol)
        baseData[i, 1] = Descriptors.MolLogP(mol)
        baseData[i, 2] = Lipinski.NumHDonors(mol)
        baseData[i, 3] = Lipinski.NumHAcceptors(mol)

    columnNames = ["MW", "LogP", "NumHDonors", "NumHAcceptors"]
    descriptors = pd.DataFrame(data=baseData, columns=columnNames)

    if verbose:
        print("Lipinski descriptors calculated successfully.")

    return descriptors



df_lipinski = lipinski(df_clean_smiles.canonical_smiles)
df_lipinski

"""### **Combine DataFrames**

Let's take a look at the 2 DataFrames that will be combined.
"""

df_lipinski

df

"""Now, let's combine the 2 DataFrame"""

df_combined = pd.concat([df,df_lipinski], axis=1)

df_combined

"""### **Convert IC50 to pIC50**
To allow **IC50** data to be more uniformly distributed, we will convert **IC50** to the negative logarithmic scale which is essentially **-log10(IC50)**.

This custom function pIC50() will accept a DataFrame as input and will:
* Take the IC50 values from the ``standard_value`` column and converts it from nM to M by multiplying the value by 10$^{-9}$
* Take the molar value and apply -log10
* Delete the ``standard_value`` column and create a new ``pIC50`` column
"""

# https://github.com/chaninlab/estrogen-receptor-alpha-qsar/blob/master/02_ER_alpha_RO5.ipynb

import numpy as np

def pIC50(input):
    pIC50 = []

    for i in input['standard_value_norm']:
        molar = i*(10**-9) # Converts nM to M
        pIC50.append(-np.log10(molar))

    input['pIC50'] = pIC50
    x = input.drop('standard_value_norm', 1)

    return x

"""Point to note: Values greater than 100,000,000 will be fixed at 100,000,000 otherwise the negative logarithmic value will become negative."""

df_combined.standard_value.describe()

-np.log10( (10**-9)* 100000000 )

-np.log10( (10**-9)* 10000000000 )

def norm_value(input):
    norm = []

    for i in input['standard_value']:
        if i > 100000000:
          i = 100000000
        norm.append(i)

    input['standard_value_norm'] = norm
    x = input.drop('standard_value', 1)

    return x

"""We will first apply the norm_value() function so that the values in the standard_value column is normalized."""

df_norm = norm_value(df_combined)
df_norm

df_norm.standard_value_norm.describe()

df_final = pIC50(df_norm)
df_final

df_final.pIC50.describe()

"""Let's write this to CSV file."""

df_final.to_csv('merged_data_04_bioactivity_data_3class_pIC50.csv')

import pandas as pd

# Load your data from a CSV file (replace 'input.csv' with your file path)
csv_file = 'merged_data_04_bioactivity_data_3class_pIC50.csv'
data = pd.read_csv(csv_file)

# Drop columns with NaN values
data = data.dropna(axis=1, how='any')

# Save the modified DataFrame to a new CSV file (replace 'output.csv' with your desired file name)
output_csv_file = 'merged_data_04_nan-bioactivity_data_3class_pIC50.csv'
data.to_csv(output_csv_file, index=False)

print("Columns with NaN values deleted. Data saved to '{output_csv_file}'.")

"""### **Removing the 'intermediate' bioactivity class**
Here, we will be removing the ``intermediate`` class from our data set.
"""

df_2class = df_final[df_final['class'] != 'intermediate']
df_2class

"""Let's write this to CSV file."""

df_2class.to_csv('merged_data_05_bioactivity_data_2class_pIC50.csv')

df_2class

"""---

## **Exploratory Data Analysis (Chemical Space Analysis) via Lipinski descriptors**

### **Import library**
"""

import seaborn as sns
sns.set(style='ticks')
import matplotlib.pyplot as plt

"""### **Frequency plot of the 2 bioactivity classes**"""

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

# Print column names to identify the correct column for frequency
print(df_2class.columns)

# Create a figure and specify the size
plt.figure(figsize=(9, 9))

# Plot a bar plot
sns.countplot(x='class', data=df_2class, edgecolor='black')

# Customize the plot
plt.xlabel('Bioactivity class', fontsize=20, fontweight='bold')
plt.ylabel('Frequency', fontsize=20, fontweight='bold')

# Adjust font properties for tick labels
mpl.rcParams['xtick.labelsize'] = 15
mpl.rcParams['ytick.labelsize'] = 15

# Save the plot in JPEG format
plt.savefig('plot_bioactivity_class_bar.jpg', dpi=600, bbox_inches='tight')

# Save the plot in PDF format
plt.savefig('plot_bioactivity_class_bar.pdf', bbox_inches='tight')

# Show the plot
plt.show()

"""### **Scatter plot of MW versus LogP**

It can be seen that the 2 bioactivity classes are spanning similar chemical spaces as evident by the scatter plot of MW vs LogP.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib as mpl

# Set font properties globally
mpl.rcParams['font.size'] = 14
mpl.rcParams['font.weight'] = 'bold'
mpl.rcParams['axes.labelsize'] = 20  # Adjust the size of x-label and y-label

# Assuming df_2class is your DataFrame

# Check for NaN or infinite values in 'pIC50'
if not np.isfinite(df_2class['pIC50']).all():
    print("DataFrame contains non-finite values in 'pIC50'.")
    # Handle or remove problematic values as needed
    df_2class = df_2class[np.isfinite(df_2class['pIC50'])]

# Create a 2x2 subplot grid for multiple plots
fig, axes = plt.subplots(2, 2, figsize=(15, 15))

# Plot 1: Scatter plot
sns.scatterplot(x='MW', y='LogP', data=df_2class, hue='class', size='pIC50', edgecolor='black', alpha=0.7, ax=axes[0, 0])
axes[0, 0].set_title('Scatter Plot')
axes[0, 0].set_xlabel('MW')
axes[0, 0].set_ylabel('LogP')
axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)

# Plot 2: Distribution of 'pIC50'
sns.histplot(df_2class['pIC50'], kde=True, color='skyblue', ax=axes[0, 1])
axes[0, 1].set_title('Distribution of pIC50')
axes[0, 1].set_xlabel('pIC50')
axes[0, 1].set_ylabel('Frequency')

# Plot 3: Box plot of 'pIC50' by class
sns.boxplot(x='class', y='pIC50', data=df_2class, ax=axes[1, 0])
axes[1, 0].set_title('Box Plot of pIC50 by Class')
axes[1, 0].set_xlabel('Class')
axes[1, 0].set_ylabel('pIC50')

# Plot 4: Violin plot of 'pIC50' by class
sns.violinplot(x='class', y='pIC50', data=df_2class, ax=axes[1, 1])
axes[1, 1].set_title('Violin Plot of pIC50 by Class')
axes[1, 1].set_xlabel('Class')
axes[1, 1].set_ylabel('pIC50')

# Adjust layout for better spacing
plt.tight_layout()

# Save the figure in PDF format
plt.savefig('multiple_plots.pdf')

# Save the figure in JPEG format
plt.savefig('multiple_plots.jpg', dpi=600)

# Show the plot
plt.show()

"""### **Box plots**

**Statistical analysis | Mann-Whitney U Test**
"""

def mannwhitney(descriptor, verbose=False):
  # https://machinelearningmastery.com/nonparametric-statistical-significance-tests-in-python/
  from numpy.random import seed
  from numpy.random import randn
  from scipy.stats import mannwhitneyu

# seed the random number generator
  seed(1)

# actives and inactives
  selection = [descriptor, 'class']
  df = df_2class[selection]
  active = df[df['class'] == 'active']
  active = active[descriptor]

  selection = [descriptor, 'class']
  df = df_2class[selection]
  inactive = df[df['class'] == 'inactive']
  inactive = inactive[descriptor]

# compare samples
  stat, p = mannwhitneyu(active, inactive)
  #print('Statistics=%.3f, p=%.3f' % (stat, p))

# interpret
  alpha = 0.05
  if p > alpha:
    interpretation = 'Same distribution (fail to reject H0)'
  else:
    interpretation = 'Different distribution (reject H0)'

  results = pd.DataFrame({'Descriptor':descriptor,
                          'Statistics':stat,
                          'p':p,
                          'alpha':alpha,
                          'Interpretation':interpretation}, index=[0])
  filename = 'mannwhitneyu_' + descriptor + '.csv'
  results.to_csv(filename)

  return results

save= mannwhitney('pIC50')
save
save.to_csv('Mann-Whitney U Test_pIC50.csv')

"""#### **MW**"""

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

# Set font properties globally
mpl.rcParams['font.size'] = 14
mpl.rcParams['font.weight'] = 'bold'

# Create a figure and specify the size
plt.figure(figsize=(10.5, 10.5))

# Plot a boxplot
sns.boxplot(x='class', y='MW', data=df_2class)

# Customize the plot
plt.xlabel('Bioactivity class', fontsize=20, fontweight='bold')
plt.ylabel('MW', fontsize=20, fontweight='bold')

# Adjust font properties for tick labels
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14

# Save the plot in PDF format with DPI
plt.savefig('plot_MW.pdf', dpi=600)

# Save the plot in JPEG format with DPI
plt.savefig('plot_MW.jpg', dpi=600)

# Show the plot
plt.show()

save_2= mannwhitney('MW')
save_2.to_csv('Mann-Whitney U Test_pIC50_MW.csv')
save_2

save_2= mannwhitney('pIC50')
save
save.to_csv('Mann-Whitney U Test_MW.csv')

"""#### **LogP**"""

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

# Set font properties globally
mpl.rcParams['font.size'] = 14
mpl.rcParams['font.weight'] = 'bold'

# Create a figure and specify the size
plt.figure(figsize=(10.5, 10.5))

# Plot a boxplot
sns.boxplot(x='class', y='LogP', data=df_2class)

# Customize the plot
plt.xlabel('Bioactivity class', fontsize=20, fontweight='bold')
plt.ylabel('LogP', fontsize=20, fontweight='bold')

# Adjust font properties for tick labels
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14

# Save the plot in PDF format with DPI
plt.savefig('plot_LogP.pdf', dpi=600)

# Save the plot in JPEG format with DPI
plt.savefig('plot_LogP.jpg', dpi=600)


# Show the plot
plt.show()

"""**Statistical analysis | Mann-Whitney U Test**"""

save_3= mannwhitney('LogP')
save_3.to_csv('Mann-Whitney U Test_logP.csv')
save_3

save_3= mannwhitney('pIC50')
save
save.to_csv('Mann-Whitney U Test_LogP.csv')

"""#### **NumHDonors**"""

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

# Set font properties globally
mpl.rcParams['font.size'] = 14
mpl.rcParams['font.weight'] = 'bold'

# Create a figure and specify the size
plt.figure(figsize=(10.5, 10.5))

# Plot a boxplot
sns.boxplot(x='class', y='NumHDonors', data=df_2class)

# Customize the plot
plt.xlabel('Bioactivity class', fontsize=20, fontweight='bold')
plt.ylabel('HBD', fontsize=20, fontweight='bold')

# Adjust font properties for tick labels
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14

# Save the plot in PDF format with DPI
plt.savefig('plot_NumHDonors.pdf', dpi=600)

# Save the plot in JPEG format with DPI
plt.savefig('plot_NumHDonors.jpg', dpi=600)


# Show the plot
plt.show()

"""**Statistical analysis | Mann-Whitney U Test**"""

save_3= mannwhitney('NumHDonors')
save_3.to_csv('Mann-Whitney U Test_NumHDoners.csv')
save_3

save_3= mannwhitney('pIC50')
save
save.to_csv('Mann-Whitney U Test_NumHDonors.csv')

"""#### **NumHAcceptors**"""

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

# Set font properties globally
mpl.rcParams['font.size'] = 14
mpl.rcParams['font.weight'] = 'bold'

# Create a figure and specify the size
plt.figure(figsize=(10.5, 10.5))

# Plot a boxplot
sns.boxplot(x='class', y='NumHAcceptors', data=df_2class)

# Customize the plot
plt.xlabel('Bioactivity class', fontsize=20, fontweight='bold')
plt.ylabel('HBA', fontsize=20, fontweight='bold')

# Adjust font properties for tick labels
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14

# Save the plot in PDF format with DPI
plt.savefig('plot_NumHAcceptors.pdf', dpi=600)

# Save the plot in JPEG format with DPI
plt.savefig('plot_NumHAcceptors.jpg', dpi=600)

# Show the plot
plt.show()

save_4= mannwhitney('NumHAcceptors')
save_4.to_csv('Mann-Whitney U Test_NumHAcceptors.csv')
save_4

"""#### **Interpretation of Statistical Results**

##### **Box Plots**

###### **pIC50 values**

Taking a look at pIC50 values, the **actives** and **inactives** displayed ***statistically significant difference***, which is to be expected since threshold values (``IC50 < 1,000 nM = Actives while IC50 > 10,000 nM = Inactives``, corresponding to ``pIC50 > 6 = Actives and pIC50 < 5 = Inactives``) were used to define actives and inactives.

###### **Lipinski's descriptors**

All of the 4 Lipinski's descriptors exhibited ***statistically significant difference*** between the **actives** and **inactives**.

## **Zip files**
"""

! zip -r results.zip . -i *.csv *.pdf

