
## **1. Import libraries**
"""

! pip install lazypredict

import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
import lazypredict
from lazypredict.Supervised import LazyRegressor

"""## **2. Load the data set**

"""

! wget https://github.com/dataprofessor/data/raw/master/acetylcholinesterase_06_bioactivity_data_3class_pIC50_pubchem_fp.csv

df = pd.read_csv('Merged_data_06_bioactivity_data_3class_pIC50_pubchem_fp.csv')

X = df.drop('pIC50', axis=1)
Y = df.pIC50

"""## **3. Data pre-processing**"""

# Examine X dimension
X.shape

# Remove low variance features
from sklearn.feature_selection import VarianceThreshold
selection = VarianceThreshold(threshold=(.8 * (1 - .8)))
X = selection.fit_transform(X)
X.shape

# Perform data splitting using 80/20 ratio
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

"""## **4. Compare ML algorithms**"""

# Defines and builds the lazyclassifier
clf = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
models, predictions = clf.fit(X_train, X_test, Y_train, Y_test)

models

predictions

# Performance table of the training set (80% subset)
save_1= models
save_1.to_csv('Performance of table training set(80%).csv')
save_1

# Performance table of the test set (20% subset)
save_2= predictions
save_2.to_csv('Performance of table test set(20%).csv')
save_2

"""## **5. Data visualization of model performance**"""

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

# Set font properties globally
mpl.rcParams['font.size'] = 14
mpl.rcParams['font.weight'] = 'bold'

# Assuming 'models' is a DataFrame with columns 'Model' and 'R-Squared'
# You may need to adapt the column names based on your actual DataFrame structure

# Sort the models based on R-Squared values for better visualization
models.sort_values(by='R-Squared', inplace=True, ascending=False)

plt.figure(figsize=(10, 8))
sns.set_theme(style="whitegrid")
ax = sns.barplot(x="R-Squared", y="Model", data=models, palette="viridis")

# Add labels and title
ax.set(xlim=(0, 1), xlabel='R-Squared', ylabel='Model', title='Model Performance')
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Adjust font properties for axis labels
ax.set_xlabel('R-Squared', fontsize=14, fontweight='bold')
ax.set_ylabel('Model', fontsize=14, fontweight='bold')

# Display the R-Squared values on the bars
for p in ax.patches:
    ax.annotate(f'{p.get_width():.2f}', (p.get_width() + 0.01, p.get_y() + p.get_height() / 2),
                ha='left', va='center', fontsize=10, color='black')

plt.tight_layout()

# Save the plot
plt.savefig('model_performance_bar_plot.png', dpi=600)
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

# Set font properties globally
mpl.rcParams['font.size'] = 14
mpl.rcParams['font.weight'] = 'bold'

# Assuming 'models' is a DataFrame with columns 'Model' and 'RMSE'
# You may need to adapt the column names based on your actual DataFrame structure

# Sort the models based on RMSE values for better visualization
models.sort_values(by='RMSE', inplace=True, ascending=False)

plt.figure(figsize=(10, 8))
sns.set_theme(style="whitegrid")
ax = sns.barplot(x="RMSE", y="Model", data=models, palette="plasma")

# Add labels and title
ax.set(xlim=(0, 10), xlabel='RMSE', ylabel='Model', title='Root Mean Squared Error (RMSE)')
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Adjust font properties for axis labels
ax.set_xlabel('RMSE', fontsize=14, fontweight='bold')
ax.set_ylabel('Model', fontsize=14, fontweight='bold')

# Display the RMSE values on the bars
for p in ax.patches:
    ax.annotate(f'{p.get_width():.2f}', (p.get_width() + 0.1, p.get_y() + p.get_height() / 2),
                ha='left', va='center', fontsize=10, color='black')

plt.tight_layout()

# Save the plot
plt.savefig('rmse_bar_plot.png', dpi=600)
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

# Set font properties globally
mpl.rcParams['font.size'] = 14
mpl.rcParams['font.weight'] = 'bold'

# Assuming 'models' is a DataFrame with columns 'Model' and 'Time Taken'
# You may need to adapt the column names based on your actual DataFrame structure

# Sort the models based on Time Taken for better visualization
models.sort_values(by='Time Taken', inplace=True, ascending=False)

plt.figure(figsize=(10, 8))
sns.set_theme(style="whitegrid")
ax = sns.barplot(x="Time Taken", y="Model", data=models, palette="magma")

# Add labels and title
ax.set(xlim=(0, 10), xlabel='Time Taken (seconds)', ylabel='Model', title='Calculation Time')
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Adjust font properties for axis labels
ax.set_xlabel('Time Taken (seconds)', fontsize=14, fontweight='bold')
ax.set_ylabel('Model', fontsize=14, fontweight='bold')

# Display the Time Taken values on the bars
for p in ax.patches:
    ax.annotate(f'{p.get_width():.2f}s', (p.get_width() + 0.1, p.get_y() + p.get_height() / 2),
                ha='left', va='center', fontsize=10, color='black')

plt.tight_layout()

# Save the plot
plt.savefig('calculation_time_bar_plot.png', dpi=600)
plt.show()
