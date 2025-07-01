# %% [markdown]
# # Global Health Survival Analysis
# Exploratory Data Analysis (EDA) on a fictional dataset.

# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('global_health_dataset.csv')
df.head()

# %% [markdown]
# ## Gender vs Survival

# %%
sns.countplot(data=df, x='Gender', hue='Survived')
plt.title('Gender vs Survival')
plt.show()

# %% [markdown]
# ## Age Distribution by Survival

# %%
sns.histplot(data=df, x='Age', hue='Survived', bins=8, kde=True)
plt.title('Age Distribution by Survival')
plt.show()

# %% [markdown]
# ## Country-wise Survival

# %%
sns.countplot(data=df, x='Country', hue='Survived')
plt.xticks(rotation=45)
plt.title('Country-wise Survival Count')
plt.show()

# %% [markdown]
# ## PreExistingCondition vs Survival

# %%
sns.countplot(data=df, x='PreExistingCondition', hue='Survived')
plt.title('Impact of PreExisting Conditions')
plt.show()


