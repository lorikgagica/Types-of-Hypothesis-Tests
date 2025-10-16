# Ex 3: Conduct ANOVA
from scipy.stats import f_oneway

# Date for groups
group1 = [10, 12, 14, 16, 18]
group2 = [9, 11, 13, 15, 17]
group3 = [8, 10, 12, 14, 16]

# Perform ANOVA
f_stat, p_value = f_oneway(group1, group2, group3)
print("F-Statistic:", f_stat)
print("P-Value:", p_value)


# Additional Practice:
# A. Perform a two-way ANOVA to test for interactions effects
""" import pandas as pd
import numpy as np
from statsmodels.formula.api import ols
import statsmodels.api as sm

# Simulated example data: 'score', 'gender', 'class' (make sure these are the column names in your DataFrame)
np.random.seed(0)
df = pd.DataFrame({
    'score': np.random.normal(10, 2, 20),
    'gender': ['M']*10 + ['F']*10,
    'class': ['A', 'A', 'B', 'B', 'A', 'A', 'B', 'B', 'A', 'A']*2
})

model = ols('score ~ C(gender) + C(class) + C(gender):C(class)', data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table) """


# B. Use real-world datasets (e.g., studen scores by gender and class) for hypothesis testing
import seaborn as sns
from scipy.stats import f_oneway

iris = sns.load_dataset("iris")
setosa = iris[iris.species == "setosa"]["petal_length"]
versicolor = iris[iris.species == "versicolor"]["petal_length"]
virginica = iris[iris.species == "virginica"]["petal_length"]

""" f_stat, p_value = f_oneway(setosa, versicolor, virginica)
print("F-Statistic:", f_stat)
print("P-Value:", p_value) """



# C. Visualize test results using boxplots or bar plots.
import matplotlib.pyplot as plt

sns.boxplot(x="species", y="petal_length", data=iris)
plt.title("Petal Length Distribution by Species (Iris Dataset)")
plt.xlabel("Species")
plt.ylabel("Petal Length")
plt.show()


