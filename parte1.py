import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

""" # Se lee el .data
df = pd.read_csv('BCW.data', header=None)

#se agregan los titulos de las columnas
df.columns = ['id', 'clump_thickness', 'cell_size_uniformity', 'cell_shape_uniformity',
              'marginal_adhesion', 'single_epithelial_cell_size', 'bare_nuclei',
              'bland_chromatin', 'normal_nucleoli', 'mitoses', 'class']

# si hay un caracter que no es numero, pasarlo a cero
df = df.apply(pd.to_numeric, errors='coerce').fillna(0)

#Cambiar tipo de las columnas
df['id'] = df['id'].astype('object')
df['class'] = df['class'].astype('int')
df[df.columns[1:-1]] = df[df.columns[1:-1]].astype('int')

# Nuevo csv
df.to_csv('cleaned_file.csv', index=False) """



#Exploracion de datos

data = pd.read_csv('cleaned_file.csv')

print("Data shape")
print(data.shape, "\n")



print("Data basic statistics")
print(data.describe(), "\n")

print(data.iloc[:10, :4])




#Tipos de datos y aislamiento
""" data = pd.read_csv('cleaned_file.csv')

print("Data types")
print(data.dtypes, "\n")

# Aislar las variables numéricas y categóricas
numeric_vars = df.select_dtypes(include=['int', 'float'])
categorical_vars = df.select_dtypes(include=['object'])

# Calcular la matriz de correlación entre las variables numéricas
correlation_matrix = numeric_vars.corr()

# Visualizar la matriz de correlación
plt.matshow(correlation_matrix)
plt.xticks(range(len(numeric_vars.columns)), numeric_vars.columns, rotation=90)
plt.yticks(range(len(numeric_vars.columns)), numeric_vars.columns)
plt.colorbar()
plt.show() """


#Histogramas
"""plt.hist(data['Cell size'],edgecolor="white")
plt.title('Histogram of Cell size')
plt.xlabel('Cell size')
plt.ylabel('Frequency')
plt.show() """


""" plt.hist(data['Thickness'],edgecolor="white")
plt.title('Histogram of thickness')
plt.xlabel('Thickness')
plt.ylabel('Frequency')
plt.show() """



#Diagrama de caja y bigote
""" x = data['Adhesion']
y = data['Cell size']

sns.boxplot(x=x, y=y)
plt.show()
 """

#Heatmap
""" corr = data.corr()

sns.heatmap(corr, annot=True)
plt.show() """


#Pie chart con porcentaje de tumores tipo 2 y 4
""" counts = data['Class'].value_counts()

plt.pie(counts, labels=counts.index, autopct='%1.1f%%')
plt.show() """