import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

""" #leer el archivo
with open('BCW.data') as f:
    reader = csv.reader(f)
    data = [row for row in reader]

#limpiar el archivo y convertir los numeros a integers
cleaned_data = []
for row in data:
    cleaned_row = []
    for value in row:
        try:
            cleaned_value = int(value)
        except ValueError:
            # Si hay valores que no tienen un numero se sustituye por un 0
            cleaned_value = 0
        cleaned_row.append(cleaned_value)
    cleaned_data.append(cleaned_row)

 #poner la información limpia en un csv
with open('cleaned_file1.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(cleaned_data) """


#Exploracion de datos

""" data = pd.read_csv('cleaned_file.csv')

print("Data shape")
print(data.shape, "\n")



print("Data basic statistics")
print(data.describe(), "\n")

print(data.iloc[:10, :4])
 """


data = pd.read_csv('cleaned_file.csv')

print("Data types")
print(data.dtypes, "\n")




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
