#!/usr/bin/env python
# coding: utf-8

# In[22]:


# Aufgabe 1: 
# Write 2 python function to get the indices of the sorted elements of given lists and compare the speed. 
# one is without numpy package and the other is with numpy. 
# (raise an error message if the input is null or not numerical)


import numpy as np
import pandas as pd
import random
import time

list1 = [23, 104, 5, 190, 8, 7, -3]
list2 = []
list3 = [random.randint(1, 100000) for _ in range(1000000)] 

# Function with numpy
def sorted_indices_numpy(lst): 
    if not lst:
        raise ValueError("There is nothing in the list!")
    for item in lst:
        if not isinstance(item, (int, float)):
            raise ValueError("Value in list is not numerical!")
    list_arr = np.array(lst)
    indices = np.argsort(list_arr)
    return indices


# Function without numpy
def sorted_indices(lst):
    if not lst:
        raise ValueError("There is nothing in the list!")
    for item in lst:
        if not isinstance(item, (int, float)):
            raise ValueError("Value in list is not numerical!")
    indices = sorted(range(len(lst)), key=lambda i: lst[i])
    return indices


# In[23]:


# Ausgabe:
# List 1
print("List 1:")
start1 = time.time()
sorted_indices(list1)
time1 = time.time() - start1
print("Indices without NumPy:", sorted_indices(list1))
print(f"Computation time without numpy: {time1}")

start2 = time.time() 
sorted_indices_numpy(list1)
time2 = time.time() - start2
print("Indices with NumPy:", sorted_indices_numpy(list1))
print(f"Computation time with numpy: {time2}")


# In[24]:


# List 2
print("List 2:")
start1 = time.time()
print("Indices without NumPy:", sorted_indices(list2))
time1 = time.time() - start1
print(f"Computation time without numpy: {time1}")
start2 = time.time() 
print("Indices with NumPy:", sorted_indices_numpy(list2))
time2 = time.time() - start2
print(f"Computation time with numpy: {time2}")


# In[25]:


#List 3:
print("List 3:")
start1 = time.time()
sorted_indices(list3)
time1 = time.time() - start1
print("Indices without NumPy:", sorted_indices(list3))
print(f"Computation time without numpy: {time1}")

start2 = time.time() 
sorted_indices_numpy(list3)
time2 = time.time() - start2
print("Indices with NumPy:", sorted_indices_numpy(list3))
print(f"Computation time with numpy: {time2}")


# In[26]:


# Aufgabe 2:

# Load the countries.csv directly via URL import into your panda data frame!
url = "https://raw.githubusercontent.com/WHPAN0108/BHT-DataScience-S23/main/python-DS/country.csv"
df = pd.read_csv(url)


# In[27]:


# Display descriptive statistics for the numerical column (count, mean, std, min, 25%, 50%, 75%, max) HINT: describe
df.describe()


# In[28]:


# Show the last 4 rows of the data frame.
print(df.tail(4))


# In[29]:


# Show all the rows of countries that have the EURO
print(df[df['Currency'] == 'EUR'])


# In[30]:


# Show only name and Currency in a new data frame
# create a dataframe
name_currency = df[['Name', 'Currency']]

# Display the new DataFrame
print(name_currency)


# In[31]:


# Show only the rows/countries that have more than 2000 GDP (it is in Milliarden USD Bruttoinlandsprodukt)
print(df[df['GDP'] > 2000])


# In[32]:


# Select all countries where with inhabitants between 50 and 150 Mio
print(df.query('50_000_000 <= Population <= 150_000_000'))


# In[33]:


# Calculate the GDP average (ignore the missing value)
average=df['GDP'].mean()
print(average)


# In[89]:


# Calculate the GDP average (missing value treated as 0)
average=df['GDP'].fillna(0).mean()
print(average)


# In[90]:


# Calculate the population density (population/area)  of all countries and add as new column
df['Population Density'] = df['Population'] / df['Area']
print(df)


# In[96]:


# Sort by country name alphabetically
sorted_countries = df['Name'].sort_values()
print(sorted_countries)


# In[98]:


# Create a new data frame from the original where the area is changed: 
# all countries with > 1000000 get BIG and <= 1000000 get SMALL in the cell replaced!
area_df = df.copy()
area_df['Area'] = area_df['Area'].apply(lambda x: 'BIG' if x > 1000000 else 'SMALL')
print(area_df)


# In[ ]:




