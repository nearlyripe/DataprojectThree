#!/usr/bin/env python
# coding: utf-8

# In[1]:


# So the old file takes ~40 mins to run, so I stripped out the essential bits 
# now that the project is closed. 

# Enjoy


# In[2]:

print("Script Running")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os.path
import sklearn as sk
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import time
from datetime import date as dt
import openpyxl
from sklearn.linear_model import LogisticRegression
Current_Date = dt.today().strftime("%d-%b-%Y")

print("Imports Complete")


# In[3]:


dailyExcel = "https://www.arcgis.com/sharing/rest/content/items/e5fd11150d274bebaaf8fe2a7a2bda11/data"
dailyCases = pd.read_excel(dailyExcel)


# In[4]:


nList = []
for x in dailyCases.index:
    nList.append(x)
    
dailyCases['n'] = nList
dailyCases.head()

cumulative = dailyCases[['n']].to_numpy()

extraValues = cumulative+cumulative.max()
predictive = np.vstack((cumulative, extraValues))
print("Setup Complete")

# In[5]:

print("Creating Directories")
home = os.getcwd()
extension =  Current_Date
path = r"C:\Users\dilan\OneDrive\Documents\GitHub\DataprojectThree\reports_"

if os.path.isdir(path + extension):
    print("directory already made")
else:
    os.mkdir(path + extension)
    print("New directory made!") 


# In[6]:

print("Setting Up Models")
linear = LinearRegression()

linear.fit(cumulative, dailyCases.CumCases.to_numpy())

poly4 = PolynomialFeatures(degree = 4)
Xgraph4 = poly4.fit_transform(cumulative)
poly4.fit(Xgraph4, dailyCases.CumCases.to_numpy())

linear2 = LinearRegression()
linear2.fit(Xgraph4, dailyCases.CumCases.to_numpy())


# In[7]:


x = predictive.ravel()
y = linear2.predict(poly4.fit_transform(predictive))
d = np.vstack((x, y))
df = pd.DataFrame(data = d)

# In[8]:

print("Writing Excel File")

writer = pd.ExcelWriter(Current_Date + '_Model.xlsx')
df.to_excel(writer, 'model', 'UTF-8')
os.chdir(path+extension)
writer.save()
os.chdir(home)
print("Done")

# In[9]:

print("Saving Exponential Graph")
plt.figure(figsize=(40,40))
plt.scatter(dailyCases.n, dailyCases.CumCases)
plt.plot(dailyCases.n, linear2.predict(poly4.fit_transform(cumulative)))
plt.figsize=(40, 10)
os.chdir(path+extension)
plt.savefig(Current_Date + '_Model_Grap')
plt.close();
os.chdir(home)
print("Done")


# In[10]:

print("Saving Daily Change Graph")
plt.figure(figsize=(40,40))
plt.plot(dailyCases.n, dailyCases.CMODateCount)
plt.figsize=(40, 40)
os.chdir(path+extension)
plt.savefig(Current_Date + '_Daily_Change')
plt.close();
os.chdir(home)
print("Done")


# In[11]:

print("Saving Daily Case Graph")
plt.figure(figsize=(40,40))
plt.plot (dailyCases.n, dailyCases.CumCases)
plt.figsize=(40, 40)
os.chdir(path+extension)
plt.savefig(Current_Date + '_Daily_Cases')

plt.close();

os.chdir(home)
print("Done")

# In[12]:

print("Starting Model Application")
bigSetRaw = pd.read_csv("https://raw.githubusercontent.com/RamiKrispin/coronavirus-csv/master/coronavirus_dataset.csv")
bigSet = bigSetRaw.drop(['Lat'], axis = 1).drop(['Long'], axis = 1).drop(['Province.State'], axis = 1)
bigSet.rename(columns = {"Country.Region" : "Country"}, inplace = True )

print("Data Setup Done")
print("Plotting Global Graph (1/3)")
plt.figure(figsize=(80,80))
for x in bigSet.Country.unique():
    plotSet = bigSet.loc[bigSet.Country == x]
    plotSet.head()
    sns.lineplot(plotSet.date, plotSet.cases)
    sns.scatterplot(plotSet.date, plotSet.cases)
    print(x + " has been plotted")
    
    plt.xticks(rotation = 45)
    os.chdir(path+extension)
    plt.savefig(Current_Date + '_Global_Graph')
plt.close()
os.chdir(home)
    


# In[ ]:


bigSet = bigSet.replace(to_replace = "Taiwan*" ,  value = "Taiwan") 

print("Plotting Global Cumulative Graph (2/3)")

plt.figure(figsize=(80,80))
for x in bigSet.Country.unique():
    plotSet = bigSet.loc[bigSet.Country == x]

    sns.lineplot(plotSet.date, plotSet.cases.cumsum())
    print(x + " has been plotted")
    
    plt.xticks(rotation = 45)
    os.chdir(path+extension)
    plt.savefig(Current_Date + '_Global_Graph_cumulative')
    os.chdir(home)   
plt.close();


# In[ ]:

print("Plotting Individual Graphs (3/3)")

for x in bigSet.Country.unique():
    
    plt.figure(figsize=(30,30))
    plotSet = bigSet.loc[bigSet.Country == x]

    sns.lineplot(plotSet.date, plotSet.cases.cumsum())
    print(x + " has been plotted")
    plt.xticks(rotation = 45)
    os.chdir(path+extension)
    plt.savefig(Current_Date + '_' + x + '_cases')
    plt.close();
    os.chdir(home)


# In[ ]:

print("Setting Up Chinese Model")

chinese = bigSet.loc[bigSet.Country == "China"]
chinaSet = chinese

nList = []
for x in chinaSet.index:
    nList.append(x)

zeroDay = nList[0]


for x in range(len(nList)):
    nList[x] = nList[x] - zeroDay  
    
chinaSet['n'] = nList

chinaSet['CumCases'] = chinaSet.cases.cumsum() 

cumulative = chinaSet[['n']].to_numpy()

dateUpper = 2000
dateLower = 1500

chinaSetTrim = chinaSet.loc[chinaSet.n >= dateLower]
x = chinaSetTrim[['n']].loc[chinaSetTrim.n <= dateUpper].to_numpy()
y = chinaSetTrim[['CumCases']].loc[chinaSetTrim.n <= dateUpper].to_numpy()

print("fitting logistic model")

print("Done")
print("Training Model")

logModel = LogisticRegression(multi_class='multinomial', solver ='newton-cg', max_iter = 5000)

xX = chinaSetTrim[['n']].loc[chinaSetTrim.n <= dateUpper].to_numpy()
yY = chinaSetTrim[['CumCases']].loc[chinaSetTrim.n <= dateUpper].to_numpy()

logModel.fit(xX,yY)

print("Logistic model Trained")


# In[ ]:


popData = 'https://population.un.org/wpp/Download/Files/1_Indicators%20(Standard)/CSV_FILES/WPP2019_TotalPopulationBySex.csv'
popData = pd.read_csv(popData)
hubeiPop = 59000000


# In[ ]:

print("Cleaning Data")
extension =  Current_Date
path = r"C:\Users\dilan\OneDrive\Documents\GitHub\DataprojectThree\Forecasts\Forecast_"

if os.path.isdir(path + extension):
    print("directory already made")
else:
    os.mkdir(path + extension)
    print("New directory made!") 

CHx = chinaSetTrim[['n']].loc[chinaSetTrim.n <= dateUpper].to_numpy()
CHy = chinaSetTrim[['CumCases']].loc[chinaSetTrim.n <= dateUpper].to_numpy()

popData = popData.replace('Congo', 'Congo B')
popData = popData.replace('Guinea', 'Guinea 2')
popData = popData.replace('Guinea 2-Bissau', 'Guinea-Bissau')
popData = popData.replace('Equitorial Guinea 2', 'Equitorial Guinea')
popData = popData.replace('Dominica', 'Dominica 2')
popData = popData.replace('Dominica 2n', 'Dominican')
popData = popData.replace('Democratic Republic of the Congo B', 'Democratic Republic of the Congo')
popData = popData[~popData.Location.str.contains("(and dependencies)")] # Hacky code I had to steal as it is getting late
popData = popData[~popData.Location.str.contains("AIMS")]
popData = popData[~popData.Location.str.contains("New Zealand (and dependencies)")]
popData = popData[~popData.Location.str.contains("Oceania")]
popData = popData[~popData.Location.str.contains("Australia/New Zealand")]
popData = popData[~popData.Location.str.contains("South Sudan")]
popData = popData.replace('Republic of Korea', 'Republic of Korea 2')
popData = popData.replace("Dem. People's Republic of Korea 2", 'NorthKorea')
popData = popData.replace('Niger', 'Niger 2')
popData = popData.replace('Niger 2ia', 'Nigeria')
popData = popData[~popData.Location.str.contains("China, Hong Kong SAR")]
popData = popData[~popData.Location.str.contains("China (and dependencies)")]
popData = popData[~popData.Location.str.contains("China, Macao SAR")]
popData = popData.replace('China, Taiwan Province of China', 'Taiwan')
popData = popData[~popData.Location.str.contains("Less developed regions, excluding China")]


# In[ ]:


bigSet = bigSet[~bigSet.Country.str.contains("Diamond Princess")]
bigSet = bigSet[~bigSet.Country.str.contains("Cruise Ship")]

newPopData = popData[popData.Time.eq(2019)]

print("Done")

print("Creating Logistical Forecasts (Final Operation)")

for x in bigSet.Country.unique(): 
    
    plt.figure(figsize=(30,30))
    plotSet = bigSet.loc[bigSet.Country == x]
    
    # Some small exception handling below for working with the population data. Of course the DRC and the normal congo
    # are being annoying. Also accents suck. Also the UN doesn't have the population of cruise ships, annoyingly.
    
    if(x == 'Congo (Brazzaville)'):
        x = 'Congo B'
    if(x == 'Congo (Kinshasa)'):
        x = 'Democratic Republic of the Congo'
    if(x == "Cote d'Ivoire"):
        x = "Côte d'Ivoire"
    if(x == "Dominica"):
        x = "Dominica 2"
    if(x == "Guinea"):
        x = "Guinea 2"  
    if(x == "Korea, South"):
        x = "Republic of Korea 2"
    if(x == "Laos"):
        x = "Lao People's Democratic Republic"
    if(x == "Niger"):
        x = "Niger 2"
    if(x == "US"):
        x = "United States of America"
    if(x == "Vietnam"):
        x = "Viet Nam"
    if(x == 'West Bank and Gaza'):
        x = "State of Palestine"
        
    print("Plotting" + x)

    newPop = newPopData[newPopData.Location.str.contains(x)]
    newPop = newPop.PopTotal.to_numpy()
    newPop = newPop*1000

    print(newPop)
    
    nList = []
    
    for y in plotSet.index:
        nList.append(y)

    zeroDay = nList[0]

    for y in range(len(nList)):
        nList[y] = nList[y] - zeroDay  
    
    plotSet['n'] = nList

    plotSet['CumCases'] = plotSet.cases.cumsum()
    
    popRatio = newPop / hubeiPop
    print(popRatio)
    
    PSx = plotSet[['n']].to_numpy()
    PSy = plotSet[['CumCases']].to_numpy()
    
    dx = CHx[0] - PSx[0] + ((PSx[0] - PSx[0]) / 7) + 440
    dy = CHy[0] - PSy[0] + ((PSy[0] - PSy[0]) / 2) 

    try:
        plt.figure(figsize=(20,20))
        plt.scatter(PSx, PSy)
        plt.plot(CHx - dx, (logModel.predict(CHx) - dy)*popRatio )

        print(x + " has been plotted")
        plt.xticks(rotation = 45)
        plt.xlim((0, 1200))
        os.chdir(path+extension)
        plt.savefig(Current_Date + '_' + x + '_cases_forecasted')
    except:
        print()
    else:
        print("Didn't work lol" + x)
    plt.close('all');
    
os.chdir(home)

print("Done")
# In[ ]:




