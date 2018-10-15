## Central_Folder

This is the main folder where most of the reusable code is stored. In most cases, an object oriented desing is used in order to facilitate further extensions and to combine the different classes developed to 
preprocess, clean, modify, and model the data. Vizualizing the spread of a disease (with or without showing the center of mass of the patients) as well as modeling the outbreaks, takes place here. 

### Preprocessing, cleaning, and modifying the data.

The most important module where all the actions above take place is [Locations.py](https://github.com/eKoulier/Deep-Learning-Infectious-Diseases/blob/master/Central_Folder/Locations.py). 
The *disease_studied* function is used to load the data from a file and the *MonthlyTransform* class takes care of most of the preprocessing part.

### Interactive Maps

An important element of this project was the development of insightful visualizations in order to understand the dynamics of the disease and how it spreads from one region to another. The deliverable here is an
interactive map visualization that has a time slider functionality. The user is able to select the month that he is interested in having the map for and to hover over the map in order to explore the number of patients
for every municipality. Other options such as using a logarithmic color scale, or using the ratio of affected people to the population of a municipality are also provided. 

Two shnapshots of the interactive map are provided below. The left side map shows the number of patients per municipality and the right side one shows the ratio of patients per inhabitants. Notice the timeslides in the 
bottom of both maps that allows the selection of the month. **Discaimer** these snapsshots do not correspond to the actual data and they are here to just to provide a representation of how the interactive map looks like.

<img src="Images/map.jpg" width="600"/> 
 
### Modeling the Disease

Many methods and algorithms were used to model the disease. The models that performed the best are stored in the 
[Predictors.py](https://github.com/eKoulier/Deep-Learning-Infectious-Diseases/blob/master/Central_Folder/Predictors.py) module.  
