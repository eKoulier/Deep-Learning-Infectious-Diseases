# Deep-Learning-Infectious-Diseases
This is the repository for the Infectious Disease Project sponsored by [GGD Netherlands](https://nl.wikipedia.org/wiki/Gemeentelijke_gezondheidsdienst). 
For this project, three GGD departments were involved: GGD West-Brabant, GGD Hart voor Brabant, and GGD Brabant Zuidoost. 
This project is done in collaboration with [JADS](https://www.jads.nl). For this repository the [PEP8](https://www.python.org/dev/peps/pep-0008/) is used to assert code readability.
The architecture of this repository is simple to facilitate further extentions and it is discribed below. 
There are mainly four different subdirectories for this project: 

* [Central_Folder](https://github.com/eKoulier/Deep-Learning-Infectious-Diseases/tree/master/Central_Folder) 
is the main file where the preprocessing of the data as well as the modeling part is automated. A tutorial on how to make an interactive choropleth map that has a time slider functionality 
is provided in [MapSlider.py](https://github.com/eKoulier/Deep-Learning-Infectious-Diseases/blob/master/Central_Folder/MapSlider.py).

* [Data](https://github.com/eKoulier/Deep-Learning-Infectious-Diseases/tree/master/Central_Folder) 
is where the data of this project is stored. We mainly focused on the disease named *Kinkhoest* which is the dutch word for Whooping Cough (medical name: Bordetella Pertussis).
Another disease studied is Mumps.

* [Analysis](https://github.com/eKoulier/Deep-Learning-Infectious-Diseases/tree/master/Analysis) 
is where the python notebooks are stored.

* [Application](https://github.com/eKoulier/Deep-Learning-Infectious-Diseases/tree/master/Application) 
is the folder that has the code to develop the dashboard that incorporates both the modeling and the visualization elements of this project.
