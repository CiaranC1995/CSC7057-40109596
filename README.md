# CSC7057-40109596
Ciarán Casey - 40109596

This is my submission for CSC7057: Individual Software Development Project. I have built a binary classifier that predicts whether a given piece of text is AI-generated or human-written. The classifier is deployed within a user-friendly, intuitive web application for ease of use, with added account holder functonality included for users who wish to sign up. 

<b> DIRECTORY LAYOUT </b>

All code developed throughout the project is found within the 'CSC7057-40109596/src' folder. This includes the code for initial dataset manipulation, model training, data visalisation and any other ancilliary scripts needed. 

The important code for the web application software is found within the <b>src/WebApp</b> subdirectory. Here is the backend Flask application code, frontend user interface HTML / CSS code and internal API code that facilitates communication between the software components and the database. Within the parent directory is a requirements.txt that includes all project dependencies that need to be installed before the project will successfully run on a local machine.

<b> INSTALLING PROJECT </b>

To run this project on a local machine, ensure that the following folders and files from the Canvas submission are locally available:

1. CSC7057-40109596/src/WebApp/REST_API (Folder)
2. CSC7057-40109596/src/WebApp/WEB_APP (Folder)
3. CSC7057-40109596/src/WebApp/DATABASE/csc7057.sql (Database File)

It is also assumed that Visual Studio Code, MAMP or XAMPP, and all the dependencies listed in the requirements.txt are installed on the local machine.

Use the following line of code in a terminal that has the directoy containing requirement.txt to install dependencies:

pip install -r requirements.txt

To enable the Rest API to connect to the database, it needs to use MySQL port 3306. When running it locally on a Windows machine, MAMP is preferred over XAMPP, which caused issues for me during development. 

Once both folders are opened in separate windows in Visual Studio Code, and the database file has been added to the local phpMyAdmin server connected using MAMP, the web application is ready to run. 

To connect them to their relevant ports and allow the servers to begin listening, open and run the app.py and api.py scripts in the CSC7057-40109596/src/WebApp/WEB_APP and CSC7057-40109596/src/WebApp/REST_API subdirectories respectively. This may take a couple of minutes to complete depending on the machine.

Messages confirming that the servers are running will be displayed through the console in each window. The user can now open a browser and navigate to "http://127.0.0.1:5000" to access the project user interface. The home page is the first page displayed to the user. To explore the site's features, use the navigation bar at the top of the screen.
