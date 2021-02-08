# ForumRec

## Introduction
This repository is for the ForumRec project, a recommendation system, that recommends users questions they are adept to answer on the [superuser](superuser.com) forum of [StackExchange](https://stackexchange.com).

##  Files

For this project, we have files for retriving the data, running the models, processing it into the desired output, and creating a front end design for our recommendation system. The files are described below and explain the purpose of each part of the repository.

> etl.py: Passes in the configs file related to it. The process for taking the data from the data files, extracting the necessary information, and splitting them into 6 separate files based around a certain date. The files are train, validation, and test sets split into questions and answers in order to be usable and trainable by our model.

> run.py: Runs etl on the data. Runs both of the cosine similarity models (user and post based). Runs the Runs baselines on the baseline file to get the baseline accuracy values. Can also run all of these files (except the etl) on test data.

> CosinePostModel.py: Passes in the configs file related to it. This file will run a cosine similarity model using the posts as the object to find results for. This will also produce an accuracy to determine our model's ability.

> CosineUserModel.py: Passes in the configs file related to it. This file will run a cosine similarity model using the users as the object to find results for. This will also produce an accuracy to determine our model's ability.

> baseline.py: Passes in the configs file related to it. Baseline file that finds the top 100 most popular users in the training data and produces predictions for the data to see how accurate top popular predictions would be for the data.

> requirements.txt: Contains the amount of processing resources recommended to run the files and the packages needed and the versions that were used to run all the processes.

> LICENSE: A file that contains the reuse and use licenses for this repository.

> SuperUser EDA.ipynb: Inside the notebook directory. Notebook containing the exploratory data analyses that was taken on the data to further understand and gain insight on the data we were using, and how we can use the data to build the recommendation system we wanted.

> ETL.ipynb: Inside the notebook directory. Notebook containing the etl processes and the results to look at the etl.

> NLP.ipynb: Inside the notebook directory. Notebook containing the natural language processing code that looks at the text data and creates a model through that code.

##  Directories

The following directories were created by us in order to be able to store and retain the necessary information needed for different purposes.

> config: Contains a list of all the config files that determines the parameters of each file. Use these files according to their use to change the parameters and change which subset of data you are running the processes on. Make sure you are changing the file paths correctly and throughout the entire config file.

> data: Location to put the original raw data in. It also would contain a final directory which would contain the data retrieved after running the repository.

> test: Contains inside the data directory and inside that the data used to run the repository on a small subset of the data to ensure the models and the scripts are running correctly.

> src: Contains inside the src, models, and baselines folders which contain the etl, cosine models, and baselines python files that are described above.

> notebooks: Contains multiple notebooks that explore the data. The notebooks are described above.

> extension: Contains code stubs in JavaScript, json, and html to build the browser extension front end for this repository as well as a png of the logo.

## Running the Code
Prior to running the code, make sure that you install all the packages listed in *requirements.txt* 

In order to obtain the data, one can follow the processes below:

### Creating the Data

To create the processed data, run this following command on the command line terminal:
```
python run.py data
```
Where the data will be replaced processed and be returned into new files usable by our models in this project and placed in the data directory.

### Running the cosine models

To run the cosine models on the data, run this following command on the command line terminal:
```
python run.py models
```
Where both of the cosine similarity models will be run to get the model accuracy and predictions.

### Comparing the model to baselines

To determine the accuracy of the baselines model, run this following command on the command line terminal:
```
python run.py baselines
```
Where the baseline model will be evaluated and the accuracy will be determined using the train data allowing us to compare the cosine similarity models to the performance of the baseline models.

### Running all the model targets

If you want to run all of these together, run this following command on the command line terminal:
```
python run.py all
```
Where the all 3 targets (excluding *test*) will run one after another in the order presented above.

### Testing all the model targets

To test how if the repository and all the models and scripts are working, run this following command on the command line terminal
```
python run.py test
```
Where 2 of the targets ('models' and 'baselines') above will all be run one after another in the order presented above, but on small test data so that we can observe how the models and scripts are working.