# CS-433 Machine Learning Project 1

The aim of this project is to determine whether a particle is Higgs Boson or not, by applying machine learning classification algorithms. In sum it is a binary classification of $30$ features vectors dataset. Applying the algorithms implemented during labs we will be creating a linear model $f$ that best classifies the data points $x_n$. 

## Getting Started


To begin you need to clone this repository, or in our case simply download it and unzip the whole folder.


### Prerequisites

The project runs on python3 and requires as external library: numpy (for mathematical operations), pandas (for exploratory data analysis), matplotlib (for plotting) and jupyter (notebooks) 

```
pip3 install numpy 
pip3 install jupyter 
pip3 install pandas
```

### Testing your installation

To test out that the installation were correctly done, there is a small notebook named: library_requirement.ipynb.
It is located in the scripts folder.  We also prepared a requirements.txt that you can simply run as follows:
```
pip install -r requirements.txt
```

## Structure
    data: 
        - test.csv (zip) : dataset we predict the output of
        - train.csv (zip) : dataset equipped with the the classification of each point, used for training
        - sample-submission.csv: example of format csv to submit
    scripts: 
        - data_processor.py: all the functions used for data processing are located here
        - HiggsBosonReport.ipynb: the notebook showing our thought process and different attempts
        - implementations.py: all the functions used for making models are located here
        - library_requirement.ipynb: its purpose is described above 
        - plot.py: all the functions used for plotting results are located here
        - proj1_helpers.py: all the functions for fetching data, outputting csv and predict labels are located here
        - run.py: python file that we run in order to output the csv for submission
    weights: for weights that were saved throughout the project, new weights are saved on that folder too
    project1_description.pdf: description of rules for this project
    report.pdf: final report, written report of all we have done
    requirements.txt: description of usage written above
    
### Data Process 
The data processing includes the preprocessing, feature augmentation and also the cross validation. 
For each section we will detail and explain the reasons for the choice.

-  preprocessing
        After analysing the data, we noticed a huge amount of NaN values. In order to deal with these values, there are multiple ways among them would be erasing the vectors that have a NaN value (-999). Unfortunately around 70% of the vectors has at least a NaN value. We can also go for a frequency replacing but as the terms were not repeating significantly (for some features), we decided that in the end we would be replacing both the NaN values from training and validation set by the mean of the training set of each feature column. As it is common to standardize the data, so we did with the NaN replaced data.
- feature augmentation
        As mentioned in the lecture, the first way of feature augmentation we thought of was building polynomial. But then we noticed that some values were too large when taken to the n-th power, and therefore decided to standardize once more after augmenting the features before feeding it to the model (not taking the first column of ones). We then thought of adding as features interaction terms, but unfortunately we think that we did not manage to exploit the full potential yet.
- cross validation
        As a cross validation we used the simplest one, being the holdout method. We therefore simply randomly (fixed seed) splitted the original data in train data and validation data (with a ratio of 9:1). As we saw that this method was enough to get a stable validation accuracy (compared to the AICrowd accuracy) we did not implement any other cross validation.
        
The processes are detailed in the HiggsBosonReport.ipynb notebook and also in the run.py.


## Authors

**Agal Aryan** -**Chan Michael** -  **Levitin Vlad**  

Our Respective github repository:
