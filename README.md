
<div align="center">
    <p align="center">
        <h1 align="center">COMP 472 Project</h1>
        "SmartClass A.I.ssistant", CNN implementation that can analyze facial expressions and categorize them into 
        different states/classes.
        <br>
    </p>
</div>

## Group Members
- 40231530, Bryan Carlo Miguel
- 40212780, Yasser Ameer
- 40213100, William Nazarian

## Table of Contents
1. [Setting up the Project](#setting-up-the-project)
2. [Executing Data Visualization Scripts](#setting-up-the-project)
2. [Project Contents](#project-contents)

## Setting up the Project
The project comes with a `requirements.txt` for installing dependencies. The simplest way to get the program running is
to create a `venv`, and then install the dependencies as seen below:
```
$ py -m venv comp472
$ comp472/Scripts/activate
$ pip install -r requirements.txt
    ...
```

<br>

## Executing Data *Visualization* Scripts
The project splits the visualization functionality into multiple scripts that can take input parameters.

### Class Distribution
The module `scripts.plot_bars` contains the functionality for visualizing the number of images in each class. Executing 
the following command executes the script.
```
$ py -m scripts.plot_bars
    ...
```

### Pixel Intensity Distribution 
The command `py -m scripts.plot_histograms <CLASS NAME>` visualizes the distribution for the specified class. For example, the
following command visualizes the intensities for the class `happy`.
```
$ py -m scripts.plot_histograms happy 
    ...
```

### Sample Images 
The command `py -m scripts.image_sampler <CLASS NAME>` samples 15 images from the specified class, plots each of the 
images' pixel intensities, then displays them all side-by-side in a 5x3 grid. For example, the following command 
does this sampling and visualization for the class `happy`.
```
$ py -m scripts.image_sampler happy 
    ...
```

### Train and Evaluate CNN Model
To train and evaluate the model first run the following command in the root of our project
```shell
python main.py
```
Then you will be asked to choose one of our different types of models to train the facial expression images
1. Main Model
2. Variant 1
3. Variant 2

After picking you will be asked to name the CNN to whatever you would like.
Finally, you get the see the training process and how the model improves after each epoch reducing the value of its loss function and increasing the accuracy

<br>

## Project Contents
- The `part1` directory contains the dataset, as well as a `.csv` file containing paths to the raw images, as well as their classification.
- The `scripts` directory contains the main scripts used for data cleaning and visualization.
- The `utils` directory contains any supporting python modules that contain common functionality.
```
project directory
├── part1 
│   ├── structured_data 
│   ├── Combined_Labels_DataFrame.csv 
│   │    ... 
│
├── scripts 
│   │    ... 
│
└── utils 
    │    ... 

```
