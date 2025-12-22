# CSlearn: Towards Bridging GIS and 3D Modeling: A Framework for Learning Coordinate Conversion Using Machine Learning

A machine learning framework for converting coordinates between GIS (geographic) and Unity 3D coordinate systems. This project evaluates multiple regression models to learn the mapping function between these two coordinate spaces, enabling seamless integration of geographic data into 3D simulation environments.

## Overview

The framework consists of three main stages:
1. **Data Generation**: Creates training datasets by generating point grids in both coordinate systems
2. **Model Training & Evaluation**: Trains and evaluates 35+ regression models across different training set sizes
3. **Analysis**: Performs comprehensive analysis of model performance, including clustering and visualization

## Files

### Notebooks

#### CSLearn_DataGen.ipynb
Generates training data for coordinate conversion by creating point grids in both GIS (latitude/longitude) and Unity 3D (x/y) coordinate systems. The notebook:
- Defines coordinate system boundaries for the Mina region (Saudi Arabia)
- Implements a point generation algorithm that creates grid points in both systems
- Visualizes the coordinate systems and the point generation process
- Prepares datasets for machine learning model training

#### CSLearn_ModelTrainingAndEval.ipynb
Trains and evaluates multiple machine learning models for coordinate conversion. This notebook:
- Tests 35+ regression models including RandomForest, LinearRegression, Neural Networks, SVMs, and ensemble methods
- Evaluates models across different training set sizes (N values: 2, 4, 8, 10, 50, 100, 200, 400)
- Collects comprehensive metrics: train/test scores, model size, fit time, and scoring time
- Runs multiple trials (5 per configuration) for statistical reliability
- Outputs results to CSV files for further analysis

#### CSLearn_Analysis_Final.ipynb
Performs comprehensive analysis of the experimental results. The notebook:
- Creates heatmaps showing train/test scores across different training set sizes
- Generates bar plots comparing fit time and test scoring time across models
- Performs clustering analysis to group models by performance characteristics
- Creates scatter plots visualizing the trade-offs between model size, fit time, and accuracy
- Identifies optimal models based on different criteria (accuracy, speed, size)

### Data Files

#### CSLearnData.csv
Contains the complete evaluation results from all model training experiments. Includes:
- Performance metrics: train/test scores with standard deviations
- Timing measurements: fit time and scoring time (train and test) with standard deviations
- Model size statistics: average and standard deviation in KB
- Results for all 35+ models across different training set sizes (N values)

#### models_table.csv
A reference table mapping model numbers (1-35) to their full names. Used for labeling and cross-referencing models in analysis visualizations.

### Other Files

#### LICENSE
CC0 1.0 Universal - This project is released into the public domain, allowing free use for any purpose.

## Key Findings

The framework evaluates models across three main clusters:
- **Small models (<11 KB)**: Fast, lightweight models with good accuracy
- **Medium models (11-2758 KB)**: Balanced performance models
- **Large models (>2758 KB)**: High-accuracy ensemble models with larger memory footprint

Models are evaluated on their ability to accurately convert coordinates while considering trade-offs between accuracy, training time, inference speed, and model size.

## Usage

1. Run `CSLearn_DataGen.ipynb` to generate training data
2. Run `CSLearn_ModelTrainingAndEval.ipynb` to train and evaluate models
3. Run `CSLearn_Analysis_Final.ipynb` to analyze results and generate visualizations

## Dependencies

- Python 3.x
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- geopandas
- shapely
- contextily
- tensorflow/keras (for neural network models)
- scikeras