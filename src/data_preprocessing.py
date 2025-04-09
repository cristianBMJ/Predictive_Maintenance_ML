# data_preprocessing.py

import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

def load_and_process_data():
    """
    Load gas turbine emission data for the years 2011 to 2015 and calculate the TAT/TIT ratio.
    
    Returns:
        pd.DataFrame: A DataFrame containing the concatenated data for the years 2011 to 2015.
    """
    # data's 2011
    gt_2011 = pd.read_csv('data/gas_turbine_emision/gt_2011.csv')
    gt_2011['Year'] = 2011

    # data's 2012
    gt_2012 = pd.read_csv('data/gas_turbine_emision/gt_2012.csv')
    gt_2012['Year'] = 2012  # Corrected year from 2014 to 2012

    # data's 2013
    gt_2013 = pd.read_csv('data/gas_turbine_emision/gt_2013.csv')
    gt_2013['Year'] = 2013

    # data's 2014
    gt_2014 = pd.read_csv('data/gas_turbine_emision/gt_2014.csv')
    gt_2014['Year'] = 2014

    # data's 2015
    gt_2015 = pd.read_csv('data/gas_turbine_emision/gt_2015.csv')
    gt_2015['Year'] = 2015

    gt = pd.concat([gt_2011, gt_2012, gt_2013, gt_2014, gt_2015], ignore_index=True)

    gt['TAT_TIT_Ratio'] = gt['TAT'] / gt['TIT']

        # Split the data into 85% and 15%
    split_index = int(len(gt) * 0.85)
    gt_85 = gt.iloc[:split_index]  # First dataset (85%)
    gt_15 = gt.iloc[split_index:]   # Second dataset (15%)

    gt_85.to_csv('data/processed_data_training.csv', index=False)

    gt_15.to_csv('data/processed_data_evaluation.csv', index=False)


