#!/usr/bin/env python
# coding: utf-8




import pyreadr as pr
import pandas as pd
import numpy as np
import pingouin as pg
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from scipy.stats import skew, kurtosis, pearsonr 
import statistics # for computing the mode of a variable
import platform   # to print the version of Python and selected libraries
import openpyxl
import pyreadstat
import matplotlib.image as mpimg





# The function `extractReliab` is useful to obtain only the alpha value and then round it to 2 numbers

def extractReliab(items):
    """
    Extracts Cronbach's alpha
    
    Parameters:
    - items: specify the items of the target construct to compute ts alpha
    
    Return:
    - float: a float with two decimals
    """
    return float(
        (round(pg.cronbach_alpha(data = items, ci = .99)[0], 2))
        )


# The function `calculate_statistics` is useful to print descriptive statistics, similar to APA style

def calculate_statistics(my_data):    
    """
    Calculates descriptive statistics for the given data frame.

    Parameters:
    - my_data (pandas.DataFrame): Input data frame containing the variables.

    Returns:
    - pandas.DataFrame: Table with calculated statistics.
    """        
    # Calculate skewness and kurtosis for each column
    skw = my_data.apply(skew)
    kurt = my_data.apply(kurtosis)

    # Create a new DataFrame with the desired statistics
    table             = pd.DataFrame()
    table['N']        = my_data.count()
    table['Mean']     = my_data.mean()
    table['SD']       = my_data.std()
    table['Median']   = my_data.median()
    table['Min']      = my_data.min()
    table['Max']      = my_data.max()
    table['Skewness'] = skw
    table['Kurtosis'] = kurt

    # Reset the index to use variable names as rows
    table.reset_index(inplace=True)
    table.rename(columns={'index': 'Variable'}, inplace=True)

    # Round the values to two decimals
    table = table.round(2)

    return table


# The function `r_pvalues` extracts p-values from all pairs of correlations in a dataframe

def r_pvalues(df):
    '''
    Print p-values from a correlation matrix
    
    Parameters:
    - df: a datframe
    
    Returns:
    - a matrix of p-values
    '''
    cols = pd.DataFrame(columns=df.columns)
    p = cols.transpose().join(cols, how='outer')
    for r in df.columns:
        for c in df.columns:
            tmp = df[df[r].notnull() & df[c].notnull()]
            p[r][c] = round(pearsonr(tmp[r], tmp[c])[1], 4)
    return p


# The function `corr_matrix` returns a correlation matrix with colours, values, and no upper triangle

def corr_matrix(my_data, my_dpi=150):
    
    '''
    Print a correlation matrix heatmap, with values inside, and no upper triangle.
   
    Parameters:
     - my_data: (pandas.DataFrame): Input data frame containing the variables.
     - my_dpi: specify dpi (default = 300).
   
    Returns:
     - a matplotlib figure.
    '''
    
    # Compute p-value matrix
    def compute_pvalues(data):
        return data.apply(lambda x: data.apply(lambda y: pearsonr(x, y)[1]))

    # Function to determine significance level
    def significance_asterisks(p_val):
        if p_val < 0.001:
            return '***'
        elif p_val < 0.01:
            return '**'
        elif p_val < 0.05:
            return '*'
        else:
            return ''
    
    fig, ax = plt.subplots(figsize=(6, 6), dpi=my_dpi)
    
    pvalues = compute_pvalues(my_data)
    corr_values = my_data.corr()

    # Mask for upper triangle and for p-values above 0.05
    mask = np.triu(np.ones_like(corr_values, dtype=bool))
    
    # Generate a mask for the significant cells
    significance = pvalues.applymap(significance_asterisks)
    
    # Merge correlation values and significance level
    annotations = corr_values.applymap('{:.2f}'.format) + '\n' + significance

    sns.heatmap(
        corr_values,
        mask=mask,
        vmin=-0.7,
        vmax=0.7,
        cmap=sns.diverging_palette(20, 220, as_cmap=True),
        annot=annotations,
        fmt="s",
        annot_kws={"size": 6}, # sets the font size of the correlation value annotations
        cbar=False             # remove vertical bar (bar with coulurs indicating the size of the correlations)
    )
    plt.xticks(fontsize=7,
               rotation=45) # Rotate x-axis labels by *n* degrees
    plt.yticks(fontsize=7,
               rotation=45)
    return ax.get_figure()


# The function `overf_check` returns an histogram for overfitting check, using bootstrap as technique and using MAE as metric.
#  It also returns median and 95% confidence intervals for both MAE and R^2, based on bootstrap (with 1000 resamplings)


   

def reverse_scale(value):
    if pd.isnull(value):
        return np.nan  # Preserve NaN values
    else:
        return 6 - int(value)  # Convert to integer before reversing

def reverse_scale_4(value):
    if pd.isnull(value):
        return np.nan  # Preserve NaN values
    else:
        return 5 - int(value)  # Convert to integer before reversing

def add_corr_to_figure(corr_ax, position, fig):
    ax_new = fig.add_axes(position)
    ax_new.axis('off')
    corr_ax.figure.canvas.draw()
    image_data = np.frombuffer(corr_ax.figure.canvas.buffer_rgba(), dtype=np.uint8)
    image_shape = corr_ax.figure.canvas.get_width_height()[::1]  # Ottieni le dimensioni in ordine corretto (height, width)
    image_data = image_data.reshape((*image_shape, 4))  # RGBA format
    ax_new.imshow(image_data)

