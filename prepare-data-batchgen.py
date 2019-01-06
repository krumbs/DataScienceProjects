# file to create a normalised and balanced data set
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

def normalise_modified_kepler_dataset(data, norm_criteria=0):
    mod_shift_up = data + np.max(data)
    if norm_criteria == 0:
        mod_shift_up_norm_mean = mod_shift_up/np.mean(mod_shift_up)
        return (1. - mod_shift_up_norm_mean)/np.var(mod_shift_up_norm_mean)
    if norm_criteria == 1:
        mod_shift_up_norm_median = mod_shift_up/np.median(mod_shift_up)
        return (1. - mod_shift_up_norm_median)/np.var(mod_shift_up_norm_median)
    if norm_criteria == 2:
        data = ((data - np.mean(data, axis=1).reshape(-1, 1))/np.std(data, axis=1).reshape(-1, 1))
        return data

def prep_data(raw_data, norm_criteria=0):
    """ Summary: normalise the Kepler Dataset
        INPUT:
        data: matrix of values
        norm_criteria = {0,1}
        norm_criteria = 0 -- normalise by mean
        norm_criteria = 1 -- normalise by median
        OUTPUT:
        df_labels: dataFrame of the labels
        norm_training_dataset: dataFrame of the values
        """

    df_subset = raw_data.ix[0:, :]
    # shuffle the new df
    unnormalized_training_dataset = shuffle(df_subset)
    extracted_data = unnormalized_training_dataset.ix[:, 1:].values
    norm_training_dataset = normalise_modified_kepler_dataset(extracted_data, norm_criteria)

    df_labels = unnormalized_training_dataset.ix[:, 'LABEL']

    return df_labels, extracted_data, norm_training_dataset


