# ==============================================================
# ==============================================================
# ================ Machine Learning Functions ==================
# ==============================================================
# ==============================================================


# ==============================================================
# Library imports
# ==============================================================

# --- Standard Library ---
import os
import time
import heapq
from math import ceil
from itertools import combinations
from copy import deepcopy


# --- Core Scientific Stack ---
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import loguniform, randint, uniform

# --- Visualization ---
import matplotlib.pyplot as plt
import seaborn as sns

# --- Utilities ---
from rapidfuzz import fuzz, process

# --- Scikit-Learn: Preprocessing & Pipeline ---
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (OneHotEncoder, OrdinalEncoder, TargetEncoder, 
                                   StandardScaler, RobustScaler)

# --- Scikit-Learn: Feature Selection & Inspection ---
from sklearn.feature_selection import RFE, SelectFromModel, SelectKBest, mutual_info_regression
from sklearn.inspection import permutation_importance

# --- Scikit-Learn: Model Selection ---
from sklearn.model_selection import PredefinedSplit, RandomizedSearchCV, train_test_split

# --- Scikit-Learn: Metrics ---
from sklearn.metrics import mean_absolute_error, r2_score, make_scorer

# --- Scikit-Learn: Models ---
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor


# ==============================================================
# EDA and data cleaning
# ==============================================================

# drop non-useful columns

def sel_columns(df):
    
    """
    Selects relevant columns by removing non-informative ones from the DataFrame.
    
    - Drops columns: 'carID', 'paintQuality%', and 'hasDamage'
    - Ignores missing columns to prevent errors
    - Returns cleaned DataFrame with only relevant attributes
    """
    
    cols_to_drop = ['carID', 'paintQuality%', 'hasDamage']
    return df.drop(columns=cols_to_drop, errors='ignore')


def correct_num_variables(df_input):
    
    """
    Maps a dictionary to one or more DataFrames.

    Accepts either a single DataFrame or a dictionary of DataFrames:
    - If input is a dict, returns a dict of transformed DataFrames.
    - If input is a single DataFrame, returns the transformed DataFrame.

    For each DataFrame:
    1. If both key_col and value_col exist:
        a. Convert key_col to UTF-8 and stringify mapping keys.
        b. Map key_col values according to mapping into value_col.
        c. Fallback to original value_col for keys not in mapping.
    2. If key_col or value_col is missing, leave the DataFrame unchanged.
    """
    
    df = df_input.copy()
    
    # Mapping of columns names to their respective cleaning rules.
    rules = {
        "year": lambda x: pd.to_numeric(x, errors='coerce').where(lambda v: v <= 2020).round(0).astype("Int64"),
        "engineSize": lambda x: pd.to_numeric(x, errors='coerce').where(lambda v: (v >= 0.6) & (v <= 8.0)), #this limits are based on real life possible engineSize ranges https://www.refusedcarfinance.com/which-car-engine-size/
        "mpg": lambda x: pd.to_numeric(x, errors='coerce').where(lambda v: (v >= 10) & (v <= 120)), #this limits are based on real life possible mpg ranges https://www.nimblefins.co.uk/cheap-car-insurance/average-mpg
        "mileage": lambda x: pd.to_numeric(x, errors='coerce').where(lambda v: v >= 0),
        "previousOwners": lambda x: pd.to_numeric(x, errors='coerce').where(lambda v: v >= 0).round(0).astype("Int64"),
        "tax": lambda x: pd.to_numeric(x, errors='coerce').where(lambda v: v >= 0),
    }

    # Apply each rule only if the corresponding column exists in the DataFrame.
    for col, rule in rules.items():
        if col in df.columns:
            df[col] = rule(df[col])
            
    return df

def correct_column(df_input, column, valid_values, threshold=30):
    
    """
    
    Function that standardizes categorical values in a DataFrame column 
    using fuzzy string matching.

    Accepts a DataFrame and a target column, and replaces values that closely match
    a set of canonical valid values.

    For each value in the column:
    1. Normalize the value by lowercasing and stripping whitespace.
    2. Compare against valid_values using fuzzy matching (token_sort_ratio).
    3. Replace with the closest match if the similarity score exceeds threshold.
    4. Keep the original value if no match exceeds threshold.
    """
    
    df = df_input.copy()
    
    # Normalize text by lowercasing and removing whitespace
    df[column] = df[column].str.lower().str.strip()
    
    # Extract unique values to avoid redundant computations
    unique_vals = df[column].dropna().unique()
    
    # Create the mapping
    mapping = {
        val: process.extractOne(
            val, valid_values, scorer=fuzz.token_sort_ratio, score_cutoff=threshold
        )[0]
        if process.extractOne(
            val, valid_values, scorer=fuzz.token_sort_ratio, score_cutoff=threshold
        )
        else val
        for val in unique_vals
    }
    
    # Apply the mapping
    df[column] = df[column].replace(mapping)
    return df

# Dictionary with the correct models for each brand
brand_model_dict = {
    "ford": ['focus','kuga','ecosport','galaxy','fiesta','mondeo','c-max','b-max','s-max','tourneo custom','edge','escort','ka+','ka','puma','mustang','tourneo connect','fusion','ranger'],
    "mercedes": ['c class','a class','e class','glc class','gla class','b class','cl class','gle class','sl class','cls class','v class','s class','gl class','slk','cla class','x-class','m-class','gls class','glb class','g class','clk'],
    "vw": ['golf','polo','tiguan','passat','up','t-roc','touareg','touran','t-cross','golf sv','sharan','arteon','scirocco','amarok','gol','caravelle','tiguan allspace','cc','beetle','shuttle','jetta','california','caddy','eos','fox'],
    "opel": ['corsa','astra','insignia','mokka','zafira','adam','viva','meriva','combo life','gtc','antara','agila','vivaro','crossland','grandland','cascada','vectra','tigra','ampera'],
    "bmw": ['3 series','1 series','2 series','5 series','4 series','x1','x3','x5','x2','x4','m4','6 series','z4','x6','7 series','x7','8 series','i3','m3','m5','i8','m2','z3','m6'],
    "audi": ['a1','a2','a3','a4','a5','a6','a7','a8','q3','q5','q2','q7','tt','q8','rs6','rs3','r8','rs4','rs5','s3','s4','sq5','sq7','s8','s5'],
    "toyota": ['yaris','aygo','auris','c-hr','rav4','corolla','prius','verso','avensis','hilux','gt86','land cruiser','proace verso','supra','camry','verso-s','iq','urban cruiser'],
    "skoda": ['fabia','octavia','superb','citigo','kodiaq','karoq','scala','kamiq','rapid','yeti','roomster'],
    "hyundai": ['tucson','i10','i30','i20','kona','ioniq','santa fe','ix20','i40','ix35','i800','getz','veloster','accent','terracan']
}

# correct brand models

def correct_models_by_brand(df_input, threshold=30): # setting the threshold to 30 
    
    """
    Function that standardizes vehicle model names using brand-specific fuzzy matching.

    For each brand:
    1. Normalize the model names by lowercasing and stripping whitespace.
    2. Extract unique model values for that brand.
    3. Fuzzy match each model to the valid model names for the brand.
       - If similarity exceeds threshold, replace with best match.
       - Otherwise, keep the original model value.
    4. Apply the mapping to the 'model' column for that brand.
    """

    df = df_input.copy()

    # Normalize all model names
    if 'model' in df.columns:
        df['model'] = df['model'].str.lower().str.strip()

    for brand, valid_models in brand_model_dict.items():
        # filter rows belonging to that brand
        mask = df['Brand'] == brand

        # extract unique models for that brand

        unique_models = df.loc[mask, 'model'].dropna().unique() #getting unique models

        mapping = {} #initialize empty dictionary to store mappings

        for model in unique_models:
            # Fuzzy match each model to the valid model list for that brand
            match = process.extractOne(
                model, valid_models, scorer=fuzz.token_sort_ratio, score_cutoff=threshold
            )
            if match: #applying fuzzy match result
                best_match = match[0]
                mapping[model] = best_match
            else: # if no match found, keep original value
                mapping[model] = model

        # Apply the mapping to the 'model' column for the current brand
        df.loc[mask, 'model'] = df.loc[mask, 'model'].replace(mapping)
    
    return df


# clean the data 

def data_cleaning(df_input):
    
    """
    Applies key-value mapping to Polars DataFrames for categorical standardization.
    
    - Accepts either a single DataFrame or dictionary of DataFrames
    - Maps values from `key_col` to standardized values in `value_col`
    - Uses provided `mapping` dictionary for transformations
    - Returns data in same structure as input (dict → dict, DataFrame → DataFrame)
    - Leaves DataFrames unchanged if they lack `key_col` or `value_col`
    - Falls back to original values for unmapped keys
    """
    df = df_input.copy()

    # Dictionary with the correct values for each categorical column
    rules = {
        "Brand": ['skoda','opel','mercedes','ford','audi','hyundai','toyota','bmw','vw'],
        "transmission": ['manual', 'semi-automatic', 'automatic', 'unknown', 'other'],
        "fuelType": ['petrol', 'diesel', 'hybrid', 'electric', 'other']
    }


    # Apply corrections for each specified column
    for col, rule in rules.items():
        if col in df.columns:
            df = correct_column(df, col, rule)


    df = correct_models_by_brand(df)


    # Replace non-informative column categories with NaN
    df.loc[df['transmission'].isin(['unknown', 'other']), 'transmission'] = np.nan
    df.loc[df['fuelType'] == 'other', 'fuelType'] = np.nan

    
    return df

# data cleaning

def clean_basic_data(df_input):
    """
    Performs basic data cleaning on a DataFrame, including selecting relevant columns,
    correcting numeric variables, and standardizing text fields using fuzzy matching.
    Can be applied independently to both training and test datasets.
    """
    df = df_input.copy()
    
    # 1. Select relevant columns
    df = sel_columns(df)

    # 2. Correcting numeric variables (years, negative values)
    df = correct_num_variables(df)
    
    # 3. Text cleaning (Brands, Models, Transmission) - Fuzzy matching
    df = data_cleaning(df)
    
        
    return df


# Multivariate Analysis

#Functions

def correlation_ratio(categories, measurements):
    """
    Calculates the Correlation Ratio (Eta) between a categorical column and a numerical column.
    Returns value between 0 and 1.
    """
    # Create temporary DF to handle alignment and NaNs together
    df_temp = pd.DataFrame({'categories': categories, 'measurements': measurements}).dropna()
    
    if df_temp.empty:
        return np.nan
    
    f_cat = df_temp['categories']
    f_num = df_temp['measurements']
    
    ss_total = np.sum((f_num - f_num.mean())**2)
    
    if ss_total == 0:
        return np.nan

    # Groupby to get means per category
    groups = f_num.groupby(f_cat)
    group_means = groups.mean()
    group_counts = groups.count()
    overall_mean = f_num.mean()
    
    # Calculate Sum of Squares Between
    ss_between = np.sum(group_counts * (group_means - overall_mean)**2)
    
    # Calculate Eta Squared and result
    eta_squared = ss_between / ss_total
    eta_squared = min(max(eta_squared, 0.0), 1.0) # Clip to [0,1] range
    
    return np.sqrt(eta_squared)



def get_cramers_v(x, y):
    """
    Calculates Cramér's V for two categorical series.
    Returns value between 0 and 1.
    """
    # Create confusion matrix
    confusion_matrix = pd.crosstab(x, y)
    
    # Check for empty matrix or single-value columns (prevent division by zero)
    if confusion_matrix.size == 0:
        return np.nan
        
    n = confusion_matrix.sum().sum()
    min_dim = min(confusion_matrix.shape) - 1
    
    # Safety catch: If min_dim is 0, we can't divide. Return 0.0 correlation.
    if min_dim == 0 or n == 0:
        return 0.0
        
    chi2 = stats.chi2_contingency(confusion_matrix)[0]
    return np.sqrt(chi2 / (n * min_dim))

def cramers_v_matrix(df_input):
    """
    Returns a DataFrame showing Cramér's V correlation 
    between all categorical variables.
    """
    df = df_input.copy()
    
    # Select categorical columns
    cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Filter out constant columns (1 unique value) to avoid empty plots
    cols = [c for c in cols if df[c].nunique() > 1]
    
    # Create empty matrix
    matrix = pd.DataFrame(index=cols, columns=cols, dtype=float)

    # Calculate V for every pair
    for col1 in cols:
        for col2 in cols:
            matrix.loc[col1, col2] = get_cramers_v(df[col1], df[col2])

    return matrix

# ==============================================================
# Preprocessing functions
# ==============================================================

# -------------------------------------------------------
# Missing Values Treatment 
# -------------------------------------------------------


imputation_config = {
    "model": {
        "method": "mode",
        "hierarchy": [
            ('L1', ['Brand', 'engine_bin', 'fuelType', 'transmission', 'mpg_bin', 'tax_bin', 'year_bin']),
            ('L2', ['Brand', 'engine_bin', 'fuelType', 'transmission', 'mpg_bin', 'tax_bin']),
            ('L3', ['Brand', 'engine_bin', 'fuelType', 'transmission', 'mpg_bin']),
            ('L4', ['Brand', 'engine_bin', 'fuelType', 'year_bin']),
            ('L5', ['Brand', 'engine_bin', 'fuelType']),
            ('L6', ['Brand', 'engine_bin']),
            ('L7', ['Brand'])
        ]
    },
        "fuelType": {
        "method": "mode",
        "hierarchy": [
            ('L1', ['model', 'engine_bin', 'transmission', 'mpg_bin', 'mileage_bin', 'tax_bin', 'year_bin']),
            ('L2', ['model', 'engine_bin', 'transmission', 'mpg_bin', 'mileage_bin', 'tax_bin']),
            ('L3', ['model', 'engine_bin', 'transmission', 'mpg_bin', 'mileage_bin']),
            ('L4', ['model', 'engine_bin', 'transmission', 'mpg_bin']),
            ('L5', ['model', 'engine_bin', 'mpg_bin']),
            ('L6', ['model', 'engine_bin']),
            ('L7', ['model'])
        ]
    },
    "tax": {
        "method": "mean",
        "decimals": 0,
        "hierarchy": [
            ('L1', ['mpg_bin', 'model', 'year_bin', 'mileage_bin', 'transmission', 'engine_bin', 'fuelType']),
            ('L2', ['mpg_bin', 'model', 'year_bin', 'mileage_bin', 'transmission']),
            ('L3', ['mpg_bin', 'model', 'year_bin', 'mileage_bin']),
            ('L4', ['mpg_bin', 'model', 'year_bin']),
            ('L5', ['mpg_bin', 'model']),
            ('L6', ['model']),
        ]
    },
    "engineSize": {
        "method": "mean",
        "decimals": 1, 
        "hierarchy": [
            ('L2', ['model', 'fuelType', 'transmission', 'mpg_bin', 'tax_bin', 'year_bin', 'previousOwners']),
            ('L3', ['model', 'fuelType', 'transmission', 'mpg_bin', 'tax_bin']),
            ('L4', ['model', 'fuelType', 'transmission', 'mpg_bin']),
            ('L5', ['model', 'fuelType', 'transmission', 'year_bin']),
            ('L6', ['model', 'fuelType']),
            ('L7', ['model']),
            ('L8', ['Brand', 'fuelType', 'transmission']), 

        ]
    },
    "transmission": {
        "method": "mode",
        "hierarchy": [
            ('L2', ['model', 'engine_bin', 'fuelType', 'tax_bin', 'mpg_bin', 'year_bin', 'previousOwners']),
            ('L3', ['model', 'engine_bin', 'fuelType', 'tax_bin', 'mpg_bin', 'year_bin']),
            ('L4', ['model', 'engine_bin', 'fuelType', 'mpg_bin']),
            ('L5', ['model', 'engine_bin', 'fuelType']),
            ('L6', ['model', 'engine_bin']),
            ('L7', ['model']),
            ('L8', ['Brand', 'engine_bin']) 
        ]
    },
    "mpg": {
        "method": "mean",
        "decimals": 1,
        "hierarchy": [
            ('L2', ['model', 'tax_bin', 'fuelType', 'transmission', 'year_bin', 'engine_bin']),
            ('L3', ['model', 'tax_bin', 'fuelType', 'transmission', 'year_bin']),
            ('L4', ['model', 'tax_bin', 'fuelType', 'transmission']),
            ('L5', ['model', 'tax_bin', 'fuelType']),
            ('L6', ['model', 'fuelType']),
            ('L7', ['model']),
        ]
    },
    "year": {
        "method": "mean",
        "decimals": 0,
        "hierarchy": [
            ('L3', ['model', 'mileage_bin', 'tax_bin', 'mpg_bin', 'transmission']),
            ('L4', ['model', 'mileage_bin', 'tax_bin', 'mpg_bin']),
            ('L5', ['model', 'mileage_bin']),
            ('L6', ['Brand', 'mileage_bin']),
            ('L7', ['mileage_bin']),
            ('L8', ['model']),
        ]
    },
    "mileage": {
        "method": "mean",
        "decimals": 0,
        "hierarchy": [
            ('L4', ['year_bin', 'mpg_bin', 'model', 'tax_bin']),
            ('L5', ['year_bin', 'mpg_bin']),
            ('L6', ['year_bin']),
            ('L7', ['model']),
        ]
    },
    "previousOwners": {
        "method": "mean",
        "decimals": 0,
        "hierarchy": [
            ('L1', ['model'])
        ]
    }
}


# creating bins for imputations

def create_percentile_bins(df_input):
    
    """
    Creates decile bins for numeric columns using percentile-based discretization.
    
    - Creates 10 bins (deciles) for each specified numeric column
    - Handles duplicate values by dropping empty bins
    - Applies to year, engineSize, mpg, tax, and mileage columns
    - Generates bin columns for use in statistical grouping operations
    """
    
    df = df_input.copy()
    
    target_cols = ['year', 'engineSize', 'mpg', 'tax', 'mileage']
    
    for col in target_cols:
        if col in df.columns:
            bin_col_name = f"{col}_bin"
            
            # qcut for Percentiles (10 bins = Deciles)
            # duplicates='drop' merges bins if many values are identical
            df[bin_col_name] = pd.qcut(df[col], q=10, labels=False, duplicates='drop')
            
    return df

def check_bins(df, original_col):
    
    """
    Displays summary statistics for percentile bins of a numeric column.
    
    - Shows bin number, minimum, maximum, and count of values in each bin
    - Useful for verifying proper discretization of numeric variables
    - Helps identify empty or problematic bins in the binning process
    """
    bin_col = f"{original_col}_bin"
    if bin_col in df.columns:
        print(f"--- Percentile Bins for {original_col} ---")
        summary = df.groupby(bin_col)[original_col].agg(['min', 'max', 'count'])
        print(summary)
        print("\n")
  



# correcting numeric variables

def create_custom_bins(df_input):
    
    """
    Creates binned versions of numeric columns for hierarchical grouping.
    
    - Creates custom year bins with decade-based intervals
    - Generates engine size bins with specific cutoffs for common sizes
    - Applies decile binning to MPG, tax, and mileage columns
    - Adds bin columns for use in hierarchical imputation strategies
    """
    df = df_input.copy()
    
    #  1. Year Bins (10 Specific Intervals) 
    if 'year' in df.columns:
        # Edges define the boundaries. 
        # (1970 to 2000 is one bin, 2000 to 2005 is another, etc.)
        year_edges = [1970, 2000, 2005, 2010, 2013, 2015, 2016, 2017, 2018, 2019, 2021]
        
        df['year_bin'] = pd.cut(df['year'], bins=year_edges, labels=False, include_lowest=True)

    # 2. Engine Bins (10 Specific Intervals) 
    if 'engineSize' in df.columns:
        # Edges designed to isolate 1.0, 1.5/1.6, and 2.0
        # Note: 2021/10.0 are just high upper bounds to catch everything
        engine_edges = [0, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1, 2.5, 3.0, 10.0]
        
        df['engine_bin'] = pd.cut(df['engineSize'], bins=engine_edges, labels=False, include_lowest=True)

    # 3. MPG & Tax (Keeping Percentiles as they worked okay) 
    # We can refine these later if needed
    for col in ['mpg', 'tax', 'mileage']:
        if col in df.columns:
            df[f'{col}_bin'] = pd.qcut(df[col], q=10, labels=False, duplicates='drop')
            
    return df



def get_group_stats(df, group_cols, target_col, method='mode', min_samples=20, decimals=0):
    
    """
    Computes group-level statistics for hierarchical imputation.
    
    - Calculates mode or mean for target column within each group
    - Filters groups with insufficient samples (below min_samples)
    - Returns dictionary mapping group keys to statistical values
    - Handles rounding for mean calculations based on decimals parameter
    """
    
    # Group and count
    counts = df.groupby(group_cols, dropna=True)[target_col].count()
    
    # keep only groups with aat least minimum number of samples
    valid_groups = counts[counts >= min_samples].index
    
    # Filter data
    df_temp = df.set_index(group_cols)
    
    # Keep only rows belonging to valid groups
    df_valid = df_temp.loc[df_temp.index.isin(valid_groups)]
    
    # If no valid groups, return empty dictionary
    if df_valid.empty:
        return {}
        
    # Calculate based on method
    grouper = df_valid.groupby(level=list(range(len(group_cols))))[target_col]
    
    if method == 'mode':
        # Returns the single most frequent value
        stat_map = grouper.agg(lambda x: x.mode()[0])
    elif method == 'mean':
        # Returns the mean, rounded
        stat_map = grouper.mean().round(decimals)
    else:
        return {}
    
    return stat_map.to_dict()



def fit_nas(df_input, config=imputation_config):
    
    """
    Learns imputation rules for missing values using hierarchical statistical methods:
    - Creates model-to-brand mapping for hierarchical imputation
    - Generates 'cheat sheets' (mode/mean maps) for each config variable
    - Uses multi-level hierarchy with fallback to global defaults
    - Applies appropriate rounding based on column type and decimals setting
    - Builds safety net with overall column defaults
    """
    df = df_input.copy()
    rules = {}
    
    # Build mapping from model names to brands
    model_to_brand = {}
    if 'brand_model_dict' in globals():
        for brand, models in brand_model_dict.items():
            for model in models:
                model_to_brand[model.lower()] = brand      
        rules['brand_map'] = model_to_brand

    #   Prepare temporary binned columns for hierarchy calculations 
    df = create_custom_bins(df)
    
    for target_col, settings in config.items():
        if target_col not in df.columns:
            continue
            
        method = settings.get('method', 'mode')
        decimals = settings.get('decimals', 0)
        hierarchy = settings['hierarchy']
        
        variable_maps = {}
        # Generate mapping for each hierarchy level
        for level_name, group_cols in hierarchy:
            if set(group_cols).issubset(df.columns):
                mapping = get_group_stats(
                    df, group_cols, target_col, 
                    method=method, decimals=decimals, min_samples=20
                )
                if mapping:
                    variable_maps[level_name] = {'cols': group_cols, 'map': mapping}
        
        rules[target_col] = variable_maps
    
    # Fill remaining missing values with overall column defaults
    defaults = {}
    int_cols = ["year", "previousOwners"]

    for col in df.columns:
        if df[col].dtype == 'object' or df[col].dtype.name == 'category':
            # Use mode for categorical columns
             if not df[col].mode().empty:
                 defaults[col] = df[col].mode()[0]
             else:
                 defaults[col] = "Unknown"
                 
        elif pd.api.types.is_numeric_dtype(df[col]):
            # Use mean for numeric columns, rounded appropriately
             mean_val = df[col].mean()
             if col in int_cols:
                 defaults[col] = int(round(mean_val))
             else:
                 defaults[col] = round(mean_val, 1)
                 
    rules['defaults'] = defaults
    return rules


def transform_nas(df_input, rules):
    
    """
    Applies learned imputation rules to fill missing values in the DataFrame.
    
    - Fills Brand NAs using model-to-brand mapping
    - Uses hierarchical imputation for other variables (Model, Engine, etc.)
    - Applies multiple hierarchy levels with ordered fallback strategy
    - Removes temporary bin columns used for imputation
    - Fills remaining NAs with global defaults as safety net
    """
    df = df_input.copy()

    # 1. Fill Brand 
    if 'brand_map' in rules and 'Brand' in df.columns:
        if df['Brand'].isna().sum() > 0:
            predicted_brands = df['model'].str.lower().map(rules['brand_map'])
            df['Brand'] = df['Brand'].fillna(predicted_brands)

    #  2. Fill Others (Hierarchies) 
    df = create_custom_bins(df)
    
    # Exclude 'defaults' from this loop
    targets = [k for k in rules.keys() if k not in ['brand_map', 'defaults']]
    
    for i in range(1): # define how many times it will correct by the hierarchy order
        for target in targets:
            if target not in df.columns or df[target].isna().sum() == 0:
                continue
                
            maps = rules[target]
            sorted_levels = sorted(maps.keys())
            
            for level in sorted_levels:
                if df[target].isna().sum() == 0:
                    break 
                
                cols = maps[level]['cols']
                mapping_dict = maps[level]['map']
                
                if set(cols).issubset(df.columns):
                    predictions = df.set_index(cols).index.map(mapping_dict)
                    fill_series = pd.Series(predictions, index=df.index)
                    df[target] = df[target].fillna(fill_series)

    #  3 Cleanup Bins 
    bin_cols = [c for c in df.columns if c.endswith('_bin')]
    df = df.drop(columns=bin_cols)

    # 4 Safety Net 
    if 'defaults' in rules:
        defaults = rules['defaults']
        for col in df.columns:
            if col in defaults and df[col].isna().sum() > 0:
                df[col] = df[col].fillna(defaults[col])

    return df


# -------------------------------------------------------
# Feature Engineering
# -------------------------------------------------------


def fit_feat_engineering_rules(df_input: pd.DataFrame, min_group_n: int = 25) -> dict:
    
    """
    Learns statistics for feature engineering from training data.
    
    - Computes global statistics: max year, median MPG, median miles per year
    - Builds usage table for mileage normalization with fallback logic
    - Uses min_group_n threshold for reliable group statistics
    - Returns dictionary of rules for consistent feature engineering
    """
    df = df_input.copy()
    rules = {}
    
    # Global rules
    rules["min_group_n"] = min_group_n
    
    # Maximum vehicle year
    if "year" in df.columns:
        rules["max_year"] = df["year"].max()
    else:
        rules["max_year"] = None

    # Median MPG value
    if "mpg" in df.columns:
        rules["mpg_median"] = df["mpg"].median()
    else:
        rules["mpg_median"] = None

    # Initialize usage table and global median
    usage_table = None
    global_miles_median = None
    
    # Compute miles_per_year statistics if both year and mileage exist
    if all(c in df.columns for c in ["year", "mileage"]):
        max_year = rules["max_year"]
        tmp = df.copy()
        # Compute car age 
        tmp["car_age"] = (max_year - tmp["year"]).clip(lower=0)
        # Compute miles driven per year
        tmp["miles_per_year"] = tmp["mileage"] / (tmp["car_age"] + 1)
        # Global median miles per year
        global_miles_median = tmp["miles_per_year"].median()
        
        # Compute group-level statistics for Brand/Model/Year
        if all(c in tmp.columns for c in ["Brand", "model", "year"]):
            group = ["Brand", "model", "year"]
            # Count of cars in each group
            group_size  = tmp.groupby(group)["miles_per_year"].size()
            # Median miles_per_year per group
            median_year = tmp.groupby(group)["miles_per_year"].median()
            # Median miles_per_year per Brand/Model
            median_model = tmp.groupby(["Brand", "model"])["miles_per_year"].median()
            
            # combine into a single usage table
            usage_table = pd.DataFrame({
                "group_size": group_size,
                "median_year": median_year,
            })
            
            # Join median_model to the table
            usage_table = usage_table.join(
                median_model.rename("median_model"),
                on=["Brand", "model"]
            )
            
            # Use group-level median if large enough, else fallback to model-level median
            usage_table["median_final"] = np.where(
                usage_table["group_size"] >= min_group_n,
                usage_table["median_year"],
                usage_table["median_model"]
            )
            
            # Keep only the final median column
            usage_table = usage_table[["median_final"]]
    
    # Store usage tables and global median in rules
    rules["usage_table"] = usage_table
    rules["global_miles_median"] = global_miles_median


    return rules



def transform_add_all_features(df_input: pd.DataFrame, rules: dict) -> pd.DataFrame:
    
    """
    Applies feature engineering rules to create new derived features.
    
    - Creates car age and annual mileage features from year and mileage
    - Adds efficiency ratios combining engine size, MPG, and tax
    - Computes usage deviation from expected mileage patterns
    - Adds model popularity and specialization scores
    - Generates usage metrics
    """
    df = df_input.copy()

    max_year      = rules.get("max_year", None)
    mpg_median    = rules.get("mpg_median", None)
    min_group_n   = rules.get("min_group_n", 25)
    usage_table   = rules.get("usage_table", None)
    global_m_mpy  = rules.get("global_miles_median", None)
    model_pop     = rules.get("model_popularity", None)

    # car_age 
    if max_year is not None and "year" in df.columns:
        df["car_age"] = (max_year - df["year"]).clip(lower=0)

    #  miles_per_year 
    if all(c in df.columns for c in ["mileage", "car_age"]):
        df["miles_per_year"] = df["mileage"] / (df["car_age"] + 1)

    #  tax_per_litre_engine 
    if all(c in df.columns for c in ["tax", "engineSize"]):
        df["tax_per_litre_engine"] = df["tax"] / df["engineSize"]

    #  power_efficiency_ratio 
    if all(c in df.columns for c in ["engineSize", "mpg"]):
        df["power_efficiency_ratio"] = df["engineSize"] / df["mpg"]

    #  tax_efficiency 
    if "tax_per_litre_engine" in df.columns and "mpg" in df.columns:
        df["tax_efficiency"] = df["tax_per_litre_engine"] / df["mpg"]

    # luxury_tax_burden
    if all(c in df.columns for c in ["tax", "engineSize"]):
        df["luxury_tax_burden"] = np.log1p(df["tax"] * df["engineSize"])

    # engine_specialization_score + is_high_efficiency 
    if all(c in df.columns for c in ["engineSize", "mpg"]):
        eng_pct  = df["engineSize"].rank(pct=True)
        l100_pct = df["mpg"].rank(pct=True)
        df["engine_specialization_score"] = eng_pct - l100_pct

    if "mpg" in df.columns and mpg_median is not None:
        cond_eff = df["mpg"] < mpg_median * 0.7
        if "fuelType" in df.columns:
            cond_eff |= (
                df["fuelType"]
                .astype(str)
                .str.contains("ELECTRIC|HYBRID", case=False, na=False)
        )

        df["is_high_efficiency"] = cond_eff.astype(int)

    # usage_dev_vs_cohort 
    if (
        usage_table is not None
        and all(c in df.columns for c in ["Brand", "model", "year"])
        and "miles_per_year" in df.columns
    ):
        
        df = df.merge(
            usage_table["median_final"].rename("usage_median_final"),
            how="left",
            left_on=["Brand", "model", "year"],
            right_index=True,
        )
        if global_m_mpy is not None:
            df["usage_median_final"] = df["usage_median_final"].fillna(global_m_mpy)
        df["usage_dev_vs_cohort"] = df["miles_per_year"] - df["usage_median_final"]
        df.drop(columns=["usage_median_final"], inplace=True)

    # model_popularity 
    if (
        model_pop is not None 
        and all(c in df.columns for c in ["Brand", "model"])
    ):
        df = df.merge(
            model_pop.rename("model_popularity"),
            how="left",
            left_on=["Brand", "model"],
            right_index=True,
        )       

    return df


# -------------------------------------------------------
# Encodings: OHE, Ordinal Encoder and Target Encoder
# -------------------------------------------------------


# One Hot Encoder
def fit_ohe(df_input):
    
    """
    Learns one-hot encoding rules for all categorical columns.
    
    - Automatically detects categorical columns in the DataFrame
    - Fits a scikit-learn OneHotEncoder with first category dropping
    - Handles unknown categories by ignoring them during transformation
    - Uses boolean dtype for efficient memory usage
    - Returns encoding rules including fitted encoder and column list
    """
    
    rules = {}
    df = df_input.copy() 

    # 1. Find categorical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # 2. Fit the OneHotEncoder 
    # Using the settings from your previous OHE function
    ohe = OneHotEncoder(drop='first', 
                        sparse_output=False, 
                        handle_unknown='ignore', 
                        dtype=bool)
    ohe.fit(df[categorical_cols])
    
    #  3. Save rules 
    rules['encoder'] = ohe
    rules['categorical_cols'] = categorical_cols
    
    return rules


def transform_ohe(df_input, rules):
    
    """
    Applies pre-fitted one-hot encoding rules to transform categorical variables.
    
    - Transforms categorical columns into binary one-hot encoded features
    - Drops first category to avoid multicollinearity
    - Preserves original numerical columns unchanged
    - Handles unknown categories by ignoring them
    - Returns DataFrame with categorical columns replaced by binary encoded features
    """
    df = df_input.copy()

    # 1. Load rules 
    ohe = rules['encoder']
    categorical_cols = rules['categorical_cols']
    
    # 2. Find numerical columns (from the current df) 
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # 3. Transform categorical columns 
    encoded = ohe.transform(df[categorical_cols])
    encoded_cols = ohe.get_feature_names_out(categorical_cols)
    encoded_df = pd.DataFrame(encoded, columns=encoded_cols, index=df.index)

    # 4. Re-combine with numerical columns 
    # This implicitly drops the original categorical columns
    final_df = pd.concat([df[numerical_cols], encoded_df], axis=1)
    
    return final_df


# Ordinal Encoder
def fit_oe(df_input):
    
    """
    Learns ordinal encoding rules for all categorical columns.
    
    - Automatically detects categorical columns in the DataFrame
    - Fits a scikit-learn OrdinalEncoder with unknown value handling
    - Returns encoding rules including fitted encoder and column list
    - Handles unseen categories by encoding them as -1
    """
    rules = {}
    df = df_input.copy() 

    # 1. Find categorical columns 
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    #  2. Fit the OrdinalEncoder 
    # We use 'use_encoded_value' and -1 to handle unknown values
    oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    oe.fit(df[categorical_cols])
    
    #  3. Save rules
    rules['encoder'] = oe
    rules['categorical_cols'] = categorical_cols
    
    return rules

def transform_oe(df_input, rules):
    
    """
    Applies pre-fitted ordinal encoding rules to transform categorical variables.
    
    - Encodes categorical columns to numerical values using fitted OrdinalEncoder
    - Preserves original numerical columns unchanged
    - Handles unseen categories by encoding them as -1
    - Returns DataFrame with categorical columns replaced by integer encodings
    - Maintains original DataFrame index for alignment
    """
    
    df = df_input.copy()

    #  1. Load rules 
    oe = rules['encoder']
    categorical_cols = rules['categorical_cols']
    
    #  2. Find numerical columns (from the current df) 
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    #  3. Transform categorical columns 
    # This replaces the string columns with number columns
    encoded = oe.transform(df[categorical_cols])
    encoded_df = pd.DataFrame(encoded, 
                              columns=categorical_cols, 
                              index=df.index,
                              dtype=int) # Ensure output is integer

    #  4. Re-combine with numerical columns
    final_df = pd.concat([df[numerical_cols], encoded_df], axis=1)
    
    return final_df


# Target Encoder
def fit_target_encoder(df_input, y, m=10):
    
    """
    Learns target encodings for categorical variables using regularization.
    
    - Automatically detects all categorical columns in the DataFrame
    - Creates smoothed encodings using target variable `y`
    - Applies regularization controlled by `m` parameter
    - Stores global mean as fallback for unseen categories
    - Returns encoding rules as dictionary for later transformation
    """
    # 1. Setup
    df = df_input.copy()
    y_name = y.name if hasattr(y, 'name') else 'target'
    
    # Internal temporary merge to calculate statistics
    # We join on index to ensure rows match correctly
    df_temp = pd.concat([df, y], axis=1)
    
    # 2. Global Mean (The safety net)
    global_mean = y.mean()
    
    # 3. Auto-detect categorical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    

    # 4. Calculate Rules
    rules = {
        'global_mean': global_mean,
        'categorical_cols': categorical_cols,
        'maps': {}
    }
    
    for col in categorical_cols:
        # Calculate Group Statistics
        agg = df_temp.groupby(col)[y_name].agg(['count', 'mean'])
        counts = agg['count']
        means = agg['mean']
        
        # Smoothing (prevents overfitting on small groups)
        smoothing = 1 / (1 + np.exp(-(counts - m)))
        
        # Calculate Final Value
        df_map = (means * smoothing) + (global_mean * (1 - smoothing))
        rules['maps'][col] = df_map.to_dict()
    
    return rules

def transform_target_encoder(df_input, rules):
    
    """
    Applies learned target encodings to transform categorical variables.
    
    - Maps categorical values to pre-computed numerical encodings
    - Uses global mean as fallback for unseen or missing categories
    - Converts encoded columns to float type for modeling
    - Returns DataFrame with categorical columns replaced by numerical encodings
    """
    df = df_input.copy()
    
    maps = rules['maps']
    global_mean = rules['global_mean']
    cat_cols = rules['categorical_cols']
    
    for col in cat_cols:
        if col in df.columns:
            # Map values. If a value is unknown (NaN after map), fill with global_mean
            df[col] = df[col].map(maps[col]).fillna(global_mean)
            
            # Convert to float to be ready for Scaling/Neural Network
            df[col] = df[col].astype(float)
            
    return df

# -------------------------------------------------------
# Scaling
# -------------------------------------------------------

def fit_scaler(df_input, target_col=None):
    
    """
    Fits a RobustScaler to numeric columns for robust feature scaling.
    
    - Automatically detects all numeric columns in the DataFrame
    - Fits RobustScaler to handle outliers in training data
    - Returns scaler and column list for consistent transformation
    - Optionally excludes target column (if specified in future implementation)
    """
    
    
    df = df_input.copy()
    
    # Select all numeric columns for scaling
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Initialize and fit the RobustScaler
    scaler = RobustScaler()
    scaler.fit(df[num_cols])

    # Store the fitted scaler and the list of numeric columns
    rules = {
        "scaler": scaler,
        "num_cols": num_cols
    }
    return rules


def transform_scaler(df_input, rules):
    
    """
    Applies pre-fitted RobustScaler to transform numeric columns.
    
    - Scales numeric columns using previously fitted RobustScaler
    - Uses median and IQR for outlier-resistant transformation
    - Returns DataFrame with scaled numeric features
    - Maintains original column names and structure
    """
    df = df_input.copy()
    scaler = rules["scaler"]
    num_cols = rules["num_cols"]

    df[num_cols] = scaler.transform(df[num_cols])
    return df


# --------------------------------------------------------------
# Feature Selection
# -------------------------------------------------------


DROP_COLS = [
    "is_high_efficiency",
    "previousOwners",
    "luxury_tax_burden",
    "tax",
    "usage_dev_vs_cohort",
    "miles_per_year",
    "year",
]


def feat_selection(df_input, drop_cols=DROP_COLS):
    """
    Drops a predefined set of features from the dataset.
    - Removes columns listed in drop_cols (safe if some are missing).
    - Helps enforce consistent feature selection after feature engineering.

    Returns:
    - A copy of the DataFrame without the dropped columns.
    """
    df = df_input.copy()
    return df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")


# ==============================================================
# Models
# ==============================================================

# -------------------------------------------------------
# Linear Models → 9 functions
# -------------------------------------------------------

def fit_processing_rules_Lin(X_input):
    """
    Learns all statistical rules based ONLY on the training set.
    Returns a transformed dataset 'X_t' and a 'rules' dictionary to be used later.
    """
    X_t = X_input.copy()
    rules = {}

    # --- A. IMPUTATION (Fit) ---
    # Learn means/medians for missing values
    rules['nas'] = fit_nas(X_t)
    X_t = transform_nas(X_t, rules['nas']) # Must apply here so the next steps work

    # --- B. FEATURE ENGINEERING (Fit) ---
    # Learn groupings or frequencies
    rules['feat'] = fit_feat_engineering_rules(X_t, min_group_n=25)
    X_t = transform_add_all_features(X_t, rules['feat'])
    
    
    # --- C. SCALING (Fit) ---
    # Learn mean and standard deviation for normalization
    rules['scaler'] = fit_scaler(X_t)
    X_t = transform_scaler(X_t, rules['scaler'])
    
    

    # --- D. ONE-HOT ENCODING (Fit) ---
    # Map categories to target means
    rules['encoder'] = fit_ohe(X_t)
    X_t = transform_ohe(X_t, rules['encoder'])    


    # --- E. FEATURE SELECTION (Save Final Columns) ---
    # Apply selection on training data to identify WHICH columns remain
    X_t = feat_selection(X_t)

    return X_t, rules



def transform_processing_rules_Lin(df_input, rules):
    """
    Applies all learned rules to a new dataset (validation or test).
    If the dataset is the test set, it will have the "carID" column and will need basic cleaning first.
    Returns the transformed DataFrame.
    """
    df = df_input.copy()
    if "carID" in df.columns:
        df = clean_basic_data(df)
    df = transform_nas(df, rules['nas'])
    df = transform_add_all_features(df, rules['feat'])      
    df = transform_scaler(df, rules['scaler'])
    df = transform_ohe(df, rules['encoder'])
    df = feat_selection(df)
    return df



def evaluate_ols(lin_model, X_train_input, y_train_input, X_val_input, y_val_input, return_objects=False):
    """
    Accepts a LinearRegression-like model and train/validation splits.
    - Fits the model on the training set.
    - Predicts on training and validation sets.
    - Computes MAE and R² for both splits.
    - Builds a validation results DataFrame with (y_true, y_pred).
    - Prints intercept and coefficients (when feature names are available).

    Returns:
    - If return_objects=False: prints metrics and coefficients.
    - If return_objects=True: returns a dict with metrics, predictions, and coefficient tables.
    """
    lin_model.fit(X_train_input, y_train_input)

    y_pred_val = lin_model.predict(X_val_input)
    val_results = pd.DataFrame(
        {
            "y_true": np.asarray(y_val_input).ravel(),
            "y_pred": np.asarray(y_pred_val).ravel(),
        },
        index=getattr(y_val_input, "index", None),
    )

    y_pred_train = lin_model.predict(X_train_input)

    r2_train  = r2_score(y_train_input, y_pred_train)
    mae_train = mean_absolute_error(y_train_input, y_pred_train)
    r2_val  = r2_score(y_val_input, y_pred_val)
    mae_val = mean_absolute_error(y_val_input, y_pred_val)

    print("OLS Regression Results (sklearn only):")
    print("Training metrics:")
    print(f"  R²:  {r2_train:.4f}")
    print(f"  MAE: {mae_train:.4f}")

    print("\nValidation metrics:")
    print(f"  R²:  {r2_val:.4f}")
    print(f"  MAE: {mae_val:.4f}")

    coef_array = np.asarray(lin_model.coef_).ravel()
    coef_df = pd.DataFrame(coef_array, index=X_train_input.columns, columns=["coef"])

    sk_coef = np.concatenate(([lin_model.intercept_], coef_array))
    sk_coef_series = pd.Series(sk_coef, index=["Intercept"] + list(X_train_input.columns))

    print("\nSklearn intercept and coefficients:")
    print(sk_coef_series)

    if return_objects:
        return {
            "val_results": val_results,
            "y_pred_train": y_pred_train,
            "y_pred_val": y_pred_val,
            "r2_train": r2_train,
            "mae_train": mae_train,
            "r2_val": r2_val,
            "mae_val": mae_val,
            "coef_df": coef_df,
            "sk_coef": sk_coef_series,
        }




def tune_ridge_holdout(X_train_input, y_train_input, X_val_input, y_val_input,
                       n_iter=25, alpha_min=1e-4, alpha_max=20.0, seed=42):
    """
    Accepts train/validation splits and performs fast holdout tuning for Ridge.
    - Randomly samples `alpha` for `n_iter` trials.
    - Fits a Ridge model per trial on the training set.
    - Selects the best trial by minimum validation MAE (lower is better).
    - Returns the best fitted model, best parameters, and train/val MAE summary.

    Steps:
    1) Convert inputs to numpy arrays (reduces overhead).
    2) Sample alpha ~ Uniform[alpha_min, alpha_max].
    3) Fit Ridge, compute MAE on train and validation.
    4) Keep the model with the lowest validation MAE.
    """
    rng = np.random.default_rng(seed)

    Xtr = np.asarray(X_train_input, dtype=np.float32)
    Xva = np.asarray(X_val_input,   dtype=np.float32)
    ytr = np.asarray(y_train_input, dtype=np.float64).ravel()
    yva = np.asarray(y_val_input,   dtype=np.float64).ravel()

    best_model, best_alpha = None, None
    best_mae_val, best_mae_tr = np.inf, np.inf

    for _ in range(int(n_iter)):
        alpha = float(rng.uniform(alpha_min, alpha_max))
        if alpha <= 0:
            alpha = 1e-12

        m = Ridge(alpha=alpha).fit(Xtr, ytr)

        pred_tr = m.predict(Xtr)
        pred_va = m.predict(Xva)

        mae_tr = float(np.mean(np.abs(ytr - pred_tr)))
        mae_va = float(np.mean(np.abs(yva - pred_va)))

        if mae_va < best_mae_val:
            best_model, best_alpha = m, alpha
            best_mae_val, best_mae_tr = mae_va, mae_tr

    return best_model, {"alpha": best_alpha}, {"mae_train": best_mae_tr, "mae_val": best_mae_val}


def evaluate_ridge(X_train_input, y_train_input, X_val_input, y_val_input, alpha, feature_names=None):
    """
    Accepts train/validation splits and evaluates a Ridge model with a fixed alpha.
    - Fits Ridge(alpha) on the training set.
    - Predicts on training and validation sets.
    - Computes MAE and R² for both splits.
    - Optionally builds a coefficient DataFrame if feature names are available.

    Returns:
    - A dict with the fitted model, train/val predictions, MAE/R², and coefficient table (if possible).
    """
    Xtr = np.asarray(X_train_input, dtype=np.float32)
    Xva = np.asarray(X_val_input,   dtype=np.float32)
    ytr = np.asarray(y_train_input, dtype=np.float64).ravel()
    yva = np.asarray(y_val_input,   dtype=np.float64).ravel()

    model = Ridge(alpha=float(alpha)).fit(Xtr, ytr)

    y_pred_train = model.predict(Xtr)
    y_pred_val   = model.predict(Xva)

    r2_train  = r2_score(ytr, y_pred_train)
    r2_val    = r2_score(yva, y_pred_val)
    mae_train = mean_absolute_error(ytr, y_pred_train)
    mae_val   = mean_absolute_error(yva, y_pred_val)

    coef_array = model.coef_

    if feature_names is None and hasattr(X_train_input, "columns"):
        feature_names = list(X_train_input.columns)

    coef_df = None
    if feature_names is not None:
        coef_df = pd.DataFrame({"feature": feature_names, "coef": coef_array}).set_index("feature")

    results = {
        "model": model,
        "y_pred_train": y_pred_train,
        "y_pred_val": y_pred_val,
        "r2_train": r2_train,
        "r2_val": r2_val,
        "mae_train": mae_train,
        "mae_val": mae_val,
        "coef_df": coef_df,
        "n_features_nonzero": int(np.sum(coef_array != 0)),
    }
    return results


def tune_lasso_holdout(
    X_train, y_train, X_val, y_val,
    n_iter=1, alpha_min=1e-6, alpha_max=2.0, seed=42,
    train_sub=30000, val_sub=8000,
    max_iter_search=1500, tol_search=2e-3,
    max_iter_final=6000, tol_final=1e-3,
    selection="cyclic"
):
    """
    Fast holdout tuning for Lasso on large datasets.

    Tunes alpha on train/val *subsamples* (faster) using log-uniform alpha sampling,
    then refits one final Lasso on the full training set with the best alpha and
    reports MAE on full train and full validation.

    Returns: (final_model, {"alpha": best_alpha}, {"mae_train": ..., "mae_val": ...})
    """
    rng = np.random.default_rng(seed)

    Xtr = np.asarray(X_train, dtype=np.float32)
    Xva = np.asarray(X_val,   dtype=np.float32)
    ytr = np.asarray(y_train, dtype=np.float64).ravel()
    yva = np.asarray(y_val,   dtype=np.float64).ravel()

    if train_sub is not None and train_sub < len(ytr):
        idx_tr = rng.choice(len(ytr), size=int(train_sub), replace=False)
        Xtr_s, ytr_s = Xtr[idx_tr], ytr[idx_tr]
    else:
        Xtr_s, ytr_s = Xtr, ytr

    if val_sub is not None and val_sub < len(yva):
        idx_va = rng.choice(len(yva), size=int(val_sub), replace=False)
        Xva_s, yva_s = Xva[idx_va], yva[idx_va]
    else:
        Xva_s, yva_s = Xva, yva

    a_min = max(float(alpha_min), 1e-12)
    a_max = max(float(alpha_max), a_min * 10.0)
    alphas = 10 ** rng.uniform(np.log10(a_min), np.log10(a_max), size=int(n_iter))

    best_alpha, best_mae = None, np.inf

    for a in alphas:
        m = Lasso(alpha=float(a), max_iter=max_iter_search, tol=tol_search,
                  selection=selection).fit(Xtr_s, ytr_s)
        mae = float(np.mean(np.abs(yva_s - m.predict(Xva_s))))
        if mae < best_mae:
            best_mae, best_alpha = mae, float(a)

    final = Lasso(alpha=float(best_alpha), max_iter=max_iter_final, tol=tol_final,
                  selection=selection).fit(Xtr, ytr)

    mae_tr = float(np.mean(np.abs(ytr - final.predict(Xtr))))
    mae_va = float(np.mean(np.abs(yva - final.predict(Xva))))

    return final, {"alpha": float(best_alpha)}, {"mae_train": mae_tr, "mae_val": mae_va}

def evaluate_lasso(model, X_train_input, y_train_input, X_val_input, y_val_input, return_objects=False):
    
    """
    Accepts train/validation splits and evaluates a Lasso model with a fixed alpha.
    - Fits the provided Lasso model on the training set.
    - Predicts on training and validation sets.
    - Computes MAE and R² for both splits.
    - Optionally returns coefficient table if feature names are available.

    Returns:
    - A dict with fitted model outputs, predictions, MAE/R², and coefficients (if possible).
    """

    model.fit(X_train_input, y_train_input)

    y_pred_val = model.predict(X_val_input)
    y_pred_train = model.predict(X_train_input)

    r2_train  = r2_score(y_train_input, y_pred_train)
    mae_train = mean_absolute_error(y_train_input, y_pred_train)
    r2_val    = r2_score(y_val_input, y_pred_val)
    mae_val   = mean_absolute_error(y_val_input, y_pred_val)

    coef_array = np.asarray(model.coef_).ravel()
    coef_df = pd.DataFrame(coef_array, index=X_train_input.columns, columns=["coef"])
    sk_coef = np.concatenate(([model.intercept_], coef_array))
    sk_coef_series = pd.Series(sk_coef, index=["Intercept"] + list(X_train_input.columns))
    
    
    if return_objects:
        return {
            "y_pred_train": y_pred_train,
            "y_pred_val": y_pred_val,
            "r2_train": r2_train,
            "mae_train": mae_train,
            "r2_val": r2_val,
            "mae_val": mae_val,
            "coef_df": coef_df,
            "sk_coef": sk_coef_series,
        }





def tune_enet_holdout(X_train, y_train, X_val, y_val,
                           n_iter=25, alpha_min=1e-6, alpha_max=3.0,
                           l1_min=0.10, l1_max=0.90, seed=42,
                           train_sub=30000, val_sub=7000,
                           max_iter_search=900, tol_search=3e-3,
                           max_iter_final=6000, tol_final=1e-3):
    """
    Accepts train/validation splits and performs fast holdout tuning for ElasticNet.
    - Samples `alpha` on a log scale (log-uniform) and `l1_ratio` on a uniform scale.
    - Uses a train subsample (`train_sub`) and validation subsample (`val_sub`) to speed up scoring during search.
    - Runs a quick search phase with loose convergence settings (max_iter_search, tol_search).
    - Refits once on the full training set with stricter settings (max_iter_final, tol_final).

    Steps:
    1) Convert inputs to numpy arrays to reduce overhead.
    2) Optionally subsample train/validation sets for fast MAE evaluation during search.
    3) Draw `n_iter` (alpha, l1_ratio) pairs and score each by validation MAE (subsample).
    4) Refit a final ElasticNet model using the best pair on full training data.
    5) Return final model, best parameters, and full train/val MAE.
    """
    rng = np.random.default_rng(seed)

    Xtr = np.asarray(X_train, np.float32); Xva = np.asarray(X_val, np.float32)
    ytr = np.asarray(y_train, np.float64).ravel(); yva = np.asarray(y_val, np.float64).ravel()

    # fast scoring on subsamples
    if train_sub is not None and train_sub < len(ytr):
        idx_tr = rng.choice(len(ytr), size=int(train_sub), replace=False)
        Xtr_s, ytr_s = Xtr[idx_tr], ytr[idx_tr]
    else:
        Xtr_s, ytr_s = Xtr, ytr

    if val_sub is not None and val_sub < len(yva):
        idx_va = rng.choice(len(yva), size=int(val_sub), replace=False)
        Xva_s, yva_s = Xva[idx_va], yva[idx_va]
    else:
        Xva_s, yva_s = Xva, yva

    alphas = 10 ** rng.uniform(np.log10(max(alpha_min, 1e-12)), np.log10(alpha_max), size=int(n_iter))
    l1s    = rng.uniform(l1_min, l1_max, size=int(n_iter))

    best_alpha, best_l1, best_mae = None, None, np.inf
    for a, l1 in zip(alphas, l1s):
        m = ElasticNet(alpha=float(a), l1_ratio=float(l1),
                       max_iter=max_iter_search, tol=tol_search,
                       selection="random", random_state=seed).fit(Xtr_s, ytr_s)
        mae = float(np.mean(np.abs(yva_s - m.predict(Xva_s))))
        if mae < best_mae:
            best_mae, best_alpha, best_l1 = mae, float(a), float(l1)

    # final refit on full train, evaluate on full val
    final = ElasticNet(alpha=best_alpha, l1_ratio=best_l1,
                       max_iter=max_iter_final, tol=tol_final,
                       selection="cyclic", random_state=seed).fit(Xtr, ytr)

    mae_tr = float(np.mean(np.abs(ytr - final.predict(Xtr))))
    mae_va = float(np.mean(np.abs(yva - final.predict(Xva))))
    return final, {"alpha": best_alpha, "l1_ratio": best_l1}, {"mae_train": mae_tr, "mae_val": mae_va}



def evaluate_elasticnet(model, X_train_input, y_train_input, X_val_input, y_val_input, return_objects=False):
    model.fit(X_train_input, y_train_input)
    """
    Accepts train/validation splits and evaluates an ElasticNet model with fixed (alpha, l1_ratio).
    - Fits ElasticNet(alpha, l1_ratio) on the training set.
    - Predicts on training and validation sets.
    - Computes MAE and R² for both splits.
    - Optionally returns a coefficient DataFrame if feature names are available.

    Returns:
    - A dict with fitted model, predictions, MAE/R², and coefficient table (if possible).
    """

    y_pred_val = model.predict(X_val_input)
    val_results = pd.DataFrame(
        {
            "y_true": np.asarray(y_val_input).ravel(),
            "y_pred": np.asarray(y_pred_val).ravel(),
        },
        index=getattr(y_val_input, "index", None),
    )

    y_pred_train = model.predict(X_train_input)

    r2_train  = r2_score(y_train_input, y_pred_train)
    mae_train = mean_absolute_error(y_train_input, y_pred_train)
    r2_val    = r2_score(y_val_input, y_pred_val)
    mae_val   = mean_absolute_error(y_val_input, y_pred_val)

    print("Elastic Net Regression Results (sklearn):")
    print("Training metrics:")
    print(f"  R²:  {r2_train:.4f}")
    print(f"  MAE: {mae_train:.4f}")

    print("\nValidation metrics:")
    print(f"  R²:  {r2_val:.4f}")
    print(f"  MAE: {mae_val:.4f}")

    coef_array = np.asarray(model.coef_).ravel()
    coef_df = pd.DataFrame(coef_array, index=X_train_input.columns, columns=["coef"])

    sk_coef = np.concatenate(([model.intercept_], coef_array))
    sk_coef_series = pd.Series(sk_coef, index=["Intercept"] + list(X_train_input.columns))

    print("\nSklearn intercept and coefficients:")
    print(sk_coef_series)

    if return_objects:
        return {
            "val_results": val_results,
            "y_pred_train": y_pred_train,
            "y_pred_val": y_pred_val,
            "r2_train": r2_train,
            "mae_train": mae_train,
            "r2_val": r2_val,
            "mae_val": mae_val,
            "coef_df": coef_df,
            "sk_coef": sk_coef_series,
        }


# -------------------------------------------------------
# K-Nearest Neighbors → 2 functions
# -------------------------------------------------------


def fit_processing_rules_Knn(X_input):
    """
    Learns all statistical rules based ONLY on the training set.
    Returns a transformed dataset 'X_t' and a 'rules' dictionary to be used later.
    """
    X_t = X_input.copy()
    rules = {}

    # --- A. IMPUTATION (Fit) ---
    # Learn means/medians for missing values
    rules['nas'] = fit_nas(X_t)
    X_t = transform_nas(X_t, rules['nas']) # Must apply here so the next steps work

    # --- B. FEATURE ENGINEERING (Fit) ---
    # Learn groupings or frequencies
    rules['feat'] = fit_feat_engineering_rules(X_t, min_group_n=25)
    X_t = transform_add_all_features(X_t, rules['feat'])
    
    # --- E. FEATURE SELECTION (Save Final Columns) ---
    # Apply selection on training data to identify WHICH columns remain
    X_t = feat_selection(X_t)

    # --- C. SCALING (Fit) ---
    # Learn mean and standard deviation for normalization
    rules['scaler'] = fit_scaler(X_t)
    X_t = transform_scaler(X_t, rules['scaler'])
    
    

    # --- D. ONE-HOT ENCODING (Fit) ---
    # Map categories to target means
    rules['encoder'] = fit_ohe(X_t)
    X_t = transform_ohe(X_t, rules['encoder'])      

    return X_t, rules


def transform_processing_rules_Knn(df_input, rules):
    """
    Applies all learned rules to a new dataset (validation or test).
    If the dataset is the test set, it will have the "carID" column and will need basic cleaning first.
    Returns the transformed DataFrame.
    """
    df = df_input.copy()
    if "carID" in df.columns:
        df = clean_basic_data(df)
    df = transform_nas(df, rules['nas'])
    df = transform_add_all_features(df, rules['feat'])      
    df = transform_scaler(df, rules['scaler'])
    df = transform_ohe(df, rules['encoder'])
    
    df = feat_selection(df)
    return df


# -------------------------------------------------------
# Neural Networks → 4 functions
# -------------------------------------------------------


def fit_processing_rules_nn(X_input, y_input):
    """
    Learns all statistical rules based on the X_input set (train or full).
    Transforms X_input step-by-step, to ensure each step has the previous rules applied.
    Returns the X_input set updated as X_t and a 'rules' dictionary to be used later (validation or test).
    """
    X_t = X_input.copy()
    y_t = y_input.copy()
    rules = {}

    # --- A. IMPUTATION (Fit) ---
    # Learn means/medians for missing values
    rules['nas'] = fit_nas(X_t)
    X_t = transform_nas(X_t, rules['nas']) # Must apply here so the next steps work

    # --- B. FEATURE ENGINEERING (Fit) ---
    # Learn groupings or frequencies
    rules['feat'] = fit_feat_engineering_rules(X_t, min_group_n=25)
    X_t = transform_add_all_features(X_t, rules['feat'])

    # --- C. TARGET ENCODING (Fit) ---
    # Map categories to target means
    rules['encoder'] = fit_target_encoder(X_t, y_t)
    X_t = transform_target_encoder(X_t, rules['encoder'])

    # --- D. SCALING (Fit) ---
    # Learn mean and standard deviation for normalization
    rules['scaler'] = fit_scaler(X_t)
    X_t = transform_scaler(X_t, rules['scaler'])

    # --- E. FEATURE SELECTION (Save Final Columns) ---
    # Apply selection on training data to identify WHICH columns remain
    X_t = feat_selection(X_t)

    return X_t, rules

def transform_processing_rules_nn(df_input, rules):
    """
    Applies all learned rules to a new dataset (validation or test).
    If the dataset is the test set, it will have the "carID" column and will need basic cleaning first.
    Returns the transformed DataFrame.
    """
    df = df_input.copy()
    if "carID" in df.columns:
        df = clean_basic_data(df)
    df = transform_nas(df, rules['nas'])
    df = transform_add_all_features(df, rules['feat'])
    df = transform_target_encoder(df, rules['encoder'])
    df = transform_scaler(df, rules['scaler'])
    df = feat_selection(df)
    return df


def avg_score_nn(model, X_t, y_t, X_v, y_v):
    """
    Fits a single model, tracks performance metrics (MAE & R2), and handles iteration counting.

    1. Starts a timer and fits the model to the training data.
    2. Generates predictions for both training and validation sets.
    3. Calculates MAE and R2 for both sets.
    4. Extracts the number of iterations.
    5. Returns formatted metrics: (Time, Train MAE, Val MAE, Train R2, Val R2, Iterations).
    """

    # Timer and Fit
    begin = time.perf_counter()
    model.fit(X_t, y_t) # Train on pre-processed data
    end = time.perf_counter()
    
    # Predictions
    pred_train = model.predict(X_t)
    pred_val = model.predict(X_v)
    
    # Calculate MAE
    mae_train = mean_absolute_error(y_t, pred_train)
    mae_val = mean_absolute_error(y_v, pred_val)

    # Calculate R2 
    r2_train = r2_score(y_t, pred_train)
    r2_val = r2_score(y_v, pred_val)
    
    # Capture Iterations 
    n_iter = 0
    if hasattr(model, "n_iter_"):
        n_iter = model.n_iter_
    elif hasattr(model, "regressor_") and hasattr(model.regressor_, "n_iter_"):
        n_iter = model.regressor_.n_iter_
        
    # Format Output
    time_res = round(end - begin, 3)
    mae_train_str = round(mae_train, 2)
    mae_val_str = round(mae_val, 2)
    r2_train_str = round(r2_train, 4)  # Round R2 to 4 decimals
    r2_val_str = round(r2_val, 4)
    
    return str(time_res), str(mae_train_str), str(mae_val_str), str(r2_train_str), str(r2_val_str), str(n_iter)

def show_results_nn(df, models_dict, X_t, y_t, X_v, y_v):
    """
    For each model in the provided dictionary:
        1. Logs training status.
        2. Calls avg_score_nn to fit and compute metrics.
        3. Appends (Time, Train MAE, Val MAE, Train R2, Val R2, Iterations) to the DataFrame.
    """
    count = 0
    total = len(models_dict)
    
    for name, model in models_dict.items():
        print(f"-> Training {name} ({count+1}/{total})...")
        
        # Call the helper function
        time_res, mae_train, mae_val, r2_train, r2_val, n_iter = avg_score_nn(model, X_t, y_t, X_v, y_v)
        
        # Save to DataFrame
        df.loc[name] = [time_res, mae_train, mae_val, r2_train, r2_val, n_iter]
        count += 1
        
    return df


# -------------------------------------------------------
# Decision Trees → 3 functions + 2 functions (for feature selection)
# -------------------------------------------------------


def fit_processing_rules_trees(X_input, y_input):
    """
    Learns all statistical rules based on the X_input set (train or full).
    Transforms X_input step-by-step, to ensure each step has the previous rules applied.
    Returns the X_input set updated as X_t and a 'rules' dictionary to be used later (validation or test).
    """
    X_t = X_input.copy()
    y_t = y_input.copy()
    rules = {}

    # --- A. IMPUTATION (Fit) ---
    # Learn means/medians/modes for missing values
    rules['nas'] = fit_nas(X_t)
    X_t = transform_nas(X_t, rules['nas']) # Must apply here so the next steps work

    # --- B. FEATURE ENGINEERING (Fit) ---
    # Learn groupings or frequencies
    rules['feat'] = fit_feat_engineering_rules(X_t, min_group_n=25)
    X_t = transform_add_all_features(X_t, rules['feat'])

    # --- C. TARGET ENCODING (Fit) ---
    # Map categories to target means -> best for tested decision trees models
    rules['encoder'] = fit_target_encoder(X_t, y_t) 
    X_t = transform_target_encoder(X_t, rules['encoder'])

    # --- D. SCALING (Fit) ---
    # Learn mean and standard deviation for normalization
    rules['scaler'] = fit_scaler(X_t)
    X_t = transform_scaler(X_t, rules['scaler'])

    # --- E. FEATURE SELECTION (Save Final Columns) ---
    # Apply selection on training data to identify WHICH columns remain
    X_t = feat_selection(X_t)

    return X_t, rules


def transform_processing_rules_trees(df_input, rules):
    """
    Applies all learned rules to a new dataset (validation or test).
    If the dataset is the test set, it will have the "carID" column and will need basic cleaning first.
    Returns the transformed DataFrame.
    """

    df = df_input.copy()
    if "carID" in df.columns:
        df = clean_basic_data(df)
    df = transform_nas(df, rules['nas'])
    df = transform_add_all_features(df, rules['feat'])
    df = transform_target_encoder(df, rules['encoder'])
    df = transform_scaler(df, rules['scaler'])
    df = feat_selection(df)

    return df


def compare_tree_models(models_dict, X_train_in, y_train_in, X_val_in, y_val_in):
    """
    Evaluates a dictionary of trained tree models on training and validation sets.
    Calculates R2 and MAE for both sets and compiles them into a comparison DataFrame.
    Returns a pandas DataFrame sorted by Validation R2.
    """
    results_list = []

    for name, model in models_dict.items():
        # Predict
        train_pred = model.predict(X_train_in)
        val_pred = model.predict(X_val_in)

        # Calculate Metrics
        r2_t = r2_score(y_train_in, train_pred)
        r2_v = r2_score(y_val_in, val_pred)
        mae_t = mean_absolute_error(y_train_in, train_pred)
        mae_v = mean_absolute_error(y_val_in, val_pred)

        # Append to list
        results_list.append({
            "Model": name,
            "Train R²": r2_t,
            "Val R²": r2_v,
            "Train MAE": mae_t,
            "Val MAE": mae_v
        })

    # Create DataFrame
    df_results = pd.DataFrame(results_list)
    return df_results

      

# helpers for visualizations start -------------- 
def correlation_matrices(df_input):
    df = df_input.copy()
    num_cols = df.select_dtypes(include=['number']).columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns

    # --- Helper: Cramers V & Correlation Ratio ---
    def cramers_v(x, y):
        conf_mat = pd.crosstab(x, y)
        n = conf_mat.sum().sum()
        return np.sqrt(stats.chi2_contingency(conf_mat)[0] / (n * (min(conf_mat.shape) - 1)))

    def corr_ratio(cat, num):
        cat, num = cat.dropna(), num.dropna()
        ss_total = np.sum((num - num.mean())**2)
        ss_between = np.sum(num.groupby(cat).count() * (num.groupby(cat).mean() - num.mean())**2)
        return 0 if ss_total == 0 else np.sqrt(ss_between / ss_total)

    # --- 1. NUM-NUM (Spearman) ---
    plt.figure(figsize=(10, 6))
    corr = df[num_cols].corr(method='spearman').round(2)
    sns.heatmap(corr, annot=True, cmap='YlOrBr', fmt='.2f', vmin=-1, vmax=1, center=0, linewidths=0.5)
    plt.title('Numeric-Numeric Correlation (Spearman)', fontweight='bold')
    plt.show()

    # --- 2. CAT-NUM (Eta / Correlation Ratio) ---
    eta_mat = pd.DataFrame([[corr_ratio(df[c], df[n]) for c in cat_cols] for n in num_cols], index=num_cols, columns=cat_cols)
    plt.figure(figsize=(10, 6))
    sns.heatmap(eta_mat.astype(float), annot=True, cmap='YlOrBr', fmt='.2f', vmin=0, vmax=1, linewidths=0.5)
    plt.title('Categorical-Numeric Correlation (Eta)', fontweight='bold')
    plt.show()

    # --- 3. CAT-CAT (Cramér\'s V) ---
    cramers_mat = pd.DataFrame([[cramers_v(df[c1], df[c2]) for c2 in cat_cols] for c1 in cat_cols], index=cat_cols, columns=cat_cols)
    plt.figure(figsize=(10, 6))
    sns.heatmap(cramers_mat.astype(float), annot=True, cmap='YlOrBr', fmt='.2f', vmin=0, vmax=1, linewidths=0.5)
    plt.title('Categorical-Categorical Correlation (Cramér\'s V)', fontweight='bold')
    plt.show()


def apply_pipeline_for_feat_importance(df_input, rules):
    """
    Applies all learned rules to the train dataset, except for feature selection.
    This is to prepare the data for feature importance analysis.
    Returns the transformed DataFrame.
    """
    df = df_input.copy()
    df = transform_nas(df, rules['nas'])
    df = transform_add_all_features(df, rules['feat'])
    df = transform_target_encoder(df, rules['encoder'])
    df = transform_scaler(df, rules['scaler'])
    return df
# helpers for visualizations end  ------------------- 


# ========================================================
# Blending → 1 function
# ========================================================


def objective_function_r2(w, *args):
    """
    w: array of weights (length = number of models)
    args: (*model_predictions, y_true)
    """

    *predictions, y_true = args

    # Prevent degenerate solutions
    weight_sum = np.sum(w)
    if weight_sum == 0:
        return 1e6

    # Weighted ensemble prediction
    final_pred = np.zeros_like(y_true, dtype=np.float64)

    for i, pred in enumerate(predictions):
        final_pred += w[i] * pred

    final_pred /= weight_sum

    # Negative R2 for minimization
    return -r2_score(y_true, final_pred)


# ========================================================
# Open Ended Section
# ========================================================


# -------------------------------------------------------
# Analytics interface for new predictions & Outlier analysis → 3 functions
# -------------------------------------------------------


def score_new_data(df_raw, rules, ref_columns, apply_clean=True):
    """
    Apply the SAME fitted preprocessing rules used in training and align columns.
    
    Parameters
    ----------
    df_raw : pd.DataFrame
        Raw input data.
    rules : dict
        Dict containing fitted rules: {"nas":..., "feat":..., "encoder":..., "scaler":...}
    ref_columns : list-like
        Training columns schema to align to.
    apply_clean : bool
        Whether to run clean_basic_data before transforms.
    """
    df = df_raw.copy()

    if apply_clean:
        df = clean_basic_data(df)

    df = transform_nas(df, rules["nas"])
    df = transform_add_all_features(df, rules["feat"])
    df = transform_target_encoder(df, rules["encoder"])
    df = transform_scaler(df, rules["scaler"])
    df = feat_selection(df)

    # Align to training schema
    df = df.reindex(columns=ref_columns, fill_value=0)
    return df

def predict_price(new_data, model, rules, ref_columns, apply_clean=True):
    """
    Predict price for new data.

    Parameters
    ----------
    new_data : dict or pd.DataFrame
        One car (dict) or many cars (DataFrame).
    model : fitted estimator with .predict()
    rules : dict
        Fitted preprocessing rules.
    ref_columns : list-like
        Training feature columns to align to.
    apply_clean : bool
        Whether to run clean_basic_data in scoring.
    """
    if isinstance(new_data, dict):
        df = pd.DataFrame([new_data])
    elif isinstance(new_data, pd.DataFrame):
        df = new_data.copy()
    else:
        raise ValueError("new_data must be a dict or a pandas DataFrame")

    Xp = score_new_data(df, rules, ref_columns, apply_clean=apply_clean)
    return np.asarray(model.predict(Xp))


def evaluate_model(model, X_train, y_train, X_val, y_val):
    """
    Fit model on (X_train, y_train) and evaluate MAE/R2 on train and val.
    Returns a dict of metrics.
    """
    model.fit(X_train, y_train)
    pred_tr = model.predict(X_train)
    pred_va = model.predict(X_val)
    return {
        "train_mae": mean_absolute_error(y_train, pred_tr),
        "train_r2":  r2_score(y_train, pred_tr),
        "val_mae":   mean_absolute_error(y_val, pred_va),
        "val_r2":    r2_score(y_val, pred_va),
    }



# -------------------------------------------------------
# Feature Importance across price levels → 3 functions
# -------------------------------------------------------

def importance_by_price_band(model, X, y, q=4, n_repeats=10, seed=42,
                             scoring="neg_mean_absolute_error",
                             normalize=True, min_bin_size=200):
    """
    Permutation feature importance computed separately within `y` quantile price bands.

    Splits `y` into `q` bands using `pd.qcut` and, for each band with at least
    `min_bin_size` samples, runs `permutation_importance` on that subset.
    Importances are clamped to >= 0 and optionally normalized to sum to 1 per band.

    Returns:
        df_long: Long table with columns [band, feature, importance, n].
        df_wide: Wide table (index=feature, columns=band) with importances.
    """
    
    if isinstance(X, pd.DataFrame):
        X_df = X.copy()
    else:
        X_df = pd.DataFrame(X)
        X_df.columns = [f"f{i}" for i in range(X_df.shape[1])]

    y = np.asarray(y).ravel()
    bands = pd.qcut(pd.Series(y), q=q, duplicates="drop")

    band_order = sorted(bands.cat.categories, key=lambda iv: float(iv.left))

    out = []
    for band in band_order:
        idx = (bands == band).to_numpy()
        n = int(idx.sum())
        if n < min_bin_size:
            continue

        Xb = X_df.loc[idx]
        yb = y[idx]

        pi = permutation_importance(
            model, Xb, yb,
            scoring=scoring,
            n_repeats=n_repeats,
            random_state=seed,
            n_jobs=-1
        )

        imp = np.maximum(pi.importances_mean.astype(float), 0.0)
        if normalize:
            s = imp.sum()
            imp = imp / s if s > 0 else imp

        for f, v in zip(X_df.columns, imp):
            out.append({"band": str(band), "feature": f, "importance": float(v), "n": n})

    if len(out) == 0:
        df_long = pd.DataFrame(columns=["band", "feature", "importance", "n"])
        df_wide = pd.DataFrame(index=X_df.columns)
        return df_long, df_wide

    df_long = pd.DataFrame(out)
    df_wide = (df_long.pivot_table(index="feature", columns="band",
                                   values="importance", aggfunc="mean")
                      .fillna(0.0))

    ordered_band_strs = [str(b) for b in band_order]
    existing = [b for b in ordered_band_strs if b in df_wide.columns]
    df_wide = df_wide.reindex(columns=existing)

    df_long["band"] = pd.Categorical(df_long["band"], categories=existing, ordered=True)
    df_long = df_long.sort_values(["band", "importance"], ascending=[True, False]).reset_index(drop=True)

    return df_long, df_wide



def plot_topk_each_band(df_long, topk=12):
    """
    Plots top-k features per band. Returns list of matplotlib Figure objects.
    """
    figs = []
    for band in df_long["band"].unique():
        sub = (df_long[df_long["band"] == band]
               .sort_values("importance", ascending=False)
               .head(topk))
        if sub.empty:
            continue
        n = int(sub["n"].iloc[0])

        fig = plt.figure()
        plt.barh(sub["feature"][::-1], sub["importance"][::-1])
        plt.title(f"Permutation importance — {band} | n={n}")
        plt.xlabel("Normalized importance (sums to 1 within band)")
        plt.tight_layout()
        figs.append(fig)
    return figs


def plot_heatmap(df_wide, top_features=25, figsize=(10, 6), sort_bands=True):
    """
    Heatmap of per-band importances. Bands can be sorted by interval lower bound.
    Returns matplotlib Figure.
    """
    # --- sort x-axis bands (columns) by their numeric lower bound
    cols = list(df_wide.columns)
    if sort_bands:
        col_ser = pd.Series(cols, index=cols).astype(str)
        lower = pd.to_numeric(
            col_ser.str.extract(r'[\(\[]\s*([^,]+)\s*,', expand=False),
            errors="coerce"
        )
        cols = list(lower.sort_values(kind="mergesort", na_position="last").index)

    # --- select top features by mean importance across bands
    avg = df_wide.mean(axis=1).sort_values(ascending=False)
    sel = avg.head(top_features).index
    M = df_wide.loc[sel, cols].values

    fig = plt.figure(figsize=figsize)
    plt.imshow(M, aspect="auto")
    plt.yticks(range(len(sel)), sel)
    plt.xticks(range(len(cols)), cols, rotation=0)
    plt.title("Feature importance by price band" + (" (sorted)" if sort_bands else ""))
    plt.colorbar(label="Normalized importance")
    plt.tight_layout()
    return fig


# -------------------------------------------------------
# General vs Segment-Specific Model Performance → 3 functions
# -------------------------------------------------------

def _loguniform(rng, low, high):
    """Sample uniformly in log10-space from [low, high]."""
    return float(10 ** rng.uniform(np.log10(low), np.log10(high)))


def hgb_param_sampler(rng):
    """
    Random sampler for HistGradientBoostingRegressor hyperparams.
    (You can keep this identical to notebook for reproducibility.)
    """
    return {
        "learning_rate":     _loguniform(rng, 0.01, 0.15),
        "max_leaf_nodes":    int(rng.integers(15, 255)),
        "min_samples_leaf":  int(rng.integers(3, 25)),
        "l2_regularization": _loguniform(rng, 1e-3, 10.0),
        "max_depth":         int(rng.integers(3, 15)),
        "max_features":      float(rng.uniform(0.30, 1.0)),
        "max_iter":          3000,
        "early_stopping":    True,
        "validation_fraction": 0.15,
        "n_iter_no_change":  20,
    }


def quick_tune_on_train_only(
    X_train_df, y_train,
    cols,
    n_iter=20,
    seed=42,
    tune_sub=25000,
    inner_val_size=0.2,
):
    """
    Quick tuning using ONLY training data:
      - optional subsample for speed (tune_sub)
      - inner split train/inner_val
      - random search with hgb_param_sampler

    Parameters
    ----------
    X_train_df : pd.DataFrame
        Training features (must be DataFrame so cols indexing works).
    y_train : array-like
        Training target.
    cols : list[str]
        Columns to keep for tuning.
    """
    rng = np.random.default_rng(seed)

    if not isinstance(X_train_df, pd.DataFrame):
        raise TypeError("X_train_df must be a pandas DataFrame (so we can select cols).")

    X = X_train_df[cols]
    y = np.asarray(y_train, dtype=np.float32).ravel()

    # subsample for tuning speed (still train-only)
    if tune_sub is not None and len(y) > tune_sub:
        idx = rng.choice(len(y), size=tune_sub, replace=False)
        X = X.iloc[idx]
        y = y[idx]

    X_in_tr, X_in_va, y_in_tr, y_in_va = train_test_split(
        X, y, test_size=inner_val_size, random_state=seed
    )

    # numpy for speed
    X_in_tr_np = np.asarray(X_in_tr, dtype=np.float32)
    X_in_va_np = np.asarray(X_in_va, dtype=np.float32)

    best_mae = np.inf
    best_params = None

    for _ in range(n_iter):
        params = hgb_param_sampler(rng)
        m = HistGradientBoostingRegressor(random_state=seed, **params)
        m.fit(X_in_tr_np, y_in_tr)
        pred = m.predict(X_in_va_np)
        mae  = float(np.mean(np.abs(y_in_va - pred)))

        if mae < best_mae:
            best_mae = mae
            best_params = params

    return best_params, best_mae


# ==============================================================
# ==============================================================
# ============================ End =============================
# ==============================================================
# ==============================================================
