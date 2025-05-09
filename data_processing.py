import streamlit as st
import pandas as pd
import numpy as np


# Function to load and clean data
@st.cache_data
def load_and_clean_data(drinks_file, food_file):
    # Load data
    df_drinks = pd.read_csv(drinks_file)
    df_food = pd.read_csv(food_file, encoding='utf-16')
    
    # Clean column names
    df_food.columns = df_food.columns.str.strip()
    df_drinks.columns = df_drinks.columns.str.strip()
    
    # Rename first column to 'Item'
    df_food.rename(columns={'Unnamed: 0': 'Item'}, inplace=True)
    df_drinks.rename(columns={'Unnamed: 0': 'Item'}, inplace=True)
    
    # Set 'Item' as index
    df_food.set_index('Item', inplace=True)
    df_drinks.set_index('Item', inplace=True)
    
    # Replace "-" with NaN
    df_drinks.replace("-", np.nan, inplace=True)
    
    # Drop NaN values
    df_drinks.dropna(inplace=True)
    
    # Convert columns to numeric
    food_cols = ['Calories', 'Fat (g)', 'Carb. (g)', 'Fiber (g)', 'Protein (g)']
    drink_cols = ['Calories', 'Fat (g)', 'Carb. (g)', 'Fiber (g)', 'Protein', 'Sodium']
    
    df_food[food_cols] = df_food[food_cols].apply(pd.to_numeric, errors='coerce')
    df_drinks[drink_cols] = df_drinks[drink_cols].apply(pd.to_numeric, errors='coerce')
    
    # Remove duplicates
    df_drinks.drop_duplicates(inplace=True)
    
    # Add estimated sugar
    df_food['Estimated Sugar (g)'] = df_food['Carb. (g)'] * 0.6
    df_drinks['Estimated Sugar (g)'] = df_drinks['Carb. (g)'] * 0.6
    
    # Add caffeine inference
    df_drinks['Has Caffeine'] = df_drinks.index.to_series().apply(infer_caffeine_presence)
    
    return df_food, df_drinks

#Start of Descriptive Statistics
# Function to infer caffeine presence
def infer_caffeine_presence(name):
    caffeine_keywords = ['coffee', 'espresso', 'latte', 'cappuccino', 'mocha', 'macchiato', 'tea', 'matcha', 'cold brew']
    name = name.lower()
    return any(keyword in name for keyword in caffeine_keywords)

# Function to get nutrition summary
def enhanced_nutrition_summary(df_food, df_drinks):
    df_food = df_food.copy()
    df_drinks = df_drinks.copy()

    # Avoid division by zero in fat-to-protein ratio
    df_food['Protein (g)'].replace(0, pd.NA, inplace=True)
    df_drinks['Protein'].replace(0, pd.NA, inplace=True)

    # Fat-to-protein ratio
    df_food['fat_protein_ratio'] = df_food['Fat (g)'] / df_food['Protein (g)']
    df_drinks['fat_protein_ratio'] = df_drinks['Fat (g)'] / df_drinks['Protein']

    # Prepare summary DataFrame with new metrics
    summary_df = pd.DataFrame({
        'Category': ['Food', 'Drinks'],

        # Calorie-related statistics
        'Total Calories': [df_food['Calories'].sum(), df_drinks['Calories'].sum()],
        'Average Calories': [df_food['Calories'].mean(), df_drinks['Calories'].mean()],
        'Median Calories': [df_food['Calories'].median(), df_drinks['Calories'].median()],
        'Min Calories': [df_food['Calories'].min(), df_drinks['Calories'].min()],
        'Max Calories': [df_food['Calories'].max(), df_drinks['Calories'].max()],
        'Calories Std Dev': [df_food['Calories'].std(), df_drinks['Calories'].std()],

        # Macronutrient composition (average percentages)
        'Avg. Carbs %': [
            (df_food['Carb. (g)'] * 4 / df_food['Calories']).mean() * 100,
            (df_drinks['Carb. (g)'] * 4 / df_drinks['Calories']).replace([np.inf, -np.inf], np.nan).mean() * 100
        ],
        'Avg. Fat %': [
            (df_food['Fat (g)'] * 9 / df_food['Calories']).mean() * 100,
            (df_drinks['Fat (g)'] * 9 / df_drinks['Calories']).replace([np.inf, -np.inf], np.nan).mean() * 100
        ],
        'Avg. Protein %': [
            (df_food['Protein (g)'] * 4 / df_food['Calories']).mean() * 100,
            (df_drinks['Protein'] * 4 / df_drinks['Calories']).replace([np.inf, -np.inf], np.nan).mean() * 100
        ],

        # Sugar-related statistics
        'Estimated Avg. Sugar (g)': [
            df_food['Estimated Sugar (g)'].mean(),
            df_drinks['Estimated Sugar (g)'].mean()
        ],
        'Estimated Sugar % of Carbs': [
            (df_food['Estimated Sugar (g)'] / df_food['Carb. (g)']).mean() * 100,
            (df_drinks['Estimated Sugar (g)'] / df_drinks['Carb. (g)']).replace([np.inf, -np.inf], np.nan).mean() * 100
        ],

        # Fat-to-protein ratio
        'Avg. Fat-to-Protein Ratio': [
            df_food['fat_protein_ratio'].mean(),
            df_drinks['fat_protein_ratio'].mean()
        ],

        # Fiber statistics
        'Avg. Fiber (g)': [
            df_food['Fiber (g)'].mean(),
            df_drinks['Fiber (g)'].mean()
        ],
        'Max Fiber (g)': [
            df_food['Fiber (g)'].max(),
            df_drinks['Fiber (g)'].max()
        ],

        # Sodium statistics
        'Avg. Sodium (mg)': [
            df_food['Sodium'].mean() if 'Sodium' in df_food.columns else np.nan,
            df_drinks['Sodium'].mean() if 'Sodium' in df_drinks.columns else np.nan
        ],

        # Nutritional density
        'Caloric Density': [
            df_food['Calories'] / (df_food['Carb. (g)'] + df_food['Protein (g)'] + df_food['Fat (g)']),
            df_drinks['Calories'] / (df_drinks['Carb. (g)'] + df_drinks['Protein'] + df_drinks['Fat (g)'])
        ],
    })

    # Calculate additional health metrics
    summary_df['Estimated Protein Quality'] = [
        min(1.0, df_food['Protein (g)'].mean() / 20),  # Simplified estimation
        min(1.0, df_drinks['Protein'].mean() / 10)     # Simplified estimation
    ]

    # Nutrient Density Score
    summary_df['Nutrient Density Score'] = [
        ((df_food['Protein (g)'].fillna(0) * 4 + df_food['Fiber (g)'].fillna(0) * 2) /
         df_food['Calories']).replace([np.inf, -np.inf], np.nan).mean(),
        ((df_drinks['Protein'].fillna(0) * 4 + df_drinks['Fiber (g)'].fillna(0) * 2) /
         df_drinks['Calories']).replace([np.inf, -np.inf], np.nan).mean()
    ]

    return summary_df

# Function to filter menu items
def filter_menu_items(
    df_food, df_drinks,
    category="all",
    max_calories=None,
    has_caffeine=None,
    min_protein=None,
    max_est_sugar=None,
    max_fat=None,
    min_fiber=None
):
    df_food = df_food.copy()
    df_drinks = df_drinks.copy()

    # Set categories
    df_food['Category'] = 'Food'
    df_drinks['Category'] = 'Drinks'

    if category.lower() == "food":
        df = df_food
    elif category.lower() == "drinks":
        df = df_drinks
    else:
        df = pd.concat([df_food, df_drinks])

    # Apply filters
    if max_calories is not None:
        df = df[df['Calories'] <= max_calories]
    
    if has_caffeine is not None and category.lower() != "food":
        # Only apply caffeine filter for drinks or combined
        if category.lower() == "drinks":
            df = df[df['Has Caffeine'] == has_caffeine]
        else:
            # For combined, filter only the drinks part
            mask = (df['Category'] == 'Food') | (df['Has Caffeine'] == has_caffeine)
            df = df[mask]
    
    if min_protein is not None:
        protein_col = 'Protein (g)' if 'Protein (g)' in df.columns else 'Protein'
        df = df[df[protein_col] >= min_protein]
    
    if max_est_sugar is not None:
        df = df[df['Estimated Sugar (g)'] <= max_est_sugar]
    
    if max_fat is not None:
        df = df[df['Fat (g)'] <= max_fat]
    
    if min_fiber is not None:
        df = df[df['Fiber (g)'] >= min_fiber]

    return df


