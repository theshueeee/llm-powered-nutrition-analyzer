import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt




#Visualization
# Function to visualize item nutrition
def visualize_item_nutrition(item_name, df_food, df_drinks):
    # Normalize index for searching
    df_food_search = df_food.copy()
    df_drinks_search = df_drinks.copy()
    
    df_food_search.index = df_food_search.index.str.lower()
    df_drinks_search.index = df_drinks_search.index.str.lower()
    
    item_name = item_name.lower()
    
    # Find the item in either food or drinks
    if item_name in df_food_search.index:
        item = df_food_search.loc[item_name]
        category = "Food"
        protein = item['Protein (g)']
    elif item_name in df_drinks_search.index:
        item = df_drinks_search.loc[item_name]
        category = "Drink"
        protein = item['Protein']
    else:
        st.error(f"Item '{item_name}' not found in menu.")
        return
    
    # Create two columns
    col1, col2 = st.columns(2)
    
    # Get nutrient values
    sodium = item['Sodium'] if 'Sodium' in item else 0
    
    # Nutrients to show
    nutrients = {
        'Calories': item['Calories'],
        'Fat (g)': item['Fat (g)'],
        'Carb. (g)': item['Carb. (g)'],
        'Estimated Sugar (g)': item['Estimated Sugar (g)'],
        'Fiber (g)': item['Fiber (g)'],
        'Protein (g)': protein,
        'Sodium (mg)': sodium
    }
    
    # Bar Chart in column 1
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.bar(nutrients.keys(), nutrients.values(), color='skyblue')
    ax1.set_title(f'Nutrient Breakdown')
    ax1.set_ylabel('Amount')
    plt.xticks(rotation=45)
    plt.tight_layout()
    col1.pyplot(fig1)
    
    # Pie Chart in column 2
    fat_cal = item['Fat (g)'] * 9 if not pd.isna(item['Fat (g)']) else 0
    prot_cal = protein * 4 if not pd.isna(protein) else 0
    carb_cal = item['Carb. (g)'] * 4 if not pd.isna(item['Carb. (g)']) else 0
    
    total_macros = fat_cal + prot_cal + carb_cal
    if total_macros > 0:
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        ax2.pie([carb_cal, fat_cal, prot_cal],
                labels=['Carbs', 'Fat', 'Protein'],
                autopct='%1.1f%%',
                colors=['#FF9999', '#66B3FF', '#99FF99'])
        ax2.set_title('Macronutrient Caloric Breakdown')
        plt.tight_layout()
        col2.pyplot(fig2)
    else:
        col2.warning("Insufficient macronutrient data to display pie chart.")