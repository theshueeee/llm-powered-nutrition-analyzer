import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from visualizations import visualize_item_nutrition
from data_processing import (
    load_and_clean_data,
    infer_caffeine_presence,
    filter_menu_items,
    enhanced_nutrition_summary
)
from llm_integration import query_nutrition_llm



# Set page configuration
st.set_page_config(
    page_title="Starbucks Nutrition Analysis",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to improve layout
st.markdown("""
<style>
    .main {
        padding: 1rem 1rem;
    }
    .streamlit-expanderHeader {
        font-size: 1.2em;
        font-weight: bold;
    }
    .stPlotlyChart {
        height: 100%;
    }
    h1 {
        color: #006241;
    }
    h2 {
        color: #006241;
    }
    .highlight {
        padding: 10px;
        border-radius: 5px;
        background-color: #f0f0f0;
    }
</style>
""", unsafe_allow_html=True)

# Title and brief description
st.title("☕ Starbucks Nutrition Analysis Tool")
st.markdown("""
This interactive dashboard analyzes nutritional information from Starbucks menu items. 
Upload the CSV files, explore the data, filter items based on your preferences, and get
AI-powered insights about your favorite Starbucks items.
""")


#Start of Streamlit integration
# Sidebar for uploading files
st.sidebar.header("Data Upload")

# Default values for demo purpose
use_default_data = st.sidebar.checkbox("Use Demo Data", value=True)

if use_default_data:
    # Create demo data
    @st.cache_data
    def create_demo_data():
        # Generate sample Starbucks food data
        food_data = {
            'Item': ['Chocolate Croissant', 'Blueberry Muffin', 'Chicken & Bacon Panini', 
                    'Oatmeal with Fruit', 'Egg Sandwich', 'Turkey Sandwich', 'Yogurt Parfait'],
            'Calories': [340, 360, 500, 220, 380, 450, 240],
            'Fat (g)': [18, 15, 22, 3, 18, 16, 6],
            'Carb. (g)': [39, 52, 43, 39, 36, 48, 38],
            'Fiber (g)': [2, 3, 2, 4, 2, 3, 1],
            'Protein (g)': [5, 6, 32, 5, 19, 28, 14],
            'Sodium': [280, 290, 950, 125, 680, 1200, 120]
        }
        
        # Generate sample Starbucks drinks data
        drinks_data = {
            'Item': ['Cappuccino', 'Latte', 'Mocha', 'Cold Brew', 'Iced Tea', 
                    'Hot Chocolate', 'Frappuccino', 'Caramel Macchiato', 'Chai Tea Latte'],
            'Calories': [120, 190, 360, 5, 60, 320, 380, 250, 240],
            'Fat (g)': [4, 7, 15, 0, 0, 9, 16, 7, 4],
            'Carb. (g)': [12, 19, 44, 0, 15, 45, 54, 33, 42],
            'Fiber (g)': [0, 0, 2, 0, 0, 3, 0, 0, 0],
            'Protein': [8, 12, 13, 0, 0, 11, 5, 10, 8],
            'Sodium': [75, 170, 150, 10, 5, 115, 240, 150, 115]
        }
        
        df_food = pd.DataFrame(food_data)
        df_drinks = pd.DataFrame(drinks_data)
        
        # Set 'Item' as index
        df_food.set_index('Item', inplace=True)
        df_drinks.set_index('Item', inplace=True)
        
        # Add estimated sugar
        df_food['Estimated Sugar (g)'] = df_food['Carb. (g)'] * 0.6
        df_drinks['Estimated Sugar (g)'] = df_drinks['Carb. (g)'] * 0.6
        
        # Add caffeine inference
        df_drinks['Has Caffeine'] = df_drinks.index.to_series().apply(infer_caffeine_presence)
        
        return df_food, df_drinks
    
    df_food, df_drinks = create_demo_data()
    st.sidebar.success("Demo data loaded successfully!")
else:
    # File upload for actual data
    uploaded_drinks = st.sidebar.file_uploader("Upload Drinks CSV", type="csv")
    uploaded_food = st.sidebar.file_uploader("Upload Food CSV", type="csv")
    
    if uploaded_drinks and uploaded_food:
        try:
            df_food, df_drinks = load_and_clean_data(uploaded_drinks, uploaded_food)
            st.sidebar.success("Data loaded and cleaned successfully!")
        except Exception as e:
            st.sidebar.error(f"Error loading data: {str(e)}")
    else:
        st.warning("Please upload both CSV files or use the demo data.")
        st.stop()

# Main app sections as tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Menu Explorer", "Item Details", "Nutritional Comparison", "AI Insights"])

with tab1:
    st.header("Starbucks Menu Overview")
    
    # Display summary statistics
    summary_df = enhanced_nutrition_summary(df_food, df_drinks)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Key Statistics")
        # Display a nice formatted table
        st.dataframe(summary_df.set_index('Category')[
            ['Average Calories', 'Estimated Avg. Sugar (g)', 'Avg. Fat %', 'Avg. Protein %', 'Avg. Fiber (g)']
        ].style.format("{:.1f}"), use_container_width=True)
    
    with col2:
        st.subheader("Food vs Drinks Distribution")
        # Create a calorie distribution chart
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create histogram for both categories
        df_food_reset = df_food.reset_index()
        df_food_reset['Category'] = 'Food'
        df_drinks_reset = df_drinks.reset_index()
        df_drinks_reset['Category'] = 'Drinks'
        
        combined_df = pd.concat([df_food_reset, df_drinks_reset])
        
        sns.histplot(data=combined_df, x='Calories', hue='Category', bins=15, kde=True, ax=ax)
        plt.title('Calorie Distribution: Food vs Drinks')
        plt.xlabel('Calories')
        plt.ylabel('Count')
        st.pyplot(fig)
    
    # Show count of items in each category
    st.subheader("Menu Composition")
    col1, col2 = st.columns(2)
    
    with col1:
        food_count = len(df_food)
        drink_count = len(df_drinks)
        total_count = food_count + drink_count
        
        # Create pie chart
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.pie([food_count, drink_count], 
               labels=['Food', 'Drinks'], 
               autopct='%1.1f%%',
               explode=(0.05, 0),
               colors=['#ff9999', '#66b3ff'],
               shadow=True)
        ax.set_title('Menu Item Distribution')
        st.pyplot(fig)
    
    with col2:
        st.markdown("### Menu Items")
        st.metric("Food Items", food_count)
        st.metric("Drink Items", drink_count)
        st.metric("Total Items", total_count)
        
        st.markdown("""
        **Note**: This analysis is based on the available data from Starbucks menu. 
        The nutritional information may vary based on customizations and serving sizes.
        """)

with tab2:
    st.header("Menu Explorer")
    st.markdown("Use the filters below to find menu items that match your dietary preferences.")
    
    # Create filter sidebar
    col1, col2 = st.columns(2)
    
    with col1:
        category = st.selectbox("Category", ["All", "Food", "Drinks"])
        max_calories = st.slider("Maximum Calories", 0, 1000, 500)
        min_protein = st.slider("Minimum Protein (g)", 0, 40, 0)
    
    with col2:
        max_fat = st.slider("Maximum Fat (g)", 0, 50, 30)
        max_sugar = st.slider("Maximum Est. Sugar (g)", 0, 100, 50)
        
        # Only show caffeine filter for drinks or all
        if category != "Food":
            has_caffeine = st.radio("Contains Caffeine", [None, True, False], horizontal=True, index=0)
        else:
            has_caffeine = None
    
    # Filter the data
    filtered_data = filter_menu_items(
        df_food, df_drinks,
        category=category,
        max_calories=max_calories,
        has_caffeine=has_caffeine,
        min_protein=min_protein,
        max_est_sugar=max_sugar,
        max_fat=max_fat
    )
    
    # Display filtered results
    if not filtered_data.empty:
        st.subheader(f"Found {len(filtered_data)} items matching your criteria")
        
        # Add category column for display if needed
        if 'Category' not in filtered_data.columns:
            filtered_data['Category'] = 'Food' if category == 'Food' else 'Drinks'
        
        # Reset index for better display
        display_df = filtered_data.reset_index()
        
        # Select columns to display
        if category == "Food":
            columns_to_display = ['Item', 'Calories', 'Fat (g)', 'Carb. (g)', 'Protein (g)', 'Estimated Sugar (g)', 'Fiber (g)']
        elif category == "Drinks":
            columns_to_display = ['Item', 'Calories', 'Fat (g)', 'Carb. (g)', 'Protein', 'Estimated Sugar (g)', 'Has Caffeine']
        else:
            columns_to_display = ['Item', 'Category', 'Calories', 'Fat (g)', 'Carb. (g)', 'Estimated Sugar (g)']
        
        # Show results
        st.dataframe(display_df[columns_to_display], use_container_width=True)
        
        # Download button for filtered results
        csv_buffer = BytesIO()
        display_df[columns_to_display].to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        st.download_button(
            label="Download Filtered Results",
            data=csv_buffer,
            file_name="filtered_starbucks_items.csv",
            mime="text/csv"
        )
    else:
        st.warning("No items found matching your criteria. Try adjusting the filters.")

with tab3:
    st.header("Item Details")
    st.markdown("Select a specific menu item to view its detailed nutritional information.")
    
    # Get all item names
    food_items = list(df_food.index)
    drink_items = list(df_drinks.index)
    all_items = food_items + drink_items
    
    # Item selector
    selected_item = st.selectbox("Select an item", all_items)
    
    if selected_item:
        st.subheader(f"Nutritional Profile: {selected_item}")
        # Show item details
        visualize_item_nutrition(selected_item, df_food, df_drinks)
        
        # Get item data
        if selected_item in food_items:
            item_data = df_food.loc[selected_item]
            category = "Food"
        else:
            item_data = df_drinks.loc[selected_item]
            category = "Drink"
        
        # Additional information
        st.markdown("### Nutritional Details")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Category", category)
            st.metric("Calories", f"{item_data['Calories']:.0f} kcal")
            
        with col2:
            if category == "Food":
                st.metric("Protein", f"{item_data['Protein (g)']:.1f} g")
            else:
                st.metric("Protein", f"{item_data['Protein']:.1f} g")
            st.metric("Fat", f"{item_data['Fat (g)']:.1f} g")
            
        with col3:
            st.metric("Carbohydrates", f"{item_data['Carb. (g)']:.1f} g")
            st.metric("Est. Sugar", f"{item_data['Estimated Sugar (g)']:.1f} g")
        
        # Health insights
        st.markdown("### Health Insights")
        
        # Calculate percentage of daily values (based on 2000 calorie diet)
        calories_percent = (item_data['Calories'] / 2000) * 100
        
        if category == "Food":
            protein_percent = (item_data['Protein (g)'] / 50) * 100
        else:
            protein_percent = (item_data['Protein'] / 50) * 100
            
        fat_percent = (item_data['Fat (g)'] / 65) * 100
        carb_percent = (item_data['Carb. (g)'] / 300) * 100
        
        # Show percentage bars
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### % Daily Value (2000 calorie diet)")
            st.progress(min(calories_percent, 100)/100, text=f"Calories: {calories_percent:.1f}%")
            st.progress(min(protein_percent, 100)/100, text=f"Protein: {protein_percent:.1f}%")
            st.progress(min(fat_percent, 100)/100, text=f"Fat: {fat_percent:.1f}%")
        
        with col2:
            st.markdown("#### Nutrient Balance")
            st.progress(min(carb_percent, 100)/100, text=f"Carbs: {carb_percent:.1f}%")
            
            # Add fiber info if available
            fiber = item_data['Fiber (g)']
            if not pd.isna(fiber):
                fiber_percent = (fiber / 25) * 100
                st.progress(min(fiber_percent, 100)/100, text=f"Fiber: {fiber_percent:.1f}%")
            
            # Add caffeine info for drinks
            if category == "Drink" and 'Has Caffeine' in item_data:
                caffeine_status = "Likely contains caffeine" if item_data['Has Caffeine'] else "Likely caffeine-free"
                st.info(caffeine_status)

with tab4:
    st.header("Nutritional Comparison")
    st.markdown("Compare nutritional profiles of different food and drink items.")
    
    # Select items to compare
    col1, col2 = st.columns(2)
    
    with col1:
        compare_type = st.radio("What would you like to compare?", 
                               ["Food items", "Drink items", "Food vs Drinks"], 
                               horizontal=True)
        
        if compare_type == "Food items":
            items_to_compare = st.multiselect("Select food items to compare", food_items, max_selections=5)
            if items_to_compare:
                comparison_df = df_food.loc[items_to_compare]
        elif compare_type == "Drink items":
            items_to_compare = st.multiselect("Select drink items to compare", drink_items, max_selections=5)
            if items_to_compare:
                comparison_df = df_drinks.loc[items_to_compare]
        else:
            food_item = st.selectbox("Select a food item", food_items)
            drink_item = st.selectbox("Select a drink item", drink_items)
            if food_item and drink_item:
                food_row = df_food.loc[[food_item]].copy()
                drink_row = df_drinks.loc[[drink_item]].copy()
                # Ensure protein column names match
                if 'Protein' in drink_row.columns and 'Protein (g)' in food_row.columns:
                    drink_row = drink_row.rename(columns={'Protein': 'Protein (g)'})
                comparison_df = pd.concat([food_row, drink_row])
    
    with col2:
        comparison_metric = st.selectbox("Select metric to visualize", 
                                        ["Calories", "Fat (g)", "Carb. (g)", 
                                         "Protein (g)" if compare_type != "Drink items" else "Protein",
                                         "Estimated Sugar (g)", "Fiber (g)"])
    
    # Display comparison
    if 'comparison_df' in locals() and not comparison_df.empty:
        st.subheader("Comparison Results")
        
        # Create bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        comparison_df[comparison_metric].plot(kind='bar', ax=ax, color=sns.color_palette("Set2", len(comparison_df)))
        plt.title(f'Comparison of {comparison_metric}')
        plt.ylabel(comparison_metric)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        
        # Display table
        st.markdown("### Detailed Comparison")
        display_cols = ['Calories', 'Fat (g)', 'Carb. (g)', 
                        'Protein (g)' if compare_type != "Drink items" else "Protein",
                        'Estimated Sugar (g)', 'Fiber (g)']
        
        st.dataframe(comparison_df[display_cols].style.highlight_max(axis=0, color='#ffdd99')
                    .highlight_min(axis=0, color='#99ff99'), use_container_width=True)
        
        # Radar chart for multiple metrics comparison
        if len(comparison_df) >= 2:
            st.markdown("### Multi-dimensional Comparison")
            st.markdown("This radar chart compares items across multiple nutritional dimensions.")
            
            # Normalize data for radar chart
            radar_metrics = ['Calories', 'Fat (g)', 'Carb. (g)', 
                           'Protein (g)' if compare_type != "Drink items" else "Protein",
                           'Estimated Sugar (g)']
            
            radar_df = comparison_df[radar_metrics].copy()
            for column in radar_metrics:
                max_val = radar_df[column].max()
                if max_val > 0:  # Avoid division by zero
                    radar_df[column] = radar_df[column] / max_val
            
            # Create radar chart
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, polar=True)
            
            # Set number of variables
            categories = radar_metrics
            N = len(categories)
            
            # Create angles for each variable
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]  # Close the loop
            
            # Plot each item
            for i, item in enumerate(radar_df.index):
                values = radar_df.loc[item].values.flatten().tolist()
                values += values[:1]  # Close the loop
                
                ax.plot(angles, values, linewidth=2, linestyle='solid', label=item)
                ax.fill(angles, values, alpha=0.1)
            
            # Set labels
            plt.xticks(angles[:-1], categories, size=12)
            
            # Draw y-axis labels
            ax.set_rlabel_position(0)
            plt.yticks([0.25, 0.5, 0.75], ["25%", "50%", "75%"], color="grey", size=10)
            plt.ylim(0, 1)
            
            # Add legend
            plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
            plt.title("Nutritional Radar Chart", size=15)
            
            st.pyplot(fig)
    else:
        st.info("Select items to compare to see results.")

with tab5:
    st.header("AI Nutrition Insights")
    st.markdown("""
    Ask questions about the Starbucks menu and get AI-powered nutrition insights.
    
    Example questions:
    - Which food items have the highest protein content?
    - What's the average calorie content of coffee drinks?
    - Which item has the lowest sugar content?
    - How do breakfast sandwiches compare to pastries in terms of nutrition?
    """)
    
    # API key input
    use_api = st.checkbox("Use Groq LLM for enhanced insights", value=False)
    
    if use_api:
        api_key = st.text_input("Enter your Groq API key", type="password")
    
    # Question input
    user_question = st.text_input("Ask a question about Starbucks nutrition:")
    
    if user_question:
        st.markdown("### Answer")
        
        if use_api and api_key:
            with st.spinner("Getting AI insights..."):
                answer = query_nutrition_llm(df_food, df_drinks, user_question, api_key)
                st.markdown(f"```{answer}```")
        else:
            # Simple rule-based responses without API
            if "highest" in user_question.lower() and "calorie" in user_question.lower():
                if "food" in user_question.lower():
                    highest_cal_food = df_food['Calories'].idxmax()
                    cal_value = df_food.loc[highest_cal_food, 'Calories']
                    st.markdown(f"The highest calorie food item is **{highest_cal_food}** with **{cal_value:.0f}** calories.")
                elif "drink" in user_question.lower():
                    highest_cal_drink = df_drinks['Calories'].idxmax()
                    cal_value = df_drinks.loc[highest_cal_drink, 'Calories']
                    st.markdown(f"The highest calorie drink is **{highest_cal_drink}** with **{cal_value:.0f}** calories.")
                else:
                    # Combined check
                    highest_food = df_food['Calories'].idxmax()
                    highest_food_val = df_food.loc[highest_food, 'Calories']
                    highest_drink = df_drinks['Calories'].idxmax()
                    highest_drink_val = df_drinks.loc[highest_drink, 'Calories']
                    
                    if highest_food_val > highest_drink_val:
                        st.markdown(f"The highest calorie item overall is **{highest_food}** (food) with **{highest_food_val:.0f}** calories.")
                    else:
                        st.markdown(f"The highest calorie item overall is **{highest_drink}** (drink) with **{highest_drink_val:.0f}** calories.")
            
            elif "lowest" in user_question.lower() and "calorie" in user_question.lower():
                if "food" in user_question.lower():
                    lowest_cal_food = df_food['Calories'].idxmin()
                    cal_value = df_food.loc[lowest_cal_food, 'Calories']
                    st.markdown(f"The lowest calorie food item is **{lowest_cal_food}** with **{cal_value:.0f}** calories.")
                elif "drink" in user_question.lower():
                    lowest_cal_drink = df_drinks['Calories'].idxmin()
                    cal_value = df_drinks.loc[lowest_cal_drink, 'Calories']
                    st.markdown(f"The lowest calorie drink is **{lowest_cal_drink}** with **{cal_value:.0f}** calories.")
                else:
                    st.markdown("Please specify if you're looking for food or drink items with the lowest calories.")
            
            elif "average" in user_question.lower() and "calorie" in user_question.lower():
                if "food" in user_question.lower():
                    avg_cal = df_food['Calories'].mean()
                    st.markdown(f"The average calorie content of food items is **{avg_cal:.1f}** calories.")
                elif "drink" in user_question.lower():
                    avg_cal = df_drinks['Calories'].mean()
                    st.markdown(f"The average calorie content of drink items is **{avg_cal:.1f}** calories.")
                else:
                    avg_food = df_food['Calories'].mean()
                    avg_drink = df_drinks['Calories'].mean()
                    st.markdown(f"The average calorie content of food items is **{avg_food:.1f}** calories.")
                    st.markdown(f"The average calorie content of drink items is **{avg_drink:.1f}** calories.")
            
            elif "protein" in user_question.lower():
                if "highest" in user_question.lower():
                    if "food" in user_question.lower():
                        highest_protein_food = df_food['Protein (g)'].idxmax()
                        protein_value = df_food.loc[highest_protein_food, 'Protein (g)']
                        st.markdown(f"The highest protein food item is **{highest_protein_food}** with **{protein_value:.1f}g** of protein.")
                    elif "drink" in user_question.lower():
                        highest_protein_drink = df_drinks['Protein'].idxmax()
                        protein_value = df_drinks.loc[highest_protein_drink, 'Protein']
                        st.markdown(f"The highest protein drink is **{highest_protein_drink}** with **{protein_value:.1f}g** of protein.")
                    else:
                        st.markdown("Please specify if you're looking for food or drink items with the highest protein.")
                else:
                    avg_protein_food = df_food['Protein (g)'].mean()
                    avg_protein_drink = df_drinks['Protein'].mean()
                    st.markdown(f"The average protein content of food items is **{avg_protein_food:.1f}g**.")
                    st.markdown(f"The average protein content of drink items is **{avg_protein_drink:.1f}g**.")
            
            elif "sugar" in user_question.lower():
                if "highest" in user_question.lower():
                    highest_sugar_food = df_food['Estimated Sugar (g)'].idxmax()
                    highest_sugar_food_val = df_food.loc[highest_sugar_food, 'Estimated Sugar (g)']
                    highest_sugar_drink = df_drinks['Estimated Sugar (g)'].idxmax()
                    highest_sugar_drink_val = df_drinks.loc[highest_sugar_drink, 'Estimated Sugar (g)']
                    
                    st.markdown(f"Highest sugar food: **{highest_sugar_food}** with **{highest_sugar_food_val:.1f}g** sugar.")
                    st.markdown(f"Highest sugar drink: **{highest_sugar_drink}** with **{highest_sugar_drink_val:.1f}g** sugar.")
                
                elif "lowest" in user_question.lower():
                    lowest_sugar_food = df_food['Estimated Sugar (g)'].idxmin()
                    lowest_sugar_food_val = df_food.loc[lowest_sugar_food, 'Estimated Sugar (g)']
                    lowest_sugar_drink = df_drinks['Estimated Sugar (g)'].idxmin()
                    lowest_sugar_drink_val = df_drinks.loc[lowest_sugar_drink, 'Estimated Sugar (g)']
                    
                    st.markdown(f"Lowest sugar food: **{lowest_sugar_food}** with **{lowest_sugar_food_val:.1f}g** sugar.")
                    st.markdown(f"Lowest sugar drink: **{lowest_sugar_drink}** with **{lowest_sugar_drink_val:.1f}g** sugar.")
                
                else:
                    avg_sugar_food = df_food['Estimated Sugar (g)'].mean()
                    avg_sugar_drink = df_drinks['Estimated Sugar (g)'].mean()
                    st.markdown(f"Average sugar in food: **{avg_sugar_food:.1f}g**")
                    st.markdown(f"Average sugar in drinks: **{avg_sugar_drink:.1f}g**")
            else:
                st.warning("For more complex questions, please enable the Groq LLM option and provide an API key.")
                
    # Health & nutrition resources
    with st.expander("Nutrition Resources"):
        st.markdown("""
        ### Understanding Nutrition Labels
        
        * **Calories**: A measure of energy from food
        * **Fat**: Essential nutrient needed in moderate amounts
        * **Protein**: Building blocks for muscles and tissues
        * **Carbohydrates**: Main source of energy
        * **Fiber**: Aids digestion and helps you feel full
        * **Sugar**: Includes natural and added sugars
        
        ### Daily Reference Values (based on 2000 calorie diet)
        
        * Total Fat: 65g
        * Saturated Fat: 20g
        * Cholesterol: 300mg
        * Sodium: 2,400mg
        * Total Carbohydrate: 300g
        * Dietary Fiber: 25g
        * Protein: 50g
        
        _Note: These are general guidelines. Individual needs may vary based on age, gender, weight, activity level, and overall health._
        """)

# Footer
st.markdown("---")
st.markdown("### About This App")
st.markdown("""
This Starbucks Nutrition Analysis Tool allows you to explore nutritional information 
from the Starbucks menu. The tool provides insights to help make informed food and beverage choices.

**Disclaimer**: This app is for informational purposes only. Nutritional information may vary
based on location, recipe changes, and customizations. Always consult with a healthcare 
professional for personalized dietary advice.
""")

# Add version info
st.sidebar.markdown("---")
st.sidebar.info("v1.0.0 | Created with Streamlit")
