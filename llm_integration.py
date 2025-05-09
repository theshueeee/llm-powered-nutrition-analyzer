import pandas as pd
import groq

#LLM integration
# Function to prepare data for LLM
def prepare_data_summary(df_food, df_drinks, category="All", max_rows=50):
    # Reset index to make sure item names are in a column
    df_food_prep = df_food.reset_index()
    df_drinks_prep = df_drinks.reset_index()
    
    # Add Category
    df_food_prep['Category'] = 'Food'
    df_drinks_prep['Category'] = 'Drink'
    
    # Combine datasets
    df_combined = pd.concat([df_food_prep, df_drinks_prep], ignore_index=True)
    
    # Filter by category if specified
    if category.lower() == "food":
        df_combined = df_combined[df_combined["Category"] == "Food"]
    elif category.lower() == "drink" or category.lower() == "drinks":
        df_combined = df_combined[df_combined["Category"] == "Drink"]
    
    # Sort and truncate
    df_combined = df_combined.sort_values(by='Calories', ascending=False).head(max_rows)
    
    # Prepare context
    context = f"""
Starbucks Nutrition Dataset Snapshot ({category.title()} items, Top {max_rows} by Calories):

{df_combined.to_string(index=False)}

Each item includes:
- Name
- Calories
- Fat (g)
- Protein (g)
- Carb. (g)
- Fiber (g)

Note: Some drink items may contain caffeine based on name inference.
"""
    return context

# Function to query LLM
def query_nutrition_llm(df_food, df_drinks, user_question, api_key):
    try:
        # Initialize Groq client
        client = groq.Groq(api_key=api_key)
        
        # Prepare data context
        data_context = prepare_data_summary(df_food, df_drinks, max_rows=50)
        
        # Construct prompt
        prompt = f"""
You are a nutrition data analyst with access to a summarized Starbucks food and drink nutrition dataset.

Your task is to provide clear, data-driven answers to user questions, strictly using the information from the dataset summary below.

Please include relevant comparisons, maximum or minimum values, or averages if applicable, and clearly state when information is not available. Do not make assumptions outside the provided data.

--- Starbucks Menu Data Summary ---

{data_context}

User's Question:
{user_question}

Your answer should be:
- Fact-based and use numerical evidence where possible
- Don't show your thinking process in the output, just show the final answer please.
"""
        
        # Send to Groq LLM
        response = client.chat.completions.create(
            model="deepseek-r1-distill-llama-70b",
            messages=[
                {"role": "system", "content": "You are a helpful nutrition analyst who explains insights clearly using only the provided information."},
                {"role": "user", "content": prompt}
            ]
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error querying LLM: {str(e)}"
    
