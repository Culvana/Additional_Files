"""
Recipe Cost Calculator with Smart Unit Conversion and Matching
---------------------------------------------------------
Extract recipes, find best matches, convert units, calculate costs.
"""

import os
import json
import logging
import pandas as pd
from decimal import Decimal
from pathlib import Path
from datetime import datetime
from openai import OpenAI
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from dotenv import load_dotenv
import pint

# Initialize the unit registry
ureg = pint.UnitRegistry()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_clients():
    """Setup API clients."""
    load_dotenv()
    
    form_client = DocumentAnalysisClient(
        endpoint=os.getenv("AZURE_FORM_RECOGNIZER_ENDPOINT"),
        credential=AzureKeyCredential(os.getenv("AZURE_FORM_RECOGNIZER_KEY"))
    )
    
    search_client = SearchClient(
        endpoint=os.getenv("AZURE_AISEARCH_ENDPOINT"),
        index_name="mariachi-bakery",
        credential=AzureKeyCredential(os.getenv("AZURE_AISEARCH_APIKEY"))
    )
    
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    return form_client, search_client, openai_client



def get_ingredient_matches(ingredient, search_client, openai_client):
    """Get top 3 matches for ingredient and select best using GPT."""
    # Get embedding
    response = openai_client.embeddings.create(
        input=ingredient['item'].strip(),
        model="text-embedding-ada-002"
    )
    embedding = response.data[0].embedding
    
    # Search for top 3 matches
    vector_query = VectorizedQuery(
        vector=embedding,
        k_nearest_neighbors=3,
        fields="inventory_item_vector"
    )
    
    results = list(search_client.search(
        search_text=None,
        vector_queries=[vector_query],
        select=["inventory_item", "cost_per_unit", "measured_in","supplier_name"],
        top=3
    ))
    
    if not results:
        return None
    
    # Format matches for GPT
    matches = []
    for idx, match in enumerate(results, 1):
        matches.append({
            'rank': idx,
            'inventory_item': match['inventory_item'],
            'cost_per_unit': Decimal(str(match['cost_per_unit'])),
            'unit': match['measured_in'],
            'supplier': match.get('supplier_name', 'Unknown')
        })
    
    # Use GPT to select best match
    match_text = "\n".join(
        f"{m['rank']}. {m['inventory_item']} ({m['unit']}) from {m['supplier']}"
        for m in matches
    )
    
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": """Select the best matching inventory item.
            Consider:
            1. Name similarity
            2. Unit compatibility
            3. Common usage in recipes
            Rank from 1 (most likely) to 3 (least likely).
            Return only the number of the best match."""},
            {"role": "user", "content": f"Recipe ingredient: {ingredient['item']}\nAmount: {ingredient['amount']} {ingredient['unit']}\nMatches:\n{match_text}"}
        ],
        response_format={"type": "text"}
    )
    
    try:
        best_idx = int(response.choices[0].message.content.strip()) - 1
        return matches[best_idx]
    except:
        return matches[0]  # Default to first match if GPT response is invalid


# Unit conversion constants and mappings
UNIT_CONVERSIONS = {
    # Volume conversions
    'volume': {
        'gallon': {'ml': Decimal('3785.41'), 'l': Decimal('3.78541'), 'oz': Decimal('128'), 'cc': Decimal('3785.41')},
        'liter': {'ml': Decimal('1000'), 'oz': Decimal('33.814'), 'gallon': Decimal('0.264172'), 'cc': Decimal('1000')},
        'ml': {'l': Decimal('0.001'), 'oz': Decimal('0.033814'), 'gallon': Decimal('0.000264172'), 'cc': Decimal('1')},
        'cc': {'ml': Decimal('1'), 'l': Decimal('0.001'), 'oz': Decimal('0.033814'), 'gallon': Decimal('0.000264172')},
        'oz': {'ml': Decimal('29.5735'), 'l': Decimal('0.0295735'), 'gallon': Decimal('0.0078125'), 'cc': Decimal('29.5735')},
        'cup': {'oz': Decimal('8'), 'ml': Decimal('236.588'), 'l': Decimal('0.236588'), 'cc': Decimal('236.588'), 'tbsp': Decimal('16')},
        'tbsp': {'oz': Decimal('0.5'), 'ml': Decimal('14.7868'), 'tsp': Decimal('3'), 'cc': Decimal('14.7868'), 'cup': Decimal('0.0625')},
        'tsp': {'oz': Decimal('0.166667'), 'ml': Decimal('4.92892'), 'tbsp': Decimal('0.333333'), 'cc': Decimal('4.92892')},
        'scoop': {'cup': Decimal('0.5'), 'oz': Decimal('2'), 'ml': Decimal('118.294'), 'cc': Decimal('118.294')},
        'dash': {'ml': Decimal('0.616115'), 'tsp': Decimal('0.125'), 'cc': Decimal('0.616115')},
        'pinch': {'tsp': Decimal('0.0625'), 'ml': Decimal('0.308057'), 'cc': Decimal('0.308057')}
    },
    # Weight conversions
    'weight': {
        'lb': {'oz': Decimal('16'), 'g': Decimal('453.592'), 'kg': Decimal('0.453592')},
        'oz': {'g': Decimal('28.3495'), 'kg': Decimal('0.0283495'), 'lb': Decimal('0.0625')},
        'kg': {'g': Decimal('1000'), 'lb': Decimal('2.20462'), 'oz': Decimal('35.274')},
        'g': {'kg': Decimal('0.001'), 'lb': Decimal('0.00220462'), 'oz': Decimal('0.035274')},
        'stone': {'lb': Decimal('14'), 'kg': Decimal('6.35029'), 'g': Decimal('6350.29')},
        'grain': {'g': Decimal('0.06479891'), 'oz': Decimal('0.002285714')},
        'dram': {'oz': Decimal('0.0625'), 'g': Decimal('1.7718451953125')}
    }
}

# Unit standardization mappings
UNIT_STANDARDIZATION = {
    # Volume units
    'fluid ounce': 'oz', 'fluid ounces': 'oz', 'fl oz': 'oz', 'fl. oz.': 'oz',
    'milliliter': 'ml', 'milliliters': 'ml', 'millilitre': 'ml', 'millilitres': 'ml',
    'cubic centimeter': 'cc', 'cubic centimeters': 'cc', 'cubic centimetre': 'cc', 'cubic centimetres': 'cc',
    'liter': 'l', 'liters': 'l', 'litre': 'l', 'litres': 'l',
    'gallon': 'gallon', 'gallons': 'gallon', 'gal': 'gallon', 'gal.': 'gallon',
    'cup': 'cup', 'cups': 'cup', 'c.': 'cup', 'c': 'cup',
    'tablespoon': 'tbsp', 'tablespoons': 'tbsp', 'tbl': 'tbsp', 'tbs': 'tbsp', 'tbsp.': 'tbsp', 
    'big spoon': 'tbsp', 'soup spoon': 'tbsp', 'serving spoon': 'tbsp',
    'teaspoon': 'tsp', 'teaspoons': 'tsp', 'tsp.': 'tsp', 
    'small spoon': 'tsp', 'coffee spoon': 'tsp', 'dessert spoon': 'tbsp',
    'scoop': 'scoop', 'scoops': 'scoop', 'ice cream scoop': 'scoop',
    'dash': 'dash', 'dashes': 'dash',
    'pinch': 'pinch', 'pinches': 'pinch',
    # Weight units
    'pound': 'lb', 'pounds': 'lb', 'lbs': 'lb', 'lb.': 'lb',
    'ounce': 'oz', 'ounces': 'oz', 'oz.': 'oz',
    'gram': 'g', 'grams': 'g', 'g.': 'g',
    'kilogram': 'kg', 'kilograms': 'kg', 'kgs': 'kg', 'kg.': 'kg',
    'stone': 'stone', 'stones': 'stone', 'st': 'stone', 'st.': 'stone',
    'grain': 'grain', 'grains': 'grain', 'gr': 'grain', 'gr.': 'grain',
    'dram': 'dram', 'drams': 'dram', 'dr': 'dram', 'dr.': 'dram'
}

def standardize_unit(unit):
    """Standardize unit notation to a common format."""
    unit = unit.lower().strip()
    return UNIT_STANDARDIZATION.get(unit, unit)

def determine_unit_type(unit):
    """Determine if unit is volume or weight."""
    unit = standardize_unit(unit)
    for type_name, conversions in UNIT_CONVERSIONS.items():
        if any(unit == std_unit for std_unit in conversions.keys()):
            return type_name
    return None

def get_conversion_path(from_unit, to_unit):
    """Find conversion path between units."""
    from_unit = standardize_unit(from_unit)
    to_unit = standardize_unit(to_unit)
    
    # Check if units are the same
    if from_unit == to_unit:
        return None, Decimal('1')
    
    # Determine unit types
    from_type = determine_unit_type(from_unit)
    to_type = determine_unit_type(to_unit)
    
    if not from_type or not to_type:
        return None, None
    
    # Must be same type of unit
    if from_type != to_type:
        return None, None
    
    # Direct conversion available
    if to_unit in UNIT_CONVERSIONS[from_type][from_unit]:
        return from_type, UNIT_CONVERSIONS[from_type][from_unit][to_unit]
    
    return from_type, None


def convert_units(amount, from_unit, to_unit):
    try:
        # Create quantity with Pint
        quantity = float(amount) * ureg(from_unit)
        # Convert to target unit
        converted_quantity = quantity.to(to_unit)
        return Decimal(str(converted_quantity.magnitude))
    except Exception as e:
        logger.error(f"Unit conversion error: [{type(e)}] {str(e)}")
        raise


###def convert_units(amount, from_unit, to_unit, openai_client=None):
    """
    Convert between units using fixed rates and GPT for verification.
    
    Args:
        amount (Decimal or float): Amount to convert
        from_unit (str): Original unit
        to_unit (str): Target unit
        openai_client (Optional[OpenAI]): OpenAI client for verification
    
    Returns:
        Decimal: Converted amount
    """
    try:
        amount = Decimal(str(amount))
        from_unit = (from_unit)
        to_unit = (to_unit)
        
        # Same unit, no conversion needed
        if from_unit == to_unit:
            return amount
        if openai_client:
                    response = openai_client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": """Convert between units.
                            Return only the numerical conversion factor as a decimal number. THis can be scoop to ounces or any other conversion factor. 
                            Example: For '1 cup to oz', return '8'"""},
                            {"role": "user", "content": f"Convert 1 {from_unit} to {to_unit}"}
                        ]
                    )
                    conversion_factor = Decimal(response.choices[0].message.content.strip())
                    converted = amount * conversion_factor
        else:
                    raise ValueError(f"No conversion path found for {from_unit} to {to_unit}")
        
        # Round to 6 decimal places to avoid floating point issues
        return converted
        
    except Exception as e:
        logger.error(f"Unit conversion error: [{type(e)}] {str(e)}")
        raise

def calculate_ingredient_cost(ingredient, match, openai_client):
    """Calculate cost for single ingredient with unit conversion."""
    try:
        # Extract amounts and units
        recipe_amount = Decimal(str(ingredient['amount']))
        recipe_unit = ingredient['unit']
        inventory_unit = match['unit']
        
        # Convert recipe amount to inventory unit
        converted_amount = convert_units(
            recipe_amount,
            recipe_unit,
            inventory_unit
        )
        
        # Calculate cost
        unit_cost = Decimal(str(match['cost_per_unit']))
        total_cost = converted_amount * unit_cost
        
        return {
            'ingredient': ingredient['item'],
            'recipe_amount': f"{recipe_amount} {recipe_unit}",
            'inventory_item': match['inventory_item'],
            'inventory_unit': inventory_unit,
            'converted_amount': float(converted_amount),
            'unit_cost': float(unit_cost),
            'total_cost': float(total_cost)
        }
        
    except Exception as e:
        logger.error(f"Cost calculation error: [{type(e)}] {str(e)}")
        return None



def export_to_excel(recipes_with_costs, output_path):
    """Export recipe costs to Excel with detailed sheets."""
    with pd.ExcelWriter(output_path) as writer:
        # Recipe Summary Sheet
        summary_data = []
        for recipe in recipes_with_costs:
            summary_data.append({
                'Recipe Name': recipe['recipe_name'],
                'Total Cost': f"${recipe['total_cost']:.2f}",

                'Ingredient Count': len(recipe['ingredients']),
                'Topping': recipe['topping']
            })
        pd.DataFrame(summary_data).to_excel(writer, sheet_name='Recipe Summary', index=False)
        
        # Detailed Ingredients Sheet
        ingredient_data = []
        for recipe in recipes_with_costs:
            for ing in recipe['ingredients']:
                ingredient_data.append({
                    'Recipe Name': recipe['recipe_name'],
                    'Ingredient': ing['ingredient'],
                    'Recipe Amount': ing['recipe_amount'],
                    'Inventory Item': ing['inventory_item'],
                    'Converted Amount': f"{ing['converted_amount']:.3f} {ing['inventory_unit']}",
                    'Unit Cost': f"${ing['unit_cost']:.3f}",
                    'Total Cost': f"${ing['total_cost']:.2f}"
                })
        pd.DataFrame(ingredient_data).to_excel(writer, sheet_name='Ingredient Details', index=False)

def extract_recipes_from_pdf(pdf_path, form_client, openai_client):
    """Extract recipes from PDF."""
    with open(pdf_path, "rb") as doc:
        result = form_client.begin_analyze_document("prebuilt-layout", doc).result()
        content = {
            "tables": [[cell.content.strip() for cell in table.cells] for table in result.tables],
            "text": [p.content.strip() for p in result.paragraphs]
        }
    Systemprompt= """You are a precise recipe extraction specialist. Your task is to extract and standardize recipe information from any source while maintaining a consistent structure.

EXTRACTION RULES:
Extract all recipes from the content provided.
1. Extract ALL recipes from the provided content
2. Maintain exact measurements and units
3. Convert all text numbers to numeric values (e.g., "one" → 1)
4. Standardize ingredients to their base names
5. Capture complete procedures 

OUTPUT STRUCTURE:
Return data in this EXACT JSON format:
{
    "recipes": [
        {
            "name": "Complete Recipe Name with Size/Yield",
            "ingredients": [
                {
                    "item": "ingredient base name",
                    "amount": number,
                    "unit": "standardized unit"
                }
            ],
            "topping": "complete topping instructions"
        }
    ]
}

STANDARDIZATION RULES:

1. Units: Use these standard units ONLY:Make sure to Convert to abbreviations
   Volume:
   - "ml" (milliliters)
   - "l" (liters)
   - "oz" (fluid ounces)
   - "cup" (cups)
   - "tbsp" (tablespoons)
   - "tsp" (teaspoons)
   - "gallon" (gallons)
   - " cc" (cubic centimeters)
   
   Weight:
   - "g" (grams)
   - "kg" (kilograms)
   - "lb" (pounds)
   - "oz" (ounces for weight)

2. Numbers:
   - Convert all written numbers to numerals
   - Convert fractions to decimals
   - Round to 2 decimal places
   - Examples:
     * "one" → 1
     * "half" → 0.5
     * "1/4" → 0.25
     * "2 1/2" → 2.5

3. Ingredients:
   - Use base ingredient names
   - Include preparation state in name if critical
   - Examples:
     * "pure vanilla extract" → "vanilla extract"
     * "cold butter, cubed" → "butter"

4. Measurements:
   - Convert all measurements to standard units
   - Handle common conversions:
     * "stick of butter" → 0.5, "cup"
     * "large egg" → 1, "unit"
     * "pinch" → 0.125, "tsp"

5. Topping Instructions:
   - Include complete application method
   - Maintain sequence of steps
   - Include any critical timing or temperature notes

VALIDATION REQUIREMENTS:
1. Every ingredient MUST have:
   - Non-empty "item" name
   - Numeric "amount"
   - Valid "unit" from standardized list

2. Every recipe MUST have:
   - Complete "name"
   - At least one ingredient
   - Either topping instructions or empty string ""

3. Numbers:
   - All amounts must be positive numbers
   - No text-based numbers allowed
   - No ranges (use average if range given)

HANDLING SPECIAL CASES:
1. Missing Measurements:
   - For "to taste" → use minimum recommended amount
   - For "as needed" → use typical serving amount
   - For decorative items → use minimum functional amount

2. Alternative Ingredients:
   - List primary ingredient only
   - Ignore "or" alternatives

3. Optional Ingredients:
   - Include in main list
   - Use minimum recommended amount

Return ONLY the JSON with no additional text or explanations."""

    
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": Systemprompt},
            {"role": "user", "content": str(content)}
        ],
        response_format={"type": "json_object"}
    )
    logging.info(response.choices[0].message.content)
    
    try:
        recipes_data = json.loads(response.choices[0].message.content)
        return recipes_data
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {e}")
        raise
    except ValueError as e:
        logger.error(f"Data validation error: {e}")
        raise

def calculate_recipe_cost(recipe, search_client, openai_client):
    """Calculate total recipe cost."""
    if not isinstance(recipe, dict) or 'name' not in recipe or 'ingredients' not in recipe:
        logger.error(f"Invalid recipe format: {recipe}")
        raise ValueError("Invalid recipe format")
        
    logger.info(f"Calculating cost for: {recipe['name']}")
    
    ingredient_costs = []
    total_cost = Decimal('0')
    
    for ingredient in recipe['ingredients']:
        # Validate ingredient structure
        if not isinstance(ingredient, dict) or not all(key in ingredient for key in ['item', 'amount', 'unit']):
            logger.warning(f"Skipping invalid ingredient: {ingredient}")
            continue
            
        # Find best matching inventory item
        match = get_ingredient_matches(ingredient, search_client, openai_client)
        if not match:
            logger.warning(f"No match found for ingredient: {ingredient['item']}")
            continue
            
        # Calculate cost with unit conversion
        try:
            cost_info = calculate_ingredient_cost(ingredient, match, openai_client)
            if cost_info:
                ingredient_costs.append(cost_info)
                total_cost += Decimal(str(cost_info['total_cost']))
        except Exception as e:
            logger.error(f"Error calculating cost for {ingredient['item']}: {e}")
            continue
    
    return {
        'recipe_name': recipe['name'],
        'ingredients': ingredient_costs,
        'total_cost': float(total_cost),
        'topping': recipe.get('topping', '')
    }

def process_recipes(pdf_path):
    """Process recipes from PDF to cost analysis."""
    try:
        # Setup
        form_client, search_client, openai_client = setup_clients()
        
        # Extract recipes
        logger.info("Extracting recipes from PDF...")
        recipes_data = extract_recipes_from_pdf(pdf_path, form_client, openai_client)
        
        if not isinstance(recipes_data, dict) or 'recipes' not in recipes_data:
            raise ValueError("Invalid recipes data structure")
        
        # Calculate costs
        logger.info("Calculating recipe costs...")
        recipe_costs = []
        for recipe in recipes_data['recipes']:
            try:
                cost_info = calculate_recipe_cost(recipe, search_client, openai_client)
                recipe_costs.append(cost_info)
            except Exception as e:
                logger.error(f"Error processing recipe {recipe.get('name', 'unknown')}: {e}")
                continue
        
        if not recipe_costs:
            raise ValueError("No recipes were successfully processed")
        
        # Export results
        output_path = f"recipe_costs_{datetime.now():%Y%m%d_%H%M}.xlsx"
        logger.info("Exporting to Excel...")
        export_to_excel(recipe_costs, output_path)
        
        return output_path
        
    except Exception as e:
        logger.error(f"Error processing recipes: {e}")
        raise

def main():
    """Run the recipe analyzer."""
    try:
        ##"C:\Users\rahul\Downloads\Chai Recipe.pages.pdf"
        pdf_path = "C:/Users/rahul/Downloads/Chai Recipe.pages.pdf"
        output_path = process_recipes(pdf_path)
        print(f"\nAnalysis complete! Results saved to: {output_path}")
    except Exception as e:
        print(f"Error: {e}")
        exit(1)

if __name__ == "__main__":
    main()