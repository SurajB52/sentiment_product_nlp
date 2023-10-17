import pandas as pd
import os

files = ["Tools_and_Home_Improvement_5.json_sampled.csv","Toys_and_Games_5.json_sampled.csv","Video_Games_5.json_sampled.csv","All_Beauty_5.json_sampled.csv",
         "AMAZON_FASHION_5.json_sampled.csv","Appliances_5.json_sampled.csv","Arts_Crafts_and_Sewing_5.json_sampled.csv","Automotive_5.json_sampled.csv",
         "Books_5.json_sampled.csv","CDs_and_Vinyl_5.json_sampled.csv","Cell_Phones_and_Accessories_5.json_sampled.csv","Clothing_Shoes_and_Jewelry_5.json_sampled.csv",
         "Digital_Music_5.json_sampled.csv","Electronics_5.json_sampled.csv","Gift_Cards_5.json_sampled.csv","Grocery_and_Gourmet_Food_5.json_sampled.csv",
         "Home_and_Kitchen_5.json_sampled.csv","Industrial_and_Scientific_5.json_sampled.csv","Kindle_Store_5.json_sampled.csv","Luxury_Beauty_5.json_sampled.csv",
         "Magazine_Subscriptions_5.json_sampled.csv","Movies_and_TV_5.json_sampled.csv","Musical_Instruments_5.json_sampled.csv","Office_Products_5.json_sampled.csv",
         "Patio_Lawn_and_Garden_5.json_sampled.csv","Pet_Supplies_5.json_sampled.csv","Prime_Pantry_5.json_sampled.csv","Software_5.json_sampled.csv"]

# Path to the directory containing the CSV files
path_to_csvs = "H:/DS Projects/NLP Sentiment anlysis/data/processed/"

all_dataframes = []

for file in files:
    # Read each CSV file
    data = pd.read_csv(os.path.join(path_to_csvs, file))
    
    # Extract product category from filename and assign to a new column
    product_category = file.replace("_5.json_sampled.csv", "")
    data['product_category'] = product_category
    
    all_dataframes.append(data)
    print(file)

print(len(all_dataframes))
# Combine all dataframes
combined_data = pd.concat(all_dataframes, ignore_index=True)
print(len(combined_data))
# Save combined dataframe to a new CSV file
combined_data.to_csv(os.path.join(path_to_csvs,"product_reviews_processed_balanced_bert.csv"), index=False)