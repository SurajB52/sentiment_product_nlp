#Libraries
import json
import re
from bs4 import BeautifulSoup
from transformers import BertTokenizer
import tensorflow as tf
import numpy as np
import random
import csv

# Initialize BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def get_line_numbers_for_ratings(filename, ratings=[5,4,3,2,1], sample_size_per_rating=7000):
    ratings_count = {rating: 0 for rating in ratings}
    line_ratings = {}
    
    with open(filename, 'r', encoding='utf-8') as f:
        for line_number, line in enumerate(f):
            entry = json.loads(line)
            overall = entry.get('overall')
            if overall in ratings:
                ratings_count[overall] += 1
                line_ratings[line_number] = overall
                
    min_samples = min([ratings_count[rating] for rating in ratings])
    if min_samples < sample_size_per_rating:
        sample_size_per_rating = min_samples

    sampled_line_numbers = []
    for rating in ratings:
        rating_lines = [ln for ln, r in line_ratings.items() if r == rating]
        sampled_line_numbers.extend(random.sample(rating_lines, sample_size_per_rating))
    print(f"Sample lines returned for file {filename}")
    return set(sampled_line_numbers)

def process_data_sample(filename, sample_size_per_rating=7000):
    line_numbers_to_sample = get_line_numbers_for_ratings(f'H:/DS Projects/NLP Sentiment anlysis/data/raw/{filename}', sample_size_per_rating=sample_size_per_rating)
    
    processed_data = []

    with open(f'H:/DS Projects/NLP Sentiment anlysis/data/raw/{filename}', 'r', encoding='utf-8') as f:
        i=0
        for line_number, line in enumerate(f):
            if line_number not in line_numbers_to_sample:
                continue
            
            entry = json.loads(line)
            review_text = entry.get('reviewText', '')
            reviewer_id = entry.get('reviewerID', '').strip()
            
            if not review_text or not reviewer_id:
                continue  # skip the entry if either field is missing or empty
            
            # Remove duplicates (Assuming reviewText and reviewerID together makes an entry unique)
            if any(d.get('reviewText', '') == review_text and d.get('reviewerID', '') == entry.get('reviewerID', '') for d in processed_data):
                continue
            
            # Remove special characters
            review_text = re.sub(r'[^a-zA-Z0-9\s]', '', review_text)

            # BERT Tokenization
            tokens = tokenizer.tokenize(review_text)
            token_ids = tokenizer.convert_tokens_to_ids(tokens)
            
            # Handling Long Sequences
            if len(token_ids) > 510:  # reserving 2 for [CLS] and [SEP]
                token_ids = token_ids[:510]

            # Attention Masks and Segment ID
            attention_masks = [1] * len(token_ids)
            # segment_ids = [0] * len(token_ids)

            # Add [CLS] and [SEP] special tokens
            token_ids = [tokenizer.cls_token_id] + token_ids + [tokenizer.sep_token_id]
            attention_masks = [1] + attention_masks + [1]
            # segment_ids = [0] + segment_ids + [0]

            # Padding for equal length
            padding_length = 512 - len(token_ids)
            token_ids += [tokenizer.pad_token_id] * padding_length
            attention_masks += [0] * padding_length
            # segment_ids += [0] * padding_length
            
            processed_entry = {
                'reviewText': review_text,
                'token_ids': token_ids,
                'attention_mask': attention_masks,
                'overall': entry.get('overall', 0)
            }
            
            processed_data.append(processed_entry)
            i+=1
            if i%1000==0:
                print(i,' times for file ',filename)
                
    # Save to CSV format
    output_path = f'H:/DS Projects/NLP Sentiment anlysis/data/processed/{filename}_sampled.csv'
    with open(output_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=processed_data[0].keys())
        writer.writeheader()
        for row in processed_data:
            writer.writerow(row)

filenames = ["Cell_Phones_and_Accessories_5.json","Clothing_Shoes_and_Jewelry_5.json",
            "Digital_Music_5.json","Electronics_5.json","Gift_Cards_5.json","Grocery_and_Gourmet_Food_5.json",
            "Home_and_Kitchen_5.json","Industrial_and_Scientific_5.json","Kindle_Store_5.json",
            "Luxury_Beauty_5.json","Magazine_Subscriptions_5.json","Movies_and_TV_5.json",
            "Musical_Instruments_5.json","Office_Products_5.json","Patio_Lawn_and_Garden_5.json",
            "Pet_Supplies_5.json","Prime_Pantry_5.json","Tools_and_Home_Improvement_5.json",
            "Toys_and_Games_5.json","Video_Games_5.json","All_Beauty_5.json",
            "AMAZON_FASHION_5.json","Appliances_5.json",
            "Arts_Crafts_and_Sewing_5.json","Automotive_5.json","Books_5.json"]
    
for fname in filenames:
    process_data_sample(fname)