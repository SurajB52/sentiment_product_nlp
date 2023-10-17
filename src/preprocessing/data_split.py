from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv(r"H:\DS Projects\NLP Sentiment anlysis\data\processed\product_reviews_processed_balanced_bert.csv")

# Split the data into training and temporary set
train_df, temp_df = train_test_split(df, random_state=42, test_size=0.2)

# Split the temporary set into validation and test sets
validation_df, test_df = train_test_split(temp_df, random_state=42, test_size=0.5)

# You can save them to separate files if needed
train_df.to_csv('train_data_bert.csv', index=False)
validation_df.to_csv('validation_data_bert.csv', index=False)
test_df.to_csv('test_data_bert.csv', index=False)
