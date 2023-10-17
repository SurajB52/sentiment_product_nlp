import streamlit as st
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
import tensorflow as tf

# Suppress some warnings from TensorFlow
tf.get_logger().setLevel('ERROR')

class SessionState(object):
    def __init__(self, **kwargs):
        """Initialize a new session state object."""
        for key, val in kwargs.items():
            setattr(self, key, val)

@st.cache_resource()
def get_session_state():
    return SessionState(text_input="", button_sent=False)


# Function to load the model
def load_model():
    model_path = "H:\\DS Projects\\NLP Sentiment anlysis\\data\\models\\sentiment_analysis_distilbert_model"  # Adjust with your model path
    model = TFDistilBertForSequenceClassification.from_pretrained(model_path)
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    return model, tokenizer

# Function to tokenize sentences
def tokenize_sentences(tokenizer, sentences, max_length=128):
    return tokenizer(sentences, max_length=max_length, padding=True, truncation=True, return_attention_mask=True, return_tensors='tf')

         
# Function to predict and display sentiment
def predict_sentiment_detailed(model, tokenizer,sentence):
    # Load the model and tokenizer
    # model, tokenizer = load_model()
    # Tokenize the sentence and predict sentiment
    inputs = tokenize_sentences(tokenizer, [sentence])
    outputs = model.predict({'input_ids': inputs['input_ids'], 'attention_mask': inputs['attention_mask']})

    # Calculate probabilities and determine the predicted class
    probabilities = tf.nn.softmax(outputs.logits[0]).numpy()
    class_names = ['negative', 'neutral', 'positive']
    sentiment_scores = {class_name: prob for class_name, prob in zip(class_names, probabilities)}
    predicted_class = class_names[np.argmax(probabilities)]
    return sentiment_scores, predicted_class

def get_random_example():
    example_reviews = [
    "Apple iPhone 12 Pro: This phone is amazing! The camera quality is outstanding, and it's super fast.",
    "Bose QuietComfort 35 II: These headphones are a game-changer for noise cancellation. Sound quality is top-notch.",
    "Instant Pot Duo: Love this kitchen appliance! It makes cooking so much easier and faster.",
    "Samsung 4K QLED TV: The picture quality on this TV is breathtaking. I'm so happy with my purchase!",
    "Fitbit Charge 4: This fitness tracker has helped me stay on top of my health goals. Highly recommend it!",
    "Kindle Paperwhite: The Kindle Paperwhite is a must-have for book lovers. It's so convenient and easy to read on.",
    "Dyson V11 Vacuum: The Dyson V11 is a beast at cleaning. My floors have never been cleaner!",
    "Nintendo Switch: Hours of entertainment! Great for gaming alone or with friends.",
    "Roku Streaming Stick: So simple to use and offers a wide range of streaming options.",
    "Canon EOS 5D Mark IV: The image quality from this camera is outstanding. Worth every penny!",
    "Apple iPhone 12 Pro: The battery life on this phone is disappointing. I have to charge it multiple times a day.",
    "Bose QuietComfort 35 II: These headphones are overpriced, and the build quality is subpar.",
    "Instant Pot Duo: The Instant Pot stopped working after just a few uses. Very disappointed.",
    "Samsung 4K QLED TV: This TV has a lot of issues with screen flickering. Not happy with my purchase.",
    "Fitbit Charge 4: The heart rate monitor on this fitness tracker is not accurate at all.",
    "Kindle Paperwhite: I find the screen to be too reflective, especially in bright light.",
    "Dyson V11 Vacuum: It's too heavy and bulky to maneuver around the house easily.",
    "Nintendo Switch: The Joy-Con controllers drift issue is frustrating and ruins the gaming experience.",
    "Roku Streaming Stick: The remote control is flimsy and often unresponsive.",
    "Canon EOS 5D Mark IV: This camera is too complex for beginners. I struggled to understand its features.",
    "Apple iPhone 12 Pro: It's a decent phone, but there's nothing groundbreaking about it.",
    "Bose QuietComfort 35 II: These headphones are okay, but I expected better for the price.",
    "Instant Pot Duo: It's good for some recipes, but not suitable for all types of cooking.",
    "Samsung 4K QLED TV: The picture quality is good, but I've experienced some software glitches.",
    "Fitbit Charge 4: It's an average fitness tracker with standard features.",
    "Kindle Paperwhite: It's convenient for reading, but the battery life could be better.",
    "Dyson V11 Vacuum: It gets the job done, but it's expensive compared to other vacuums.",
    "Nintendo Switch: It's fun for gaming, but the graphics could be better.",
    "Roku Streaming Stick: It's a simple streaming device, but not very impressive.",
    "Canon EOS 5D Mark IV: This camera has a bad learning curve, and it's not for everyone."
]

    return random.choice(example_reviews)

def main():

    session_state = get_session_state()
    
    st.title("Sentiment Analysis Tool")

    # User input section
    st.subheader("Click button to Analyze your text")
    
    # Guidance for example reviews
    st.markdown("_Need inspiration? Click the button below to auto-fill a random example review!_")

    # Handling the text input with session state
    sentence = session_state.text_input
    sentence = st.text_area("Enter text here...", value=sentence, key="input_text_area")
    st.write("Hint: You can enter multiple reviews separated by ';' for bulk analysis.")

    if st.button('Click Here to Get Example'):
        # Fetch a random example and update the session state
        session_state.text_input = get_random_example()
        sentence = session_state.text_input  # Ensure the current sentence is updated


    # File upload guidance and feature
    st.markdown("#### 📁 Alternatively you can Upload your text or CSV file here:")
    uploaded_file = st.file_uploader("Choose a text or CSV file", type=['txt', 'csv'])
    st.write("1. If you upload csv file, make sure first column contains trview text.")
    st.write("2. If you upload text file, make sure every line has review text")
    if uploaded_file is not None:
        # Process the uploaded file
        if uploaded_file.type == "text/csv":
            # Use Pandas to process the CSV file
            data = pd.read_csv(uploaded_file)
            
            # If we're assuming the texts are in the first column regardless of header name
            if data.shape[1] < 1:  # if there's no column at all
                st.error("CSV file is empty or does not have any columns.")
            else:
                first_column = data.columns[0]  # Get the name of the first column
                reviews = data[first_column].dropna().tolist()  # Extract the reviews and drop empty entries
        # Dropping NA values and converting to a list
        elif uploaded_file.type == "text/plain":
            # Read the contents of the text file
            content = uploaded_file.getvalue().decode("utf-8")
            reviews = content.split("\n")  # Split by new lines to get individual reviews
        else:
            st.error("Unsupported file type.")
            reviews = []

        # Join reviews with ';' for processing, as your system supports multiple reviews separated by ';'
        sentence = ";".join(reviews)

    if st.button('Click here to analyze the text', key='analyze', help='Click here to analyze the sentiment of the text'):
        if sentence:
            sentences = sentence.split(';')
            results = []

            with st.spinner('Analyzing...'):
                model, tokenizer = load_model()  # Load the model

                for single_sentence in sentences:
                    sentiment_scores, predicted_class = predict_sentiment_detailed(model, tokenizer, single_sentence)
                    results.append((single_sentence, sentiment_scores, predicted_class))

            # Display results with color-coding and bar charts
            for single_sentence, sentiment_scores, predicted_class in results:
                emoji = "😢" if predicted_class == "negative" else "😐" if predicted_class == "neutral" else "😄"
                # Color-coding based on sentiment
                if predicted_class == 'positive':
                    color = 'green'
                elif predicted_class == 'neutral':
                    color = 'blue'
                else:
                    color = 'red'

                st.markdown(f"**Sentiment:** <span style='color: {color};'>{predicted_class}</span> {emoji}", unsafe_allow_html=True)
                st.markdown(f"**Your Text:** {single_sentence}\n")
                
                # Creating a bar chart for sentiment scores
                fig, ax = plt.subplots()
                ax.barh(list(sentiment_scores.keys()), list(sentiment_scores.values()), color=['red', 'yellow', 'green'])
                plt.xlabel('Probability')
                plt.ylabel('Sentiment')
                plt.title('Sentiment Analysis Scores')
                st.pyplot(fig)
            session_state.button_sent = False

            # Feedback mechanism
            feedback = st.text_input("Was this analysis correct? If not, what should the sentiment be?", key='feedback_input')
            if st.button('Submit Feedback', key='feedback_button'):
                st.write("Thank you for your feedback!")
                # Here, you can add code to save feedback for future model improvement
        else:
            st.write("Please enter some text or upload a file before analyzing.")

if __name__ == "__main__":
    main()


# # Educational content about Sentiment Analysis and BERT
#     st.write(
#         """
#         ## What is Sentiment Analysis?
#         Sentiment Analysis is like reading people's minds! Well, kind of. It's a technique that computers use to understand if the words people write are happy, sad, or neutral. This way, companies know what their customers are feeling about their products or services.

#         ## How does BERT help in Sentiment Analysis?
#         Imagine understanding all the languages in the world! BERT is like that. It's a brain for the computer that understands not just words, but also the context, like sarcasm or excitement. It's super smart because it got trained with lots of books, websites, and all sorts of writings. So, when you tell BERT about any sentence or a review, it can tell if it's a happy, sad, or meh message.

#         ## The Magic Behind Training BERT
#         BERT went to a special school called 'Fine-tuning' where it learned about specific stuff like product reviews. It read millions of comments about different products, like toys, phones, or pizzas, understanding what makes customers happy or grumpy. Now, BERT is like a sentiment wizard for all sorts of product reviews!
#         """
#     )
