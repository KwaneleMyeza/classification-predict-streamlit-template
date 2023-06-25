"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import re
import streamlit as st
import joblib
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from wordcloud import WordCloud


# Vectorizer
news_vectorizer = open("Streamlit/vectorizer.pkl", "rb")
tweet_cv = joblib.load(news_vectorizer)

# Load your raw data
raw = pd.read_csv("resources/train.csv")


def switch_case(case):
    if case == 1:
        return 'Pro: In a parliament which includes climate change deniers, austerity junkies &amp; voracious vandals of the cultura'
    elif case == 2:
        return 'Swedish politicians troll Trump administration while signing climate change law'
    elif case == 0:
        return 'Neutral: Look who's back in town! And thank you, global warming'
    elif case == -1:
        return 'Anti: Hey I tweeted about the Man made global warming scam and immediately lost a couple of followers'


def pie_chart():
	"""Docstring"""
        
	st.markdown("")
	st.markdown("##### Sentiment Comparison")
	st.markdown("Shows the percentage quantity of the climate change sentiment in the data collection.")
	st.markdown("")
    
	#plotting pie chart
	values = raw["sentiment"].value_counts()/raw.shape[0]
	labels = (raw["sentiment"].value_counts()/raw.shape[0]).index
	plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=90, explode=(0.04,0.02,0.02,0.02), radius=2)
	fig, ax = plt.subplots()  # Create the figure object
	plt.tight_layout()
	# plt.show()
	st.pyplot(fig)


def word_cloud():
    """Docstring"""

    st.markdown("")
    st.markdown("##### Frequent Words")
    st.markdown("Shows the most frequent words that appear in tweets for all sentiments.")
    st.markdown("")
    # Concatenate the tweets based on sentiment
    news_tweets = " ".join(raw.loc[raw['sentiment'] == 2, 'message'])
    pro_tweets = " ".join(raw.loc[raw['sentiment'] == 1, 'message'])
    neutral_tweets = " ".join(raw.loc[raw['sentiment'] == 0, 'message'])
    anti_tweets = " ".join(raw.loc[raw['sentiment'] == -1, 'message'])

	# Create subplots for word clouds
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Generate word clouds for each sentiment
    news_wordcloud = WordCloud(max_words=40, random_state=21, max_font_size=100, collocations=False,
                           background_color='white', colormap="cool_r").generate(news_tweets)
    axes[0, 0].imshow(news_wordcloud, interpolation="bilinear")
    axes[0, 0].set_title('News Climate Change Frequent Words', fontsize=25)
    axes[0, 0].axis('off')

    pro_wordcloud = WordCloud(max_words=40, random_state=100, max_font_size=100, collocations=False,
                          background_color='white', colormap="BuPu").generate(pro_tweets)
    axes[0, 1].imshow(pro_wordcloud, interpolation="bilinear")
    axes[0, 1].set_title('Pro Climate Change Frequent Words', fontsize=25)
    axes[0, 1].axis('off')

    neutral_wordcloud = WordCloud(max_words=40, random_state=21, max_font_size=100, collocations=False,
								background_color='white', colormap="plasma").generate(neutral_tweets)
    axes[1, 0].imshow(neutral_wordcloud, interpolation="bilinear")
    axes[1, 0].set_title('Neutral Climate Change Frequent Words', fontsize=25)
    axes[1, 0].axis('off')

    anti_wordcloud = WordCloud(max_words=40, random_state=100, max_font_size=100, collocations=False,
							background_color='white', colormap="cool").generate(anti_tweets)
    axes[1, 1].imshow(anti_wordcloud, interpolation="bilinear")
    axes[1, 1].set_title('Anti Climate Change Frequent Words', fontsize=25)
    axes[1, 1].axis('off')

	# Adjust the spacing between subplots
    plt.tight_layout()

	# Display the word cloud visualization
    st.pyplot(fig)
    

def bar_chart():
    """Docstring"""

    st.markdown("")
    st.markdown("##### Sentiment Distribution")
    st.markdown("Shows the distribution of tweets that show the different sentiments.")
    st.markdown("")
    # Count the number of tweets in each sentiment class
    sentiment_counts = raw['sentiment'].value_counts()

    # Define the labels and colors for the bar chart
    labels = ['News', 'Pro', 'Neutral', 'Anti']
    colors = ['blue', 'green', 'gray', 'red']

    # Create the bar chart
    fig, ax = plt.subplots()
    bars = ax.bar(labels, sentiment_counts, color=colors)

    # Set the chart title and labels
    ax.set_title('Distribution of Sentiments in Training Data')
    ax.set_xlabel('Sentiment')
    ax.set_ylabel('Number of Tweets')

    # Enable interactivity to show/hide bars based on selection
    selected_sentiments = st.checkbox("Select Sentiments", value=True, key='select_sentiments')
    if selected_sentiments:
        selected_sentiments = st.multiselect("Select sentiments to display", labels, default=labels, key='selected_sentiments')
        # Iterate over the bars and hide/show based on selection
        for bar, label in zip(bars, labels):
            bar.set_visible(label in selected_sentiments)

    # Display the chart in the Streamlit app
    st.pyplot(fig)


def visuals():
    """Contains all the app visuals"""
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.subheader("Data Visuals")
    st.markdown("The following visuals provide more insight into the data that was used to train the clamate change sentiment classification models.")
    word_cloud()
    st.markdown("")
    st.markdown("")
    pie_chart()
    st.markdown("")
    st.markdown("")
    bar_chart()


def data_description():
    """Docstring"""

    st.markdown("")
    st.markdown("")
    st.markdown("""Where is this data from?  
    
The collection of this data was funded by a Canada Foundation for Innovation JELF Grant to Chris Bauch, University of Waterloo. The dataset aggregates tweets pertaining to climate change collected between Apr 27, 2015 and Feb 21, 2018. In total, 43,943 tweets were collected. Each tweet is labelled as one of 4 classes, which are described below.

**Class Description**

2 - News: Swedish politicians troll Trump administration while signing climate change law

1 - Pro: In a parliament which includes climate change deniers, austerity junkies &amp; voracious vandals of the cultura

0 - Neutral: Look who's back in town! And thank you, global warming

-1 - Anti: Hey I tweeted about the Man made global warming scam and immediately lost a couple of followers

**Features**

Sentiment:  Which class a tweet belongs in (refer to Class Description above)

Message:  Tweet body""")
    

# The main function where we will build the actual app
def main():
    """Tweet Classifier App with Streamlit"""
    # Creates a main title and subheader on your page -
    # these are static across all pages
    logo_image = "PredictiveEdge_Inc_Logo.png"
    st.sidebar.image(logo_image, width=150)
    st.sidebar.markdown("")
    st.sidebar.markdown("")
    st.sidebar.markdown("")
    st.title("Tweet Classifier")
    st.subheader("Climate change sentiment classification")

    # Creating sidebar with selection box -
    # you can create multiple pages this way
    options = ["Prediction", "Information"]
    selection = st.sidebar.selectbox("Choose Option", options)

    # Building out the "Information" page
    if selection == "Information":
        st.markdown("")
        st.markdown("")
        st.info("General Information")
        # You can read a markdown file from supporting resources folder
        st.markdown("")
        st.markdown("")

        st.subheader("Raw Twitter data and label")
        if st.checkbox('Show raw data'):  # data is hidden if box is unchecked
            st.write(raw[['sentiment', 'message']])  # will write the df to the page
        data_description()
        visuals()

    # Building out the prediction page
    if selection == "Prediction":
        st.info("Prediction using machine Learning Models")

        st.markdown("Predicts climate change sentiment based on a given text statement using a machine learning model. The models were trained using tweets that shows individuals' sentiments on climate change.  \n\n")
        st.markdown("""1. Choose a model to use for the prediction.  \n2. Then enter text in the box below and click the 'classify' button.
        You will get one of the following sentiment predictions:  """)
        st.info("""\t News: Swedish politicians troll Trump administration while signing climate change law  
        \t Pro: In a parliament which includes climate change deniers, austerity junkies &amp; voracious vandals of the cultura  
        \t Neutral: Look who's back in town! And thank you, global warming  
        \t Anti: Hey I tweeted about the Man made global warming scam and immediately lost a couple of followers""")
        
        # Model options
        model_options = ["Logistic Regression", "Neural Network Classifier", "Support Vector Machines",
                         "Stochastic Gradient Descent"]
        model_selection = st.selectbox("Choose Model", model_options)

        # Creating a text box for user input
        tweet_text = st.text_area("Enter Text", "Type Here")

        if st.button("Classify"):
            # Transforming user input with vectorizer
            vect_text = tweet_cv.transform([tweet_text]).toarray()
            # Load your .pkl file with the model of your choice + make predictions
            # Try loading in multiple models to give the user a choice

            if model_selection == "Logistic Regression":
                predictor = joblib.load(open(os.path.join("Streamlit/lr_model.pkl"),"rb"))
                
            elif model_selection == "Neural Network Classifier":
                predictor = joblib.load(open(os.path.join("Streamlit/nn_model.pkl"),"rb"))

            elif model_selection == "Support Vector Machines":
                predictor = joblib.load(open(os.path.join("Streamlit/svm_model.pkl"),"rb"))

            elif model_selection == "Stochastic Gradient Descent":
                predictor = joblib.load(open(os.path.join("Streamlit/rf_model.pkl"),"rb"))

            # predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"), "rb"))
            prediction = predictor.predict(vect_text)
            result = switch_case(prediction)

            # When the model has successfully run, it will print the prediction
            # You can use a dictionary or similar structure to make this output
            # more human interpretable.
            st.success("Text Categorized as: {} - {}".format(prediction, result))


# Required to let Streamlit instantiate our web app.
if __name__ == '__main__':
    main()
