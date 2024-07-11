import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
from string import punctuation
from googleapiclient.discovery import build
import re
import nltk

# Ensure the necessary NLTK data is downloaded
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Initialize NLTK components
stop_words = set(stopwords.words('english'))
porter_stemmer = PorterStemmer()
lzr = WordNetLemmatizer()

# Load vectorizer and model
cv = pickle.load(open('vectorizer.pkl', 'rb'))
loaded_model = pickle.load(open('trained_model.sav', 'rb'))

def text_processing(text):   
    # convert text into lowercase
    text = text.lower()

    # remove new line characters in text
    text = re.sub(r'\n',' ', text)
    
    # remove punctuations from text
    text = re.sub('[%s]' % re.escape(punctuation), "", text)
    
    # remove references and hashtags from text
    text = re.sub("^a-zA-Z0-9$,.", "", text)
    
    # remove multiple spaces from text
    text = re.sub(r'\s+', ' ', text, flags=re.I)
    
    # remove special characters from text
    text = re.sub(r'\W', ' ', text)

    text = ' '.join([word for word in word_tokenize(text) if word not in stop_words])
    
    # stemming using porter stemmer from nltk package - msh a7sn 7aga - momken: lancaster, snowball
    text=' '.join([porter_stemmer.stem(word) for word in word_tokenize(text)])
    
    # lemmatizer using WordNetLemmatizer from nltk package
    text=' '.join([lzr.lemmatize(word) for word in word_tokenize(text)])

    return text



# create a function for prediction
def sentiment_analysis(s):
    user_input=text_processing(s)
    # Split the input into a list of comments
    comments = user_input.split(',')
    # Create a dictionary with the 'comment' column
    data = {'Comment': comments}
    # Create a DataFrame
    df = pd.DataFrame(data)
    X = cv.transform(df["Comment"]).toarray()
    Y= loaded_model.predict(X)
    if(Y==0):
        return "Negative"
    elif(Y==1):
        return "Neutral"
    else:
        return "Positive"


def main():
    #code for prediction
    result=''
    #title
    st.title("Sentimental Analysis of sample comments ")    
    #gettiung the input
    data=st.text_input("Enter a comment")
    #creataing a button for prediction
    if st.button("Predict"):
        result=sentiment_analysis(data)
    if(result =='Positive'):
        st.success(result)
    elif(result=='Negative'):
        st.error(result)
    elif(result=='Neutral'):
        st.warning(result)    


def get_youtube_comments(api_key, video_id, max_comments):
    youtube = build('youtube', 'v3', developerKey=api_key)

    # Retrieve video comments
    comments = []
    nextPageToken = None
    total_comments_retrieved = 0

    while True and total_comments_retrieved < max_comments:
        request = youtube.commentThreads().list(
            part='snippet',
            videoId=video_id,
            textFormat='plainText',
            pageToken=nextPageToken
        )
        response = request.execute()

        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            comments.append(comment)
            total_comments_retrieved += 1

            if total_comments_retrieved >= max_comments:
                break

        nextPageToken = response.get('nextPageToken')

        if not nextPageToken or total_comments_retrieved >= max_comments:
            break

    return comments

def extract_video_id(url):
        # Regular expression to extract the video ID from various YouTube URL formats
        video_id_match = re.search(r"(?:v=|v\/|vi\/|videos\/|embed\/|youtu.be\/|watch\?v=|\?v=|&v=|\?id=)([a-zA-Z0-9_-]+)", url)

        if video_id_match:
            return video_id_match.group(1)
        else:
            return None

def ytlink():
    # Prompt the user for the YouTube API key
    API_KEY = 'AIzaSyBUqOdo68VVZla4bOMF-cLqH6JSm4WvxBQ'
    #title
    st.title("Sentimental Analysis of Youtube Video ")  
    # Prompt the user for the YouTube video URL
    youtube_url = st.text_input("Enter the YouTube video URL ")
    # Extract the video ID from the URL
    if(st.button('Analyse')):
        video_id = extract_video_id(youtube_url)
        if video_id:
            MAX_COMMENTS = 2000  # Maximum comments to retrieve
            # Fetch YouTube comments for the specified video
            comments = get_youtube_comments(API_KEY, video_id, MAX_COMMENTS)
            # Create a DataFrame from the comments
            df = pd.DataFrame(comments, columns=['Comment'])
            # Display the DataFrame
            # print(df)
        else:
            print("Invalid YouTube video URL. Please provide a valid URL.")
        
        df.rename(columns={'text': 'Comment'}, inplace=True)
        df["Comment"] = df["Comment"].apply(lambda x : text_processing(x))
        X = cv.transform(df["Comment"]).toarray()
        Y= loaded_model.predict(X) 
        st.write("Negative :" + str(np.sum(Y == 0)))   
        st.write("Neutral :" + str(np.sum(Y==1)))
        st.write("Positive :" + str(np.sum(Y==2)))
        # Data

        data = {
            "categories": ["NEGATIVE","NEUTRAL","POSITIVE"],
            "values": [np.sum(Y==0),np.sum(Y==1),np.sum(Y==2)]
        }

        df = pd.DataFrame(data)

        fig, ax = plt.subplots()
        ax.imshow(df, cmap='viridis')
        st.pyplot(fig)
               

if __name__ =='__main__':
    main()
    ytlink()        
