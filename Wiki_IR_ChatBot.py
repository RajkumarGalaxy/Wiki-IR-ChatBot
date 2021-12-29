"""
Wiki-IR-ChatBot
by RAJKUMAR LAKSHMANAMOORTHY

source code at https://github.com/RajkumarGalaxy/Wiki-IR-ChatBot
more details at README.md in the repo 
refer requirements.txt in the repo to meet the code needs


This project builds an information retrieval (IR) chatbot 
that can scrape Wikipedia using BeautifulSoup in the topic 
of user's interest and collect information against user's queries 
following a heuristic backed by TF-IDF score and cosine-similarity score. 
This Wiki-IR-ChatBot is user-friendly in permitting users 
to choose any topic and presenting either crisp and short response 
or detailed response. It leverages NLTK library to do text processing 
and scikit-learn library to do modeling. 


find complete article on Kaggle
https://www.kaggle.com/rajkumarl/wiki-ir-chatbot 
""" 

# Create Environment by Importing Libraries
# To scrape Wikipedia
from bs4 import BeautifulSoup
# To access contents from URLs
import requests
# to preprocess text
import nltk
# to handle punctuations
from string import punctuation
# TF-IDF vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
# cosine similarity score
from sklearn.metrics.pairwise import cosine_similarity 
# to do array operations
import numpy as np
# to have sleep option
from time import sleep 

# Create a ChatBot Class
# A chatbot class that can perform information retrieval (IR) 
# from Wikipedia, make coversations with human users based on the retrieved data!

class ChatBot():
    
    # initialize bot
    def __init__(self):
        # flag whether to end chat
        self.end_chat = False
        # flag whether topic is found in wikipedia
        self.got_topic = False
        # flag whether to call respond()
        # in some cases, response be made already
        self.do_not_respond = True
        
        # wikipedia title
        self.title = None
        # wikipedia scraped data as paragraphs
        self.text_data = []
        # data as sentences
        self.sentences = []
        # to keep track of paragraph indices
        # corresponding to all sentences
        self.para_indices = []
        # currently retrieved sentence id
        self.current_sent_idx = None
		
		        
        # a punctuation dictionary
        self.punctuation_dict = str.maketrans({p:None for p in punctuation})
        # wordnet lemmatizer for preprocessing text
        self.lemmatizer = nltk.stem.WordNetLemmatizer()
        # collection of stopwords
        self.stopwords = nltk.corpus.stopwords.words('english')
        # initialize chatting
        self.greeting()

    # greeting method - to be called internally
    # chatbot initializing chat on screen with greetings
    def greeting(self):
        print("Initializing ChatBot ...")
        # some time to get user ready
        sleep(2)
        # chat ending tags
        print('Type "bye" or "quit" or "exit" to end chat')
        sleep(2)
        # chatbot descriptions
        print('\nEnter your topic of interest when prompted. \
        \nChaBot will access Wikipedia, prepare itself to \
        \nrespond to your queries on that topic. \n')
        sleep(3)
        print('ChatBot will respond with short info. \
        \nIf you input "more", it will give you detailed info \
        \nYou can also jump to next query')
        # give time to read what has been printed
        sleep(3)
        print('-'*50)
        # Greet and introduce
        greet = "Hello, Great day! Please give me a topic of your interest. "
        print("ChatBot >>  " + greet)
		
	# chat method - should be called by user
    # chat method controls inputs, responses, data scraping, preprocessing, modeling.
    # once an instance of ChatBot class is initialized, chat method should be called
    # to do the entire chatting on one go!
    def chat(self):
        # continue chat
        while not self.end_chat:
            # receive input
            self.receive_input()
            # finish chat if opted by user
            if self.end_chat:
                print('ChatBot >>  See you soon! Bye!')
                sleep(2)
                print('\nQuitting ChatBot ...')
            # if data scraping successful
            elif self.got_topic:
                # in case not already responded
                if not self.do_not_respond:
                    self.respond()
                # clear flag so that bot can respond next time
                self.do_not_respond = False
				
	# receive_input method - to be called internally
    # recieves input from user and makes preliminary decisions
    def receive_input(self):
        # receive input from user
        text = input("User    >> ")
        # end conversation if user wishes so
        if text.lower().strip() in ['bye', 'quit', 'exit']:
            # turn flag on 
            self.end_chat=True
        # if user needs more information 
        elif text.lower().strip() == 'more':
            # respond here itself
            self.do_not_respond = True
            # if at least one query has been received 
            if self.current_sent_idx != None:
                response = self.text_data[self.para_indices[self.current_sent_idx]]
            # prompt user to start querying
            else:
                response = "Please input your query first!"
            print("ChatBot >> " + response)
        # if topic is not chosen
        elif not self.got_topic:
            self.scrape_wiki(text)
        else:
            # add user input to sentences, so that we can vectorize in whole
            self.sentences.append(text)
	
	# respond method - to be called internally
    def respond(self):
        # tf-idf-modeling
        vectorizer = TfidfVectorizer(tokenizer=self.preprocess)
        # fit data and obtain tf-idf vector
        tfidf = vectorizer.fit_transform(self.sentences)
        # calculate cosine similarity scores
        scores = cosine_similarity(tfidf[-1],tfidf) 
        # identify the most closest sentence
        self.current_sent_idx = scores.argsort()[0][-2]
        # find the corresponding score value
        scores = scores.flatten()
        scores.sort()
        value = scores[-2]
        # if there is matching sentence
        if value != 0:
            print("ChatBot >> " + self.sentences[self.current_sent_idx]) 
        # if no sentence is matching the query
        else:
            print("ChatBot >>  I am not sure. Sorry!" )
        # remove the user query from sentences
        del self.sentences[-1]
		
	        
    # scrape_wiki method - to be called internally.
    # called when user inputs topic of interest.
    # employs requests to access Wikipedia via URL.
    # employs BeautifulSoup to scrape paragraph tagged data
    # and h1 tagged article heading.
    # employs NLTK to tokenize data
    def scrape_wiki(self,topic):
        # process topic as required by Wikipedia URL system
        topic = topic.lower().strip().capitalize().split(' ')
        topic = '_'.join(topic)
        try:
            # creata an url
            link = 'https://en.wikipedia.org/wiki/'+ topic
            # access contents via url
            data = requests.get(link).content
            # parse data as soup object
            soup = BeautifulSoup(data, 'html.parser')
            # extract all paragraph data
            # scrape strings with html tag 'p'
            para = soup.findAll('p')
            # iterate over all paragraphs
            for p in para:
                # a bucket to collect processed data
                a = []
                # iterate over para contents and tags
                for i in p.contents:
                    # exclude references, superscripts, formattings
                    if i.name != 'sup' and i.string != None:
                        stripped = ' '.join(i.string.strip().split())
                        # collect data pieces
                        a.append(stripped)
                # with collected string pieces formulate a single string
                # each string is a paragraph
                self.text_data.append(' '.join(a))
				
				
            # obtain sentences from paragraphs
            for i,para in enumerate(self.text_data):
                sentences = nltk.sent_tokenize(para)
                self.sentences.extend(sentences)
                # for each sentence, its para index must be known
                # it will be useful in case user prompts "more" info
                index = [i]*len(sentences)
                self.para_indices.extend(index)
            
            # extract h1 heading tag from soup object
            self.title = soup.find('h1').string
            # turn respective flag on
            self.got_topic = True
            # announce user that chatbot is ready now
            print('ChatBot >>  Topic is "Wikipedia: {}". Let\'s chat!'.format(self.title)) 
        # in case of unavailable topics
        except Exception as e:
            print('ChatBot >>  Error: {}. \
            Please input some other topic!'.format(e))
        
		
    # preprocess method - to be called internally by Tf-Idf vectorizer
    # text preprocessing, stopword removal, lemmatization, word tokenization
    def preprocess(self, text):
        # remove punctuations
        text = text.lower().strip().translate(self.punctuation_dict) 
        # tokenize into words
        words = nltk.word_tokenize(text)
        # remove stopwords
        words = [w for w in words if w not in self.stopwords]
        # lemmatize 
        return [self.lemmatizer.lemmatize(w) for w in words]
		


# Happy Chatting!
# Initialize ChatBot and start chatting.
		
if __name__ == '__main__':
	# instantiate an object
	wiki = ChatBot()
	# call chat method
	wiki.chat()
# end of script