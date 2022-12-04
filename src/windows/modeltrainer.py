import html
import os
import pickle
import re
import warnings

import customtkinter
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import ssl
import json
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

warnings.filterwarnings('ignore')
nltk.download('stopwords')
nltk.download('wordnet')
from string import punctuation
import pandas as pd

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor


def preprocess(message):
    stemmer = SnowballStemmer('english')
    stuff_to_be_removed = list(stopwords.words('english'))+list(punctuation)

    # Convert message to lower case 
    message = message.lower()
    
    # Remove all the links from the messages 
    message = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                    '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', message)
    # Remove all the mentions
    message =re.sub("(@[A-Za-z0-9_]+)","", message)

    # Remove all the emojis
    message = re.sub(re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
        "]+", flags=re.UNICODE), '', message)

    # Remove HTML entities
    message = html.unescape(message)

    # strip blank spaces
    message = message.strip()

    # Remove all the punctuations
    message = message.translate(str.maketrans('', '', punctuation))

    # Remove stopwords and perform stemming 
    message = ' '.join([stemmer.stem(word) for word in message.split() if word not in stuff_to_be_removed])
    
    # Return the message 
    return message 

class ModelTrainer(customtkinter.CTkToplevel):
    model_dir = 'models/'

    def __init__(self, parent, posts):
        super().__init__(parent)
        self.parent = parent
        self.posts = posts # posts is already a dataframe
        self.features = ['title', 'selftext', 'subreddit', 'distinguished', 'hour', 'day']
        self.targets = ['ups', 'num_comments']
        for col in self.posts.columns:
            if col not in self.features + self.targets:
                self.posts.drop(col, axis=1, inplace=True)
        
        self.categorical_features = ['subreddit', 'distinguished', 'hour', 'day']
        self.posts['text'] = self.posts['title'] + ' ' + self.posts['selftext']
        self.posts['text'] = self.posts['text'].astype(str)
        self.posts['text'] = self.posts['text'].apply(lambda x: preprocess(x))
        self.posts.drop(['title', 'selftext'], axis=1, inplace=True)

        # convert categorical features
        for col in self.categorical_features:
            self.posts[col] = self.posts[col].astype('category')

        self.posts = pd.get_dummies(self.posts, columns=self.categorical_features)

        self.text_features = ['text']
        self.title('Reddit Data Analysis - Building Models')
        posx = int(self.winfo_screenwidth()/2 - 300)
        posy = int(self.winfo_screenheight()/2 - 150)
        self.geometry('600x300+{}+{}'.format(posx, posy))
        self.resizable(False, False)
        self.protocol('WM_DELETE_WINDOW', self.disable_event)

        self.updates = customtkinter.CTkTextbox(self, height=300, width=600, state = 'disabled')
        self.updates.pack(fill='both', expand=True)

        # Create a hash table to store the model objects
        self.model_hashmap = {
            "DummyRegressor": DummyRegressor(),
            "LinearRegression": LinearRegression(),
            "RidgeCV": RidgeCV(cv=10),
            "KNeighborsRegressor": KNeighborsRegressor(),
            "DecisionTreeRegressor": DecisionTreeRegressor(min_samples_split=45, min_samples_leaf=45, random_state = 10),
            "RandomForestRegressor": RandomForestRegressor(n_jobs=-1, n_estimators=70, min_samples_leaf=10, random_state = 10),
            "GradientBoostingRegressor": GradientBoostingRegressor(n_estimators=70, max_depth=5)
        }
        self.ups_dict = {}
        self.num_comments_dict = {}

        self.start()

    def disable_event(self):
        pass

    def edit_textbox(self, text, line, type='wait'):
        emoji = '🕐' if type == 'wait' else '✅'
        line_next = line + 1
        line = str(line) + '.0'
        line_next = str(line_next) + '.0'
        self.updates.configure(state='normal')
        if type == 'wait':
            self.updates.insert(line, emoji + ' ' + text + '...' + '\n\n')
        else:
            self.updates.delete(line, line_next)
            self.updates.insert(line, emoji + ' ' + text + '\n\n')
        self.updates.configure(state='disabled')

        # scroll to line
        self.updates.see(line)

        # update the window
        self.update()


    def start(self):
        self.tfidf = TfidfVectorizer()
        self.X = self.tfidf.fit_transform(self.posts['text'])

        self.edit_textbox('Preparing Data (Upvotes)', 1, 'wait')
        # dataframes for ups
        self.ups_df = self.posts.drop(['num_comments'], axis=1)

        # split data into train and test sets for ups
        self.X_train_ups, self.X_test_ups, self.y_train_ups, self.y_test_ups = train_test_split(self.X, self.ups_df['ups'], test_size=0.2, random_state=10)

        self.edit_textbox('Preparing Data (Upvotes)', 1, 'done')

        self.edit_textbox('Preparing Data (Number of Comments)', 3, 'wait')
        # dataframes for num_comments
        self.num_comments_df = self.posts.drop(['ups'], axis=1)

        # split data into train and test sets for num_comments
        self.X_train_num_comments, self.X_test_num_comments, self.y_train_num_comments, self.y_test_num_comments = train_test_split(self.X, self.num_comments_df['num_comments'], test_size=0.2, random_state=10)
        
        self.edit_textbox('Preparing Data (Number of Comments)', 2, 'done')

        # train models
        self.train_models()

    # Create a function to save the models
    def save_model(self, model, model_name):
        """
        Saves the model to the models/ directory
        """
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)
        with open(self.model_dir + model_name + '.pkl', 'wb') as f:
            pickle.dump(model, f)


    def train_models(self):
        line_count = 3
        for model_name, model in self.model_hashmap.items():
            self.edit_textbox('Training {} for Upvotes'.format(model_name), line_count, 'wait')
            model.fit(self.X_train_ups, self.y_train_ups)
            self.save_model(model, model_name + '_ups')

            ups_y_pred = model.predict(self.X_test_ups)
            ups_mse = mean_squared_error(self.y_test_ups, ups_y_pred)
            ups_mae = mean_absolute_error(self.y_test_ups, ups_y_pred)
            ups_r2 = r2_score(self.y_test_ups, ups_y_pred)
            self.ups_dict[model_name] = {
                'mse': ups_mse,
                'mae': ups_mae,
                'r2': ups_r2,
                'rmse': np.sqrt(ups_mse),
                'pred': list(ups_y_pred),
                'actual': list(self.y_test_ups)
            }

            self.edit_textbox('Training {} for Upvotes'.format(model_name), line_count, 'done')
            line_count += 1

            self.edit_textbox('Training {} for Number of Comments'.format(model_name), line_count, 'wait')
            model.fit(self.X_train_num_comments, self.y_train_num_comments)
            self.save_model(model, model_name + '_num_comments')

            num_comments_y_pred = model.predict(self.X_test_num_comments)
            num_comments_mse = mean_squared_error(self.y_test_num_comments, num_comments_y_pred)
            num_comments_mae = mean_absolute_error(self.y_test_num_comments, num_comments_y_pred)
            num_comments_r2 = r2_score(self.y_test_num_comments, num_comments_y_pred)
            self.num_comments_dict[model_name] = {
                'mse': num_comments_mse,
                'mae': num_comments_mae,
                'r2': num_comments_r2,
                'rmse': np.sqrt(num_comments_mse),
                'pred': list(num_comments_y_pred),
                'actual': list(self.y_test_num_comments)
            }

            self.edit_textbox('Training {} for Number of Comments'.format(model_name), line_count, 'done')
            line_count += 1

        # dump the vectorizer
        with open(self.model_dir + 'vectorizer.pkl', 'wb') as f:
            pickle.dump(self.tfidf, f)

        # save the metrics
        with open(self.model_dir + 'ups_metrics.json', 'w') as f:
            json.dump(self.ups_dict, f)

        with open(self.model_dir + 'num_comments_metrics.json', 'w') as f:
            json.dump(self.num_comments_dict, f)

        self.edit_textbox('Training Complete. Models saved to models/ directory. You may now close this window.', line_count, 'done')

        # allow user to close window
        self.protocol("WM_DELETE_WINDOW", self.enable_close)

    def enable_close(self):
        self.destroy()
