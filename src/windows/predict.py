import random
import math
import customtkinter
import tkinter
import pandas as pd
import os 
import pickle
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from .modeltrainer import preprocess
import tkinter.messagebox

class Predict(customtkinter.CTkToplevel):
    model_dir = "models/"
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.title("Predictions")
        self.grab_set()
        self.focus_set()
        posx = int(self.winfo_screenwidth()/2 - 150)
        posy = int(self.winfo_screenheight()/2 - 350)
        self.geometry("300x650+{}+{}".format(posx, posy))
        self.resizable(False, False)

        # title
        self.title_label = customtkinter.CTkLabel(self, text="Title", anchor='w')
        self.title_label.pack(pady=5, padx=30, fill='x', side=tkinter.TOP, anchor='w')
        self.title_entry = customtkinter.CTkEntry(self, width=240)
        self.title_entry.pack(pady=10, padx=10)
        self.title_entry.insert(0, 'A good openCV tutorial?')

        # selftext
        self.selftext_label = customtkinter.CTkLabel(self, text="Selftext", anchor='w')
        self.selftext_label.pack(pady=5, padx=30, fill='x', side=tkinter.TOP, anchor='w')
        self.selftext_entry = customtkinter.CTkTextbox(self, width=240, height=100)
        self.selftext_entry.pack(pady=10, padx=10)
        self.selftext_entry.insert("0.0", "So I'm learning openCV in python, and now I want as a project to develop some score calculator for a scrabble game. I watched this tutorial from codecamp, and i read about of functionalities of opencv module (such as medianblur, gaussianblur, addweighted, Canny, threshold, and so on), but i still can't grasp it together. Like, i know how to blur an image, to reduce noise let's say, but i don't know when to do that, and especially, why and how much, so I'm searching for a good openCV tutorial that explains these situations. \n As an example, yesterday I did a project where i would've get a sudoku box from an image(by getting the top left, top right, bottom left, bottom right corners of the sudoku box). However, when I tried the same code for the project with the scrabble board, it's a total mess.")

        # subreddit
        self.subreddit_label = customtkinter.CTkLabel(self, text="Subreddit", anchor='w')
        self.subreddit_label.pack(pady=5, padx=30, fill='x', side=tkinter.TOP, anchor='w')

        self.subreddit_entry = customtkinter.CTkOptionMenu(self, values=self.parent.posts['subreddit'].unique(), width=240)
        self.subreddit_entry.pack(pady=10, padx=10)
        self.subreddit_entry.set(self.parent.posts['subreddit'].unique()[0])

        # day
        self.day_label = customtkinter.CTkLabel(self, text="Day", anchor='w')
        self.day_label.pack(pady=5, padx=30, fill='x', side=tkinter.TOP, anchor='w')
        self.day_entry = customtkinter.CTkOptionMenu(self, values=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], width=240)
        self.day_entry.pack(pady=10, padx=10)
        self.day_entry.set('Monday')

        # hour
        self.hour_label = customtkinter.CTkLabel(self, text="Hour", anchor='w')
        self.hour_label.pack(pady=5, padx=30, fill='x', side=tkinter.TOP, anchor='w')

        hours = ['{}:00'.format(i) for i in range(24)]

        self.hour_entry = customtkinter.CTkOptionMenu(self, values=hours, width=240)
        self.hour_entry.pack(pady=10, padx=10)
        self.hour_entry.set('10:00')
        
        # distinguished
        self.distinguished_entry = customtkinter.CTkCheckBox(self, text="Distinguished")
        self.distinguished_entry.pack(pady=10, padx=10)

        # button for predicting
        self.predict_button = customtkinter.CTkButton(self, text="Predict", command=self.predict)
        self.predict_button.pack(pady=10, padx=10)

        # button for closing the window
        self.close_button = customtkinter.CTkButton(self, text="Close", command=self.destroy)
        self.close_button.pack(pady=10, padx=10)

    def predict(self):
        title = self.title_entry.get()
        selftext = self.selftext_entry.get("1.0", tkinter.END)
        subreddit = self.subreddit_entry.get()
        day = self.day_entry.get()
        hour = self.hour_entry.get()
        hour = int(hour.split(':')[0])
        distinguished = self.distinguished_entry.get()

        if not title or not selftext or not subreddit or not day or not hour:
            tkinter.messagebox.showerror('Error', 'Please fill all the fields')
            return

        # load the model
        ups_model = pickle.load(open(os.path.join(self.model_dir, "DummyRegressor_ups.pkl"), 'rb'))
        num_comments_model = pickle.load(open(os.path.join(self.model_dir, "DummyRegressor_num_comments.pkl"), 'rb'))

        text = title + " " + selftext
        text = preprocess(text)

        post = pd.DataFrame({
            'text': [text],
            'subreddit': [subreddit],
            'day': [day],
            'hour': [hour],
            'distinguished': [distinguished]
        })

        self.tfidf_vectorizer = TfidfVectorizer()
        self.label_binarizer = LabelBinarizer()

        post_cat = [self.label_binarizer.fit_transform(post[col]) for col in ['subreddit', 'day', 'hour', 'distinguished']]
        post_text = self.tfidf_vectorizer.fit_transform(post['text'])
        postX = hstack([post_text] + post_cat).tocsr()

        ups = int(ups_model.predict(postX)[0])
        num_comments = int(num_comments_model.predict(postX)[0])

        # random bias from the prediction
        ups = int(ups // math.log(ups + 1))
        num_comments = int(num_comments // math.log(num_comments + 1))

        tkinter.messagebox.showinfo('Predictions', 'Predicted ups: {}\nPredicted num_comments: {}'.format(ups, num_comments))
