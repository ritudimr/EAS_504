import datetime
import os

import tkinter
from tkinter import ttk

import customtkinter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from helpers.subreddits import SUBREDDITS
from .modeltrainer import ModelTrainer
import numpy as np


def pretty_number(number):
    # Convert number to in B, M, K format
    if number >= 1000000000:
        return '{:.2f} B'.format(number / 1000000000)
    elif number >= 1000000:
        return '{:.2f} M'.format(number / 1000000)
    elif number >= 1000:
        return '{:.2f} K'.format(number / 1000)
    else:
        return number

# Author Scoring Function
def author_scores(df):
    df_author = df[['author', 'score', 'subreddit', 'num_comments', 'upvote_ratio']]
    df_author['final_score'] = df_author['score'] * df_author['upvote_ratio'] + df_author['num_comments']
    df_author = df_author.groupby(['author', 'subreddit']).sum()
    df_author = df_author.reset_index()
    return df_author

# Plot Viewer Window
class PlotViewer(customtkinter.CTk):
    def __init__(self, posts):
        super().__init__()
        self.title('Reddit Data Analysis - Plot Viewer')
        posx = int(self.winfo_screenwidth() / 2 - 600)
        posy = int(self.winfo_screenheight() / 2 - 400)
        self.geometry(f'1200x800+{posx}+{posy}')
        self.posts = posts
        self.create_tabs()

    def create_tabs(self):
        self.tabview = customtkinter.CTkTabview(self)
        self.tabview.add("Posts")
        self.tabview.add("Subscribers")
        self.tabview.add("Author Activity")
        self.tabview.add("Multi-Subreddit Analysis")
        self.tabview.add("Posts per Day")
        self.tabview.add("Top 10 Authors")
        self.tabview.add("Best Time Analysis")
        self.tabview.add("Scores Boxplot")
        self.tabview.add("Scores vs Comments")
        self.tabview.add("View Data / Predictions")

        fig = Figure(figsize=(12, 8), dpi=72)
        self.posts_plot = fig.add_subplot(111)
        self.posts_plot.set_title('Number of posts per subreddit')
        self.posts_plot.set_xlabel('Subreddit')
        self.posts_plot.set_xticklabels(np.arange(len(SUBREDDITS)), rotation=45)
        self.posts_plot.set_ylabel('Number of posts')
        sns.countplot(x='subreddit', data=self.posts, ax=self.posts_plot)
        for p in self.posts_plot.patches:
            self.posts_plot.annotate('{:1.0f} posts'.format(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()), 
                       ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
        self.posts_plot.figure.tight_layout()
        self.posts_plot = FigureCanvasTkAgg(fig, self.tabview.tab("Posts"))
        self.posts_plot.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)

        fig = Figure(figsize=(12, 8), dpi=72)
        self.subscribers_plot = fig.add_subplot(111)
        self.subscribers_plot.set_title('Number of subscribers per subreddit')
        self.subscribers_plot.set_xlabel('Subreddit')
        self.subscribers_plot.set_xticklabels(np.arange(len(SUBREDDITS)), rotation=45)
        self.subscribers_plot.set_ylabel('Number of subscribers')
        sns.barplot(x='subreddit', y='subreddit_subscribers', data=self.posts, ax=self.subscribers_plot)
        for p in self.subscribers_plot.patches:
            self.subscribers_plot.annotate('{}'.format(pretty_number(p.get_height())), (p.get_x() + p.get_width() / 2., p.get_height()),
                          ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
        self.subscribers_plot.figure.tight_layout()
        self.subscribers_plot = FigureCanvasTkAgg(fig, self.tabview.tab("Subscribers"))
        self.subscribers_plot.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)

        fig = Figure(figsize=(12, 8), dpi=72)
        self.author_activity_plot = fig.add_subplot(111)
        self.author_activity_plot.set_title('Authors Posting in multiple Subreddits')
        n_subreddits = self.posts.groupby('author')['subreddit'].nunique()
        sns.countplot(x=n_subreddits, palette=sns.color_palette("husl"), ax=self.author_activity_plot)
        for p in self.author_activity_plot.patches:
            self.author_activity_plot.annotate('{:1.0f} authors'.format(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()),
                          ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
        self.author_activity_plot.set_xlabel('Number of Subreddits')
        self.author_activity_plot.set_ylabel('Number of Authors')
        self.author_activity_plot.figure.tight_layout()
        self.author_activity_plot = FigureCanvasTkAgg(fig, self.tabview.tab("Author Activity"))
        self.author_activity_plot.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)

        fig = Figure(figsize=(12, 8), dpi=72)
        n_upvotes = self.posts.groupby('author')['ups'].sum()
        self.multi_subreddit_plot = fig.add_subplot(111)
        self.multi_subreddit_plot.set_title('Does posting in multiple subreddits drives more upvotes?')
        sns.barplot(x=n_subreddits, y=n_upvotes, palette=sns.color_palette("pastel"), ax=self.multi_subreddit_plot)
        for p in self.multi_subreddit_plot.patches:
            self.multi_subreddit_plot.annotate('{:1.0f} upvotes'.format(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()),
                          ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
        self.multi_subreddit_plot.set_xlabel('Number of Subreddits')
        self.multi_subreddit_plot.set_ylabel('Number of Upvotes')
        self.multi_subreddit_plot.set_xticks(list(range(0, len(n_subreddits.unique()))), list(map(lambda x: '{} Subreddits'.format(x) if x > 1 else '{} Subreddits'.format(x), list(range(1, len(n_subreddits.unique()) + 1)))))
        self.multi_subreddit_plot.figure.tight_layout()
        self.multi_subreddit_plot = FigureCanvasTkAgg(fig, self.tabview.tab("Multi-Subreddit Analysis"))
        self.multi_subreddit_plot.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)
        

        ppd_df = self.posts.groupby(['subreddit', 'created_utc']).size().reset_index(name='counts')
        ppd_df['created_utc'] = pd.to_datetime(ppd_df['created_utc']).dt.date
        ppd_df = ppd_df.groupby(['subreddit', 'created_utc']).sum().reset_index()
        ppd_df = ppd_df.pivot(index='created_utc', columns='subreddit', values='counts')
        ppd_df = ppd_df.fillna(0)
        last_6M = datetime.date.today() - datetime.timedelta(days=180)
        ppd_df = ppd_df.loc[ppd_df.index >= last_6M]
        palette = sns.color_palette("dark6", len(SUBREDDITS))


        fig, axes = plt.subplots(5, 3, figsize=(20, 20), dpi=24)
        fig.suptitle('Number of posts per day per subreddit (Last 6 Months)\n', fontsize=16)
        fig.subplots_adjust(hspace=0.5, wspace=0.5)
        for i, subreddit in enumerate(SUBREDDITS):
            ax = axes[i // 3, i % 3]
            ax.set_title(subreddit)
            ax.set_xlabel('Date')
            ax.set_ylabel('Number of Posts')
            ax.set_xticklabels(ppd_df.index, rotation=0)
            sns.lineplot(data=ppd_df[subreddit], ax=ax, color=palette[i])
        self.ppd_plot = FigureCanvasTkAgg(fig, self.tabview.tab("Posts per Day"))
        self.ppd_plot.figure.tight_layout()
        self.ppd_plot.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)

        top_10_authors_per_subreddit = author_scores(self.posts).groupby('subreddit').apply(lambda x: x.nlargest(10, 'final_score'))
        top_10_authors_per_subreddit = top_10_authors_per_subreddit.reset_index(drop=True)
        fig, axes = plt.subplots(5, 3, figsize=(20, 20), dpi=24)
        fig.suptitle('Top 10 Authors per Subreddit\n', fontsize=16)
        fig.subplots_adjust(hspace=0.5, wspace=0.5)
        for i, subreddit in enumerate(SUBREDDITS):
            ax = axes[i // 3, i % 3]
            ax.set_title(subreddit)
            ax.set_xticklabels(axes[i//3, i%3].get_xticklabels(), rotation=30, horizontalalignment='right')
            sns.barplot(x='author', y='final_score', data=top_10_authors_per_subreddit[top_10_authors_per_subreddit['subreddit'] == subreddit], ax=ax, palette=sns.color_palette("pastel", 10)) 
            ax.set_ylabel('Final Score')
            ax.set_xlabel('')
            for p in axes[i//3, i%3].patches:
                axes[i//3, i%3].annotate('{:1.0f}'.format(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()),
                              ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
        self.top_10_authors_plot = FigureCanvasTkAgg(fig, self.tabview.tab("Top 10 Authors"))
        self.top_10_authors_plot.figure.tight_layout()
        self.top_10_authors_plot.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)

        # Finding the best time to post on each subreddit
        best_time_df = self.posts[['subreddit', 'created_utc', 'score', 'num_comments']]
        best_time_df['final_score'] = best_time_df['score'] + best_time_df['num_comments']
        best_time_df.drop(['score', 'num_comments'], axis=1, inplace=True)

        # Convert the created_utc column to datetime
        best_time_df['created_utc'] = pd.to_datetime(best_time_df['created_utc'])
        best_time_df['day'] = best_time_df['created_utc'].dt.day_name()
        best_time_df['hour'] = best_time_df['created_utc'].dt.hour
        best_time_df.drop('created_utc', axis=1, inplace=True)

        # Find total engagement per hour
        best_time_df = best_time_df.groupby(['subreddit', 'day', 'hour']).sum()
        best_time_df = best_time_df.reset_index()
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        best_time_df['day'] = pd.Categorical(best_time_df['day'], categories=days, ordered=True)

        # Plotting the best time to post on each subreddit
        fig, axes = plt.subplots(5, 3, figsize=(20, 20), dpi=24)
        fig.suptitle('Best Time to Post on Each Subreddit\n', fontsize=20)
        for i, subreddit in enumerate(best_time_df['subreddit'].unique()):
            sns.lineplot(x='hour', y='final_score', hue='day', data=best_time_df[best_time_df['subreddit'] == subreddit], ax=axes[i//3, i%3], palette=sns.color_palette("husl", 7))
            axes[i//3, i%3].set_title(subreddit)
            axes[i//3, i%3].set_xticks(range(0, 24))
            axes[i//3, i%3].set_xticklabels(list(map(lambda x: (f'0{x}:00' if x < 10 else f'{x}:00'), list(range(0, 24)))), rotation=45, horizontalalignment='right')
            axes[i//3, i%3].set_xlabel('Time of Day')
            axes[i//3, i%3].set_ylabel('Total Engagement')
        self.best_time_plot = FigureCanvasTkAgg(fig, self.tabview.tab("Best Time Analysis"))
        self.best_time_plot.figure.tight_layout()
        self.best_time_plot.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)

        fig = Figure(figsize=(12, 8), dpi=72)
        self.scores_boxplot = fig.add_subplot(111)
        sns.boxplot(x='subreddit', y='score', data=self.posts, ax=self.scores_boxplot)
        self.scores_boxplot.set_title('Boxplot of Scores in Each Subreddit')
        self.scores_boxplot.set_xlabel('Subreddit')
        self.scores_boxplot.set_ylabel('Score')
        self.scores_boxplot = FigureCanvasTkAgg(fig, self.tabview.tab("Scores Boxplot"))
        self.scores_boxplot.figure.tight_layout()
        self.scores_boxplot.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)

        # Scatterplot of the scores and number of comments in each subreddit
        fig, axes = plt.subplots(5, 3, figsize=(20, 20), dpi=24)
        fig.suptitle('Scatterplot of Scores and Number of Comments in Each Subreddit\n', fontsize=20)
        palette=sns.color_palette("deep", 15)
        for i, subreddit in enumerate(self.posts['subreddit'].unique()):
            sns.scatterplot(x='score', y='num_comments', data=self.posts[self.posts['subreddit'] == subreddit], ax=axes[i//3, i%3], color=palette[i])
            axes[i//3, i%3].set_title(subreddit)
            axes[i//3, i%3].set_xlabel('Score')
            axes[i//3, i%3].set_ylabel('Number of Comments')
        self.scores_comments_plot = FigureCanvasTkAgg(fig, self.tabview.tab("Scores vs Comments"))
        self.scores_comments_plot.figure.tight_layout()
        self.scores_comments_plot.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)


        # View Data / Predictions tab
        # show the posts dataframe in a table
        self.posts_table = ttk.Treeview(self.tabview.tab("View Data / Predictions"))
        self.posts_table.pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)
        self.posts_table['columns'] = list(self.posts.columns)
        for column in self.posts_table['columns']:
            self.posts_table.column(column, anchor='w')
            self.posts_table.heading(column, text=column, anchor='w')

        # hide the first column (index)
        self.posts_table.column('#0', width=0, stretch=tkinter.NO)

        for i, row in self.posts.iterrows():
            if i < 100:
                self.posts_table.insert('', 'end', values=list(row))

        if not os.path.exists('models') or len(os.listdir('models')) == 0:
            try:
                os.mkdir('models')
            except:
                pass
            self.models_label = customtkinter.CTkLabel(self.tabview.tab("View Data / Predictions"), text="No models found. Please train the models first.", pady= 10)
            self.models_label.pack()
            self.models_button = customtkinter.CTkButton(self.tabview.tab("View Data / Predictions"), text="Train Models", command=self.train_models)
            self.models_button.pack()
        else:
            self.models_label = customtkinter.CTkLabel(self.tabview.tab("View Data / Predictions"), text="Models found. Predict by entering data on the next screen.", pady= 10)
            self.models_label.pack()
            self.models_button = customtkinter.CTkButton(self.tabview.tab("View Data / Predictions"), text="Predict")
            self.models_button.pack()

        self.tabview.pack(expand=True, fill='both')

    def train_models(self):
        # open model training child window
        mt = ModelTrainer(self, self.posts)
        mt.grab_set()
        mt.focus_set()
        self.wait_window(mt)

        self.models_label.destroy()
        self.models_button.destroy()
        self.models_label = customtkinter.CTkLabel(self.tabview.tab("View Data / Predictions"), text="Models found. Predict by entering data on the next screen.", pady= 10)
        self.models_label.pack()
        self.models_button = customtkinter.CTkButton(self.tabview.tab("View Data / Predictions"), text="Predict")
        self.models_button.pack()


