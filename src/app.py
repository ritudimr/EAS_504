import tkinter
import tkinter.messagebox

import pandas as pd

from helpers.database_handler import DatabaseHandler
from helpers.preprocessor import Preprocessor
from windows.data_fetcher import DataDownloader
from windows.plotviewer import PlotViewer


def fetch_data():
    downloader = DataDownloader()
    downloader.start()
    return downloader.posts

if DatabaseHandler().read().empty:
    # show a message box
    response = tkinter.messagebox.askokcancel('No Data Found', 'No data found in database. Do you want to fetch data from Reddit?', icon='warning')
    if response:
        posts = fetch_data()
        # ask if user wants to save the data
        response = tkinter.messagebox.askokcancel('Save Data', 'Do you want to save the data for future use?', icon='warning')
        posts = pd.DataFrame(posts)
        posts = Preprocessor(posts).get_preprocessed_data()
        if response:
            DatabaseHandler().write(posts)
else:
    posts = DatabaseHandler().read()

if __name__ == '__main__':
    plot_viewer = PlotViewer(posts)
    plot_viewer.mainloop()
