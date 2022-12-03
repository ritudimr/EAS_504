import time
import tkinter
import tkinter.messagebox
from tkinter import ttk

import customtkinter
import requests

from helpers.subreddits import SUBREDDITS


class DataDownloader:
    posts = []
    after = None
    downloaded = 0
    start_time = time.time()

    def __init__(self):
        self.root = tkinter.Tk()
        self.root.title('Downloading Data - 0%')
        
        # center the window
        posx = int(self.root.winfo_screenwidth() / 2 - 250)
        posy = int(self.root.winfo_screenheight() / 2 - 50)
        self.root.geometry(f'500x100+{posx}+{posy}')

        self.root.resizable(False, False)
        self.root.protocol('WM_DELETE_WINDOW', self.on_closing)
        self.subreddits = SUBREDDITS
        self.posts_per_subreddit = 1000
        
        self.progress = ttk.Progressbar(self.root, orient='horizontal', length=500, mode='determinate')
        self.progress['value'] = 0
        self.progress['maximum'] = 100

        self.download_label = tkinter.Label(self.root, text='Downloading: 0 / {} Posts'.format(len(SUBREDDITS) * self.posts_per_subreddit))

        self.download_label.pack(fill='x', padx=10, pady=10, side= tkinter.TOP, anchor='w')
        self.progress.pack(fill='x', padx=10, pady=10)

        self.root.bind('<<DownloadComplete>>', lambda e: self.on_closing())
        client_id = 'dog1LGxsD9M3bXtglOzKsQ'
        client_secret = 'nc-HHPBGtz51-_r4vNLGcuCHmT39Lw'
        username = 'NoSeason1949'
        password = 'Password@1234'
        auth = requests.auth.HTTPBasicAuth(client_id, client_secret)
        data = {
            'grant_type': 'password',
            'username': username,
            'password': password,
        }
        headers = {'User-Agent': 'RedditTest/0.1 by {}'.format(username)}
        res = requests.post('https://www.reddit.com/api/v1/access_token',
                            auth=auth, data=data, headers=headers)
        self.token = res.json()['access_token']
        self.headers = {**headers, **{'Authorization': f"bearer {self.token}"}}

    def on_closing(self):
        if self.downloaded < self.posts_per_subreddit * len(self.subreddits):
            if tkinter.messagebox.askokcancel('Quit', 'Are you sure you want to quit?'):
                self.root.destroy()
        else:
            self.root.destroy()

    def download(self):
        for subreddit in self.subreddits:
            subreddit_download = 0
            while subreddit_download < self.posts_per_subreddit:
                url = f"https://oauth.reddit.com/r/{subreddit}/hot?limit=100"
                if self.after:
                     url += f"&after={self.after}"
                res = requests.get(url, headers=self.headers)
                if res.status_code == 200:
                    data = res.json()
                    self.after = data['data']['after']
                    for post in data['data']['children']:
                        self.posts.append(post['data'])
                        subreddit_download += 1
                        self.downloaded += 1
                        dwval = (self.downloaded / (self.posts_per_subreddit * len(self.subreddits))) * 100
                        self.progress['value'] = float(dwval) if dwval < 100 else 100
                        self.root.title(f'Downloading Data - {int(self.progress["value"])}%')
                        time_remaining = (time.time() - self.start_time) / (self.downloaded / (self.posts_per_subreddit * len(self.subreddits))) - (time.time() - self.start_time)
                        time_remaining = time.strftime('%H:%M:%S', time.gmtime(time_remaining))
                        self.download_label['text'] = f'Downloading: {self.downloaded} / {self.posts_per_subreddit * len(self.subreddits)} Posts - {time_remaining} Remaining' if self.downloaded < self.posts_per_subreddit * len(self.subreddits) else 'Download Complete'
                        self.root.update()
                    
        self.root.event_generate('<<DownloadComplete>>', when='tail')

    def start(self):
        self.root.after(0, self.download)
        self.root.mainloop()
