import pandas as pd

class Preprocessor:
    def __init__(self, dataframe):
        self.df = dataframe
        # Finding Saturated Columns – Columns with same values in all rows
        saturated_cols = []
        for col in self.df.columns:
            first_value = self.df[col].iloc[0]
            if self.df[col].equals(pd.Series([first_value] * len(self.df[col]))):
                saturated_cols.append(col)

        # At this point, we can drop the saturated columns as they don't provide any useful information
        self.df.drop(saturated_cols, axis=1, inplace=True)

        # Replace all NaN values with 0 if the column is numeric or empty string if the column is string
        for col in self.df.columns:
            if self.df[col].dtype == 'float64' or self.df[col].dtype == 'int64':
                self.df[col].fillna(0, inplace=True)
            if self.df[col].dtype == 'object':
                self.df[col].fillna('', inplace=True)

        # Replace all NaN values with 0 if the column is numeric or empty string if the column is string
        for col in self.df.columns:
            if self.df[col].dtype == 'float64' or self.df[col].dtype == 'int64':
                self.df[col].fillna(0, inplace=True)
            if self.df[col].dtype == 'object':
                self.df[col].fillna('', inplace=True)

        # Convert column to string if column is not numeric or boolean
        for col in self.df.columns:
            if self.df[col].dtype != 'float64' and self.df[col].dtype != 'int64' and self.df[col].dtype != 'bool':
                self.df[col] = self.df[col].astype(str)

        # Check for title duplicates
        print('Duplicate titles: {}'.format(self.df['title'].duplicated().sum()))

        # Same post data can be repeated from the API -Delete titles that appear more than once
        self.df.drop_duplicates(subset=['title'], keep='first', inplace=True)

        # Find all columns that contain 'flair'
        columns = list(self.df.columns)
        flair_columns = self.search(columns, 'flair')
        # remove everything from df columns except link_flair_text and author_flair_text
        flair_columns = list(filter(lambda x: x not in ['link_flair_text', 'author_flair_text'], flair_columns))
        self.df.drop(flair_columns, axis=1, inplace=True)

        # Any rows containing [deleted] and [removed] are not useful for our analysis. Find any rows with these values and drop them.
        columns = list(self.df.columns)
        for column in columns:
            self.df = self.df[self.df[column] != '[deleted]']
            self.df = self.df[self.df[column] != '[removed]']
        
        # Remove all posts which are polls - where poll_data is not ""
        try:
            self.df = self.df[self.df['poll_data'] == '']
        except:
            pass

        self.df['created_utc'] = pd.to_datetime(self.df['created_utc'], unit='s')
        self.df['hour'] = self.df['created_utc'].dt.hour
        self.df['day'] = self.df['created_utc'].dt.day_name()
        # self.df.drop('created_utc', axis=1, inplace=True)

        cols_to_keep = ['title', 'selftext', 'link_flair_text', 'subreddit', 'ups', 'num_comments', 'hour', 'day', 'distinguished', 'author_premium', 'subreddit_subscribers', 'author', 'score', 'created_utc', 'upvote_ratio']
        self.df = self.df[cols_to_keep]

        # "distinguished" coloumn has 2 values - "moderator" and "" - We can convert this to a boolean column
        self.df['distinguished'] = self.df['distinguished'].apply(lambda x: True if x == 'moderator' else False)

        # Convert author_premium to boolean
        self.df['author_premium'] = self.df['author_premium'].apply(lambda x: True if x == True else False)

        # Convert title, selftext, link_flair_text, subreddit to string
        self.df['title'] = self.df['title'].astype(str)
        self.df['selftext'] = self.df['selftext'].astype(str)
        self.df['link_flair_text'] = self.df['link_flair_text'].astype(str)
        self.df['subreddit'] = self.df['subreddit'].astype(str)
        self.df['day'] = self.df['day'].astype(str)
        self.df['distinguished'] = self.df['distinguished'].astype(bool)
        self.df['hour'] = self.df['hour'].astype(int)
        self.df['ups'] = self.df['ups'].astype(int)
        self.df['num_comments'] = self.df['num_comments'].astype(int)

    # Supplimentary Column Search Function
    def search(self, array, search_term):
        """
        Returns a list of strings that contain the search term.
        """
        return [string for string in array if search_term in string]


    def get_preprocessed_data(self):
        return self.df