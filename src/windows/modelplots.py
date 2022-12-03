import customtkinter
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import json
import tkinter
import numpy as np

class ModelPlots(customtkinter.CTkToplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.title("Model Plots")
        posx = int(self.winfo_screenwidth()/2 - 400)
        posy = int(self.winfo_screenheight()/2 - 300)
        self.geometry("800x600+{}+{}".format(posx, posy))
        self.grab_set()
        self.focus_set()
        self.resizable(True, True)
        with open(f'models/ups_metrics.json', 'r') as f:
            self.ups_metrics = json.load(f)
        with open(f'models/num_comments_metrics.json', 'r') as f:
            self.num_comments_metrics = json.load(f)

        self.create_widgets()

    def create_widgets(self):
        self.tabview = customtkinter.CTkTabview(self)
        self.tabview.add('R-Squares')
        self.tabview.add('MAE')
        self.tabview.add('MSE')
        self.tabview.add('R-Square Comparison')
        self.tabview.add('Predictions')
        self.tabview.add('Residuals')

        # R-Squares
        fig, ax = plt.subplots(1, 2, figsize=(15, 3), dpi=36)
        ups_m = {}
        for k, v in self.ups_metrics.items():
            ups_m[k] = v['r2']
        sns.barplot(x=list(ups_m.keys()), y=list(ups_m.values()), ax=ax[0], palette='Blues_d')
        ax[0].set_title('R-Square for Ups')

        num_comments_m = {}
        for k, v in self.num_comments_metrics.items():
            num_comments_m[k] = v['r2']
        sns.barplot(x=list(num_comments_m.keys()), y=list(num_comments_m.values()), ax=ax[1], palette='Greens_d')
        ax[1].set_title('R-Square for Number of Comments')

        for i in range(2):
            ax[i].set_xticklabels(ax[i].get_xticklabels(), rotation=45)
        self.r2plot = FigureCanvasTkAgg(fig, self.tabview.tab('R-Squares'))
        self.r2plot.figure.tight_layout()
        self.r2plot.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)

        # MAE
        fig, ax = plt.subplots(1, 2, figsize=(15, 3), dpi=36)
        ups_m = {}
        for k, v in self.ups_metrics.items():
            ups_m[k] = v['mae']
        sns.barplot(x=list(ups_m.keys()), y=list(ups_m.values()), ax=ax[0], palette='Reds_d')
        ax[0].set_title('MAE for Ups')

        num_comments_m = {}
        for k, v in self.num_comments_metrics.items():
            num_comments_m[k] = v['mae']
        sns.barplot(x=list(num_comments_m.keys()), y=list(num_comments_m.values()), ax=ax[1], palette='Oranges_d')
        ax[1].set_title('MAE for Number of Comments')

        for i in range(2):
            ax[i].set_xticklabels(ax[i].get_xticklabels(), rotation=45)
        self.maeplot = FigureCanvasTkAgg(fig, self.tabview.tab('MAE'))
        self.maeplot.figure.tight_layout()
        self.maeplot.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)

        # MSE
        fig, ax = plt.subplots(1, 2, figsize=(15, 3), dpi=36)
        ups_m = {}
        for k, v in self.ups_metrics.items():
            ups_m[k] = v['mse']
        sns.barplot(x=list(ups_m.keys()), y=list(ups_m.values()), ax=ax[0], palette='Purples_d')
        ax[0].set_title('MSE for Ups')

        num_comments_m = {}
        for k, v in self.num_comments_metrics.items():
            num_comments_m[k] = v['mse']
        sns.barplot(x=list(num_comments_m.keys()), y=list(num_comments_m.values()), ax=ax[1], palette='Greys_d')
        ax[1].set_title('MSE for Number of Comments')

        for i in range(2):
            ax[i].set_xticklabels(ax[i].get_xticklabels(), rotation=45)
        self.mseplot = FigureCanvasTkAgg(fig, self.tabview.tab('MSE'))
        self.mseplot.figure.tight_layout()
        self.mseplot.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)
        
        # R-Square Comparison
        fig, ax = plt.subplots(1, 2, figsize=(20, 5), dpi=36)
        # Ups
        ax[0].set_title('Ups')
        ax[0].set_xlabel('Model')
        ax[0].set_ylabel('R2 Score')
        sns.barplot(x=list(self.ups_metrics.keys()), y=[r2['r2'] for r2 in self.ups_metrics.values()], ax=ax[0], palette='Blues_d')
        ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=45)
        # Number of Comments
        ax[1].set_title('Number of Comments')
        ax[1].set_xlabel('Model')
        ax[1].set_ylabel('R2 Score')
        sns.barplot(x=list(self.num_comments_metrics.keys()), y=[r2['r2'] for r2 in self.num_comments_metrics.values()], ax=ax[1], palette='Greens_d')
        ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=45)
        self.r2comparisonplot = FigureCanvasTkAgg(fig, self.tabview.tab('R-Square Comparison'))
        self.r2comparisonplot.figure.tight_layout()
        self.r2comparisonplot.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)

        # Predictions
        fig, ax = plt.subplots(7, 2, figsize=(12, 16), dpi=24)
        for i, (k, v) in enumerate(self.ups_metrics.items()):
            # ups
            ax[i, 0].set_title('Ups - {}'.format(k))
            ax[i, 0].set_xlabel('Actual')
            ax[i, 0].set_ylabel('Predicted')
            sns.regplot(x=v['actual'], y=v['pred'], ax=ax[i, 0], color='blue', scatter_kws={'alpha': 0.3})

            # num_comments
            ax[i, 1].set_title('Number of Comments - {}'.format(k))
            ax[i, 1].set_xlabel('Actual')
            ax[i, 1].set_ylabel('Predicted')
            sns.regplot(x=self.num_comments_metrics[k]['actual'], y=self.num_comments_metrics[k]['pred'], ax=ax[i, 1], color='green', scatter_kws={'alpha': 0.3})
        self.predplot = FigureCanvasTkAgg(fig, self.tabview.tab('Predictions'))
        self.predplot.figure.tight_layout()
        self.predplot.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)

        # Residuals
        fig, ax = plt.subplots(7, 6, figsize=(20, 16), dpi=16)
        for i, (k, v) in enumerate(self.ups_metrics.items()):
            # ups model
            ax[i, 0].set_title(k + ' - Ups Residuals')
            ax[i, 0].set_xlabel('Residuals')
            ax[i, 0].set_ylabel('Frequency')
            sns.distplot(np.array(v['actual']) - np.array(v['pred']), ax=ax[i, 0], color='blue', kde=False)

            ax[i, 1].set_title(k + ' Ups Test Scores')
            ax[i, 1].set_xlabel('Ups')
            ax[i, 1].set_ylabel('Frequency')
            sns.distplot(v['actual'], ax=ax[i, 1], color='blue', kde=False)

            ax[i, 2].set_title(k + ' Ups Predicted Scores')
            ax[i, 2].set_xlabel('Ups')
            ax[i, 2].set_ylabel('Frequency')
            sns.distplot(v['pred'], ax=ax[i, 2], kde=False, color='red')

            # num_comments model
            ax[i, 3].set_title(k + ' - Number of Comments Residuals')
            ax[i, 3].set_xlabel('Residuals')
            ax[i, 3].set_ylabel('Frequency')
            sns.distplot(np.array(self.num_comments_metrics[k]['actual']) - np.array(self.num_comments_metrics[k]['pred']), ax=ax[i, 3], color='green', kde=False)

            ax[i, 4].set_title(k + ' Number of Comments Test Scores')
            ax[i, 4].set_xlabel('Number of Comments')
            ax[i, 4].set_ylabel('Frequency')
            sns.distplot(self.num_comments_metrics[k]['actual'], ax=ax[i, 4], kde=False, color='green')

            ax[i, 5].set_title(k + ' Number of Comments Predicted Scores')
            ax[i, 5].set_xlabel('Number of Comments')
            ax[i, 5].set_ylabel('Frequency')
            sns.distplot(self.num_comments_metrics[k]['pred'], ax=ax[i, 5], kde=False, color='red')
        self.residualplot = FigureCanvasTkAgg(fig, self.tabview.tab('Residuals'))
        self.residualplot.figure.tight_layout()
        self.residualplot.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)

        self.tabview.pack(fill='both', expand=True)
