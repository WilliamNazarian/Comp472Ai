import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib.gridspec import GridSpec
from pipe import *
from scripts.model.types import *


class TrainingVisualizations:
    @staticmethod
    def __get_overall_training_metrics(training_logger: TrainingLogger):
        def process_data(data):
            epochs = list(range(len(data)))
            df_data = list(data
                           | map(lambda x: np.mean(x))
                           | izip(epochs))

            return pd.DataFrame(data=df_data, columns=["score", "epoch"])

        training_precision_history = training_logger.training_precision_history | Pipe(process_data)
        training_recall_history = training_logger.training_recall_history | Pipe(process_data)
        training_accuracy_history = training_logger.training_accuracy_history | Pipe(process_data)
        training_f1_score_history = training_logger.training_f1_score_history | Pipe(process_data)

        training_precision_history['metric'] = 'precision'
        training_recall_history['metric'] = 'recall'
        training_accuracy_history['metric'] = 'accuracy'
        training_f1_score_history['metric'] = 'f1-score'

        return pd.concat(
            [training_precision_history,
             training_recall_history,
             training_accuracy_history,
             training_f1_score_history])

    @staticmethod
    def __process_metric_history(metric_history):
        def process_column(data):
            epochs = list(range(len(data)))
            return pd.DataFrame(data=list(zip(epochs, data)), columns=["epoch", "score"])

        stacked_df = pd.DataFrame(np.column_stack(metric_history))
        rows = [stacked_df.iloc[i].values for i in range(len(stacked_df))]

        individual_dfs = list(rows | map(lambda x: process_column(x)))
        individual_dfs[0]['class'] = 'anger'
        individual_dfs[1]['class'] = 'engaged'
        individual_dfs[2]['class'] = 'happy'
        individual_dfs[3]['class'] = 'neutral'
        return pd.concat([individual_dfs[0], individual_dfs[1], individual_dfs[2], individual_dfs[3]])

    @staticmethod
    def plot_training_metrics(training_logger: TrainingLogger):

        # preparing the data
        overall_metrics = TrainingVisualizations.__get_overall_training_metrics(training_logger)
        precisions_per_class = TrainingVisualizations.__process_metric_history(training_logger.training_precision_history)
        recalls_per_class = TrainingVisualizations.__process_metric_history(training_logger.training_recall_history)
        accuracies_per_class = TrainingVisualizations.__process_metric_history(training_logger.training_accuracy_history)
        f1_scores_per_class = TrainingVisualizations.__process_metric_history(training_logger.training_f1_score_history)

        # doing the actual plotting
        sns.set_theme(style="darkgrid")

        fig = plt.figure(figsize=(20, 11))
        gs = GridSpec(2, 3, width_ratios=[2, 1, 1])

        ax0 = fig.add_subplot(gs[:, 0])
        ax0.set_title('Overall training metrics')
        plot0 = sns.lineplot(ax=ax0, x="epoch", y="score", hue="metric", data=overall_metrics, marker='o')

        ax1 = fig.add_subplot(gs[0, 1])
        ax1.set_title('Precision per class')
        plot1 = sns.lineplot(ax=ax1, x="epoch", y="score", hue="class", data=precisions_per_class, marker='o')

        ax2 = fig.add_subplot(gs[1, 1])
        ax2.set_title('Recall per class')
        plot2 = sns.lineplot(ax=ax2, x="epoch", y="score", hue="class", data=recalls_per_class, marker='o')

        ax3 = fig.add_subplot(gs[0, 2])
        ax3.set_title('Accuracy per class')
        plot3 = sns.lineplot(ax=ax3, x="epoch", y="score", hue="class", data=accuracies_per_class, marker='o')

        ax4 = fig.add_subplot(gs[1, 2])
        ax4.set_title('F1-score per class')
        plot4 = sns.lineplot(ax=ax4, x="epoch", y="score", hue="class", data=f1_scores_per_class, marker='o')

        # create vertical line partition
        line_x = 0.475
        fig.add_artist(plt.Line2D([line_x, line_x], [0, 1],
                                  color="black",
                                  alpha=0.25,
                                  linewidth=1,
                                  transform=fig.transFigure))

        plt.subplots_adjust(hspace=0.3)
