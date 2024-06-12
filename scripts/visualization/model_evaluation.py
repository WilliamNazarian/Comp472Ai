import numpy as np
import pandas as pd
import builtins
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib.gridspec import GridSpec
from pipe import *
from src.types import *
from src.utils.confusion_matrix import ConfusionMatrix


cm = ConfusionMatrix
cm_macro = ConfusionMatrix.Macro
cm_micro = ConfusionMatrix.Micro


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
        ax0.set_title('Overall training metrics (micro)')
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

        axes = [ax0, ax1, ax2, ax3, ax4]
        for ax in axes:
            ax.axhspan(0.9, ax.get_ylim()[1], color='silver', alpha=0.3)
            """
            ax.text(0.5, 0.95, 'Region where y > 0.9', horizontalalignment='center', verticalalignment='top',
                    transform=ax.transAxes, backgroundcolor='silver')
            """

        # create vertical line partition
        line_x = 0.475
        fig.add_artist(plt.Line2D([line_x, line_x], [0, 1],
                                  color="black",
                                  alpha=0.25,
                                  linewidth=1,
                                  transform=fig.transFigure))

        plt.subplots_adjust(hspace=0.3)
        return fig


class TestingVisualizations:
    @staticmethod
    def plot_metrics_per_class(evaluation_results: EvaluationResults):
        def process_metrics(_data, metric_name):
            as_df = pd.DataFrame(_data, columns=["score"])
            as_df.insert(0, "class", ["anger", "engaged", "happy", "neutral"])
            as_df.insert(1, "metric", metric_name)
            return as_df

        confusion_matrix = evaluation_results.confusion_matrix
        precisions, recalls, f1_scores, accuracies = cm.calculate_per_class_metrics(confusion_matrix)

        processed_precisions = process_metrics(precisions, "precision")
        processed_recalls = process_metrics(recalls, "recall")
        processed_f1_scores = process_metrics(f1_scores, "f1_score")
        processed_accuracies = process_metrics(accuracies, "accuracy")

        df = pd.concat([processed_precisions, processed_recalls, processed_f1_scores, processed_accuracies])

        g = sns.catplot(df, kind="bar", x="class", y="score", hue="metric",
                        errorbar="sd", alpha=0.6, height=6)
        g.despine(left=True, bottom=True)
        g.legend.set_title("metric")
        return g.fig

    @staticmethod
    def generate_metrics_per_class_table(evaluation_results: EvaluationResults):
        # getting the data
        confusion_matrix = evaluation_results.confusion_matrix
        precisions, recalls, f1_scores, accuracies = cm.calculate_per_class_metrics(confusion_matrix)

        def floats_to_strings(_data: list):
            return list(_data | select(lambda x: f"{x:.5f}"))

        metrics = ['precision', 'recall', 'f1_score', 'accuracy']
        data = {
            'anger': floats_to_strings(list(precisions)),
            'engaged': floats_to_strings(list(recalls)),
            'happy': floats_to_strings(list(f1_scores)),
            'neutral': floats_to_strings(list(accuracies))
        }

        # generating the plot
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.axis('tight')
        ax.axis('off')

        placeholder_df = pd.DataFrame(data, index=metrics)
        table = ax.table(cellText=placeholder_df.values, colLabels=placeholder_df.columns,
                         rowLabels=placeholder_df.index, cellLoc='center', loc='center')

        # general table properties
        table.auto_set_font_size(False)
        table.set_fontsize(12)

        for key, cell in table.get_celld().items():
            cell.set_height(0.15)
            cell.set_width(0.15)
            cell.set_text_props(ha='left')

            if key[0] == 0:
                cell.set_facecolor('#d3d3d3')

        plt.title('Performance metrics per class on the test set')
        return fig

    @staticmethod
    def generate_overall_metrics_table(evaluation_results: EvaluationResults):
        # getting the data
        confusion_matrix = evaluation_results.confusion_matrix
        macro_precision, macro_recall, macro_f1_score, macro_accuracy = cm_macro.calculate_overall_metrics(
            confusion_matrix)
        micro_precision, micro_recall, micro_f1_score, micro_accuracy = cm_micro.calculate_overall_metrics(
            confusion_matrix)
        accuracy = (macro_accuracy + micro_accuracy) / 2  # should be the same for both

        data = [[macro_precision, macro_recall, macro_f1_score, micro_precision, micro_recall, micro_f1_score, accuracy]]

        # generating the plot
        fig, ax = plt.subplots(figsize=(9, 3))
        ax.axis('tight')
        ax.axis('off')

        placeholder_df = pd.DataFrame('', index=range(len(data) + 2), columns=range(10))
        table = ax.table(cellText=placeholder_df.values,
                         rowLabels=["", ""] + [f'Row {i + 1}' for i in range(len(data))],
                         cellLoc='center', loc='center',
                         )

        # general table properties
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.auto_set_column_width(col=list(range(len(placeholder_df.columns))))

        # setting table text
        table[0, 1].get_text().set_text("macro")
        table[0, 4].get_text().set_text("micro")

        texts: list = ["precision", "recall", "f1-score", "precision", "recall", "f1-score", "accuracy"]
        for col, text in builtins.enumerate(texts):
            table[1, col].get_text().set_text(text)

        for row, data in builtins.enumerate(data):
            for col, metric in builtins.enumerate(data):
                table[row + 2, col].get_text().set_text(f"{metric:.4f}")

        # shading stuff
        macro_color = "#f0f0f0"
        micro_color = "#B2BEB5"

        table[0, 1].set_facecolor(macro_color)
        table[1, 6].set_facecolor("#7393B3")
        for col in range(3):
            table[1, col].set_facecolor(macro_color)

        table[0, 4].set_facecolor(micro_color)
        for col in range(3):
            table[1, col + 3].set_facecolor(micro_color)

        # setting cell properties
        for key, cell in table.get_celld().items():
            cell.set_text_props(ha='center', va='center')
            cell.set_edgecolor('none')
            cell.set_height(0.09)
            cell.set_width(0.1)
            cell.set_text_props(ha='left')

        # Display the plot
        plt.title("Performance metrics of the model on the test set")
        return fig

    @staticmethod
    def generate_confusion_matrix_table(evaluation_results: EvaluationResults):
        confusion_matrix = evaluation_results.confusion_matrix

        # generating the plot
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.axis('tight')
        ax.axis('off')

        row_labels = ['anger', 'engaged', 'happy', 'neutral']
        col_labels = ['anger', 'engaged', 'happy', 'neutral']
        table = ax.table(cellText=confusion_matrix, rowLabels=row_labels, colLabels=col_labels, cellLoc='center', loc='center')

        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.auto_set_column_width(col=list(range(len(col_labels))))

        # shading stuff
        for col in range(4):
            table[1, col].set_facecolor("#f0f0f0")
            table[3, col].set_facecolor("#f0f0f0")

        # setting cell properties
        for key, cell in table.get_celld().items():
            cell.set_text_props(ha='center', va='center')
            cell.set_edgecolor('none')
            cell.set_height(0.09)
            cell.set_width(0.1)
            cell.set_text_props(ha='left')

        # Display the plot
        plt.title('Confusion Matrix Table')
        return fig
