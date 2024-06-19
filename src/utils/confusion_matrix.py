import numpy as np
import numpy.typing as npt


# "Static" class for reasoning about an N x N confusion matrix
class ConfusionMatrix:
    @classmethod
    def total(cls, confusion_matrix: npt.NDArray[int]):
        return np.sum(confusion_matrix)

    @classmethod
    def calculate_confusion_matrix_metrics(cls, confusion_matrix: npt.NDArray[int]):
        true_positives = np.diag(confusion_matrix)
        false_positives = np.sum(confusion_matrix, axis=0) - true_positives
        false_negatives = np.sum(confusion_matrix, axis=1) - true_positives
        true_negatives = np.sum(confusion_matrix) - (true_positives + false_positives + false_negatives)
        return true_positives, false_positives, true_negatives, false_negatives

    @classmethod
    def calculate_per_class_metrics(cls, confusion_matrix: npt.NDArray[int]):
        true_positives, false_positives, true_negatives, false_negatives = (
            cls.calculate_confusion_matrix_metrics(confusion_matrix))

        with np.errstate(divide="ignore", invalid="ignore"):
            precisions_per_class = np.where((true_positives + false_positives) != 0,
                                            true_positives / (true_positives + false_positives),
                                            0)
            recalls_per_class = np.where((true_positives + false_negatives) != 0,
                                         true_positives / (true_positives + false_negatives),
                                         0)
            accuracy_per_class = np.where((true_positives + false_positives + true_negatives + false_negatives) != 0,
                                          (true_positives + true_negatives) / (true_positives + false_positives + true_negatives + false_negatives),
                                          0)
            f1_score_per_class = np.where((precisions_per_class + recalls_per_class) != 0,
                                          2 * (precisions_per_class * recalls_per_class) / (precisions_per_class + recalls_per_class),
                                          0)

        return precisions_per_class, recalls_per_class, f1_score_per_class, accuracy_per_class

    class Macro:
        @classmethod
        def calculate_overall_metrics(cls, confusion_matrix: npt.NDArray[int]):
            true_positives, false_positives, true_negatives, false_negatives = (
                ConfusionMatrix.calculate_confusion_matrix_metrics(confusion_matrix))

            with np.errstate(divide="ignore", invalid="ignore"):
                precision = np.where((true_positives + false_positives) != 0,
                                     true_positives / (true_positives + false_positives),
                                     0)
                recall = np.where((true_positives + false_negatives) != 0,
                                  true_positives / (true_positives + false_negatives),
                                  0)
                f1_score = np.where((precision + recall) != 0,
                                    2 * (precision * recall) / (precision + recall),
                                    0)
                accuracy = np.where(np.sum(confusion_matrix),
                                    np.sum(true_positives) / np.sum(confusion_matrix),
                                    0)

            mean_precision = np.mean(precision)
            mean_recall = np.mean(recall)
            mean_f1_score = np.mean(f1_score)

            return mean_precision, mean_recall, mean_f1_score, accuracy

    class Micro:
        @classmethod
        def calculate_overall_metrics(cls, confusion_matrix: npt.NDArray[int]):
            true_positives, false_positives, true_negatives, false_negatives = (
                ConfusionMatrix.calculate_confusion_matrix_metrics(confusion_matrix))

            tp_sum = np.sum(true_positives)
            fp_sum = np.sum(false_positives)
            fn_sum = np.sum(false_negatives)

            with np.errstate(divide="ignore", invalid="ignore"):
                precision = np.where((tp_sum + fp_sum),
                                     tp_sum / (tp_sum + fp_sum),
                                     0)
                recall = np.where((tp_sum + fn_sum),
                                  tp_sum / (tp_sum + fn_sum),
                                  0)
                f1_score = np.where((precision + recall),
                                    2 * (precision * recall) / (precision + recall),
                                    0)
                accuracy = np.where(np.sum(confusion_matrix),
                                    np.sum(true_positives) / np.sum(confusion_matrix),
                                    0)

            return precision, recall, f1_score, accuracy
