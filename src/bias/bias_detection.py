import scikit_posthocs as sp
import torchvision.datasets as datasets
import pandas as pd
import pipe as pipe
import src.evaluation
import src.data_loader as data_loader

from functools import partial
from scipy.stats import kruskal


transform = data_loader.transform
create_data_loader = data_loader.create_data_loader
evaluate_model = partial(src.evaluation.evaluate_model, logger=None)


def test_for_biases(model):
    # testing
    young_trainset = datasets.ImageFolder(root=r"../dataset/bias/age/young", transform=transform)
    young_dataloader = create_data_loader(young_trainset)
    eval_results_young = evaluate_model(model=model, dataloader=young_dataloader)
    young_metrics = eval_results_young.get_metrics_table_as_df()

    middle_aged_trainset = datasets.ImageFolder(root=r"../dataset/bias/age/middle_aged", transform=transform)
    middle_aged_dataloader = create_data_loader(middle_aged_trainset)
    eval_results_middle_aged = evaluate_model(model=model, dataloader=middle_aged_dataloader)
    middle_aged_metrics = eval_results_middle_aged.get_metrics_table_as_df()

    old_trainset = datasets.ImageFolder(root=r"../dataset/bias/age/old", transform=transform)
    old_dataloader = create_data_loader(old_trainset)
    eval_results_old = evaluate_model(model=model, dataloader=old_dataloader)
    old_metrics = eval_results_old.get_metrics_table_as_df()

    male_trainset = datasets.ImageFolder(root=r"../dataset/bias/gender/male", transform=transform)
    male_dataloader = create_data_loader(male_trainset)
    eval_results_male = evaluate_model(model=model, dataloader=male_dataloader)
    male_metrics = eval_results_male.get_metrics_table_as_df()

    female_trainset = datasets.ImageFolder(root=r"../dataset/bias/gender/female", transform=transform)
    female_dataloader = create_data_loader(female_trainset)
    eval_results_female = evaluate_model(model=model, dataloader=female_dataloader)
    female_metrics = eval_results_female.get_metrics_table_as_df()

    # test 'age'
    young_metrics_list = list(young_metrics.iloc[0].tolist() | pipe.map(lambda x: float(x)))
    middle_aged_metrics_list = list(middle_aged_metrics.iloc[0].tolist() | pipe.map(lambda x: float(x)))
    old_metrics_list = list(old_metrics.iloc[0].tolist() | pipe.map(lambda x: float(x)))

    data = young_metrics_list + middle_aged_metrics_list + old_metrics_list
    groups = ['Young'] * len(young_metrics_list) + ['Middle-aged'] * len(middle_aged_metrics_list) + ['Old'] * len(old_metrics_list)
    df = pd.DataFrame({'Score': data, 'Group': groups})

    kruskal_stat, kruskal_p_value = kruskal(young_metrics_list, middle_aged_metrics_list, old_metrics_list)
    print(f"Kruskal-Wallis Test for 'age' bias attribute: H={kruskal_stat}, p-value={kruskal_p_value}")
    if kruskal_p_value < 0.05:
        print("The differences among the groups are statistically significant.")

        nemenyi_test = sp.posthoc_nemenyi(df, val_col='Score', group_col='Group')
        print("\nNemenyi test results:\n", nemenyi_test)
    else:
        print("The differences among the groups are not statistically significant.")

    # test 'gender'
    male_metrics_list = list(male_metrics.iloc[0].tolist() | pipe.map(lambda x: float(x)))
    female_metrics_list = list(female_metrics.iloc[0].tolist() | pipe.map(lambda x: float(x)))

    data = male_metrics_list + female_metrics_list
    groups = ['Male'] * len(male_metrics_list) + ['Female'] * len(female_metrics_list)
    df_gender = pd.DataFrame({'Score': data, 'Group': groups})

    kruskal_stat, kruskal_p_value = kruskal(male_metrics_list, female_metrics_list)
    print(f"\n\nKruskal-Wallis Test for 'gender' bias attribute: H={kruskal_stat}, p-value={kruskal_p_value}")
    if kruskal_p_value < 0.05:
        print("The differences among the groups are statistically significant.")

        nemenyi_test = sp.posthoc_nemenyi(df_gender, val_col='Score', group_col='Group')
        print("\nNemenyi test results:\n", nemenyi_test)
    else:
        print("The differences among the groups are not statistically significant.")

    # format metrics for df
    metrics = [
        young_metrics_list,
        middle_aged_metrics_list,
        old_metrics_list,
        male_metrics_list,
        female_metrics_list
    ]

    trainsets = [
        young_trainset,
        middle_aged_trainset,
        old_trainset,
        male_trainset,
        female_trainset,
    ]

    metrics_formatted = list(metrics
                             | pipe.map(__extract_metrics)
                             | pipe.Pipe(__restructure_metrics_list_list)
                             | pipe.map(__format_metrics))

    precisions = metrics_formatted[0]
    recalls = metrics_formatted[1]
    f1_scores = metrics_formatted[2]
    accuracies = metrics_formatted[3]
    images = __format_metrics(list(trainsets | pipe.map(lambda _trainset: _trainset.__len__())), decimal_places=1)

    # make and return styled df
    data = {
        'Attribute': ['Age', '', '', '', '‎', 'Gender', '', '', '‎', 'Overall System Mean'],
        'Group': ['Young', 'Middle-aged', 'Senior', '(Mean)', '', 'Male', 'Female', '(Mean)', '', ''],
        '#Images': images,
        'Accuracy': accuracies,
        'Precision': precisions,
        'Recall': recalls,
        'F1-Score': f1_scores,
    }

    df = pd.DataFrame(data)
    df.index = [''] * len(df)
    return df


def __extract_metrics(metrics_list):
    return metrics_list[:3] + metrics_list[-1:]


def __restructure_metrics_list_list(metrics_list_list):
    _precisions = []
    _recalls = []
    _f1_scores = []
    _accuracies = []
    for metrics_list in metrics_list_list:
        _precisions.append(metrics_list[0])
        _recalls.append(metrics_list[1])
        _f1_scores.append(metrics_list[2])
        _accuracies.append(metrics_list[3])

    return [_precisions, _recalls, _f1_scores, _accuracies]


def __format_metrics(metrics_list, decimal_places=4):
    age_avg = sum(metrics_list[:3]) / 3
    gender_avg = sum(metrics_list[-2:]) / 2
    total_avg = sum(metrics_list) / len(metrics_list)

    age_avg = format(age_avg, f".{decimal_places}f")
    gender_avg = format(gender_avg, f".{decimal_places}f")
    total_avg = format(total_avg, f".{decimal_places}f")

    return (list(metrics_list[:3] | pipe.map(lambda x: format(x, f".{decimal_places}f")))
            + [age_avg, '']
            + list(metrics_list[-2:] | pipe.map(lambda x: format(x, f".{decimal_places}f")))
            + [gender_avg, '', total_avg])
