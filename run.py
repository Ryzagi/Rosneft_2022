import pandas as pd
import numpy as np
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import tqdm

DT_PARAMS = {
    0: {'ccp_alpha': 0.0, 'criterion': 'squared_error', 'max_depth': None, 'max_features': None, 'max_leaf_nodes': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'random_state': None, 'splitter': 'best'},
    1: {'ccp_alpha': 0.0, 'criterion': 'squared_error', 'max_depth': None, 'max_features': None, 'max_leaf_nodes': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'random_state': None, 'splitter': 'best'},
    2: {'ccp_alpha': 0.0, 'criterion': 'squared_error', 'max_depth': None, 'max_features': None, 'max_leaf_nodes': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'random_state': None, 'splitter': 'best'},
    3: {'ccp_alpha': 0.0, 'criterion': 'squared_error', 'max_depth': None, 'max_features': None, 'max_leaf_nodes': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'random_state': None, 'splitter': 'best'},
}


def read_signals(filename):
    samples_count = 5000
    c = ['name', 'x', 'y']
    for i in range(0, samples_count):
        c.append(f'v{i}')
    c = c + ['cluster', 'p0', 'p1', 'p2', 'p3']
    df = pd.read_csv(filename, names=c, dtype=np.float32)
    df = df.set_index('name', drop=True)
    return df


def write_signals(df, filename):
    df.to_csv(filename, header=False)


def clusterize(df: pd.DataFrame) -> pd.DataFrame:
    feat_columns = ['x', 'y']
    train_df = df[df['cluster'] != -1].reset_index(drop=False)
    test_df = df[df['cluster'] == -1].reset_index(drop=False)

    clf = neighbors.KNeighborsClassifier(1, weights='distance')
    clf.fit(train_df.loc[:, feat_columns], train_df.loc[:, 'cluster'])

    test_df['cluster'] = clf.predict(test_df.loc[:, feat_columns])

    return pd.concat([train_df, test_df]).reset_index(drop=False)


def train_model(df: pd.DataFrame) -> pd.DataFrame:
    for t_num in range(4):
        model = DecisionTreeRegressor(**DT_PARAMS[t_num])
        features = df[df[f'p{t_num}'] > 0].iloc[:, 3:5004].to_numpy()
        target = df[df[f'p{t_num}'] > 0].iloc[:, 5005 + t_num:5006 + t_num].to_numpy().ravel()
        model.fit(features, target)
        indexes = df[df[f'p{t_num}'] != -1].index.tolist()
        y_pred = model.predict(df.iloc[:, 3:5004].to_numpy())
        for index in tqdm.tqdm(indexes, f'predicting: p{t_num}'):
            y_pred[index] = df.loc[index, f'p{t_num}']
        df[f'p{t_num}'] = y_pred
    return df


if __name__ == "__main__":
    df = read_signals('./data/signals.csv')
    df = clusterize(df)
    df = train_model(df)
    write_signals(df, './data/result.csv')