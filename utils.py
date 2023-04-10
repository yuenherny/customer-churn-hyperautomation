import numpy as np
import seaborn as sns
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.metrics import classification_report, confusion_matrix
import pickle as pkl


def make_countplot_facet_grid(df: pd.DataFrame, column: str, row: str, x: str):
    g = sns.FacetGrid(df, col=column, row=row)
    g.fig.tight_layout()
    g.map_dataframe(sns.countplot, x=x)
    g.set_titles(row_template = '{row_name}', col_template = '{col_name}')


def compute_cramers_v(df, column1, column2):
    """ calculate Cramers V statistic for categorial-categorial association.
        uses correction from Bergsma and Wicher, 
        Journal of the Korean Statistical Society 42 (2013): 323-328

        Ref: https://stackoverflow.com/questions/20892799/using-pandas-calculate-cram%C3%A9rs-coefficient-matrix
    """
    conf_mat = pd.crosstab(df[column1], df[column2])
    chi2 = chi2_contingency(conf_mat)[0]
    n = conf_mat.to_numpy().sum()
    phi2 = chi2/n
    r,k = conf_mat.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))


def inference(pipe, X_train, y_train, X_test, y_test, model_name):
    pipe.fit(X_train, y_train)
    logr_score = pipe.score(X_test.values, y_test.values)
    print("R2 score: ", logr_score)
    y_pred = pipe.predict(X_test.values)

    with open(f"models/{model_name}.pkl", "wb") as f:
        pkl.dump(pipe, f)

    print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

    return pd.DataFrame(classification_report(y_test, y_pred, output_dict=True))