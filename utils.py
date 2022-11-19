import numpy as np
import seaborn as sns
import pandas as pd
from scipy.stats import chi2_contingency


def make_countplot_facet_grid(df: pd.DataFrame, column: str, row: str, x: str):
    g = sns.FacetGrid(df, col=column, row=row)
    g.fig.tight_layout()
    g.map_dataframe(sns.countplot, x=x)

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