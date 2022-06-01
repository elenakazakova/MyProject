import warnings

import numpy as np
import pandas as pd



def select_analogs(analogs, inds):
    out = np.empty(len(analogs))
    for i, ind in enumerate(inds):
        out[i] = analogs[i, ind]
    return out


class NamedColumnBaseEstimator(BaseEstimator):
    def _validate_data(
        self,
        X='no_validation',
        y='no_validation',
        **check_params,
    ):
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns
        else:
            feature_names = None
        X, y = super()._validate_data(X, y=y, **check_params)
        if feature_names is not None:
            X = pd.DataFrame(X, columns=feature_names)
        return X, y


class AnalogBase(RegressorMixin, NamedColumnBaseEstimator):
    _fit_attributes = ['kdtree_', 'X_', 'y_', 'k_']

    def fit(self, X, y):

        X, y = self._validate_data(X, y=y, y_numeric=True)

        if len(X) >= self.n_analogs:
            self.k_ = self.n_analogs
        else:
            warnings.warn('length of X is less than n_analogs, setting n_analogs = len(X)')
            self.k_ = len(X)

        kdtree_kwargs = default_none_kwargs(self.kdtree_kwargs)
        self.kdtree_ = KDTree(X, **kdtree_kwargs)

        self.X_ = X
        self.y_ = y

        return self

    def _more_tags(self):
        return {
            '_xfail_checks': {
                'check_fit_score_takes_y': 'GARD models output 3 columns pandas dataframe instead of one during predict',
                'check_pipeline_consistency': 'GARD models output 3 columns pandas dataframe instead of one during predict',
                'check_regressors_train': 'GARD models output 3 columns pandas dataframe instead of one during predict',
            },
        }


class AnalogMethod(AnalogBase):

    n_outputs = 3
    output_names = ['pred', 'exceedance_prob', 'prediction_error']

    def __init__(
        self,
        n_analogs=200,
        thresh=None,
        kdtree_kwargs=None,
        query_kwargs=None,
        logistic_kwargs=None,
        lr_kwargs=None,
    ):
        self.n_analogs = n_analogs
        self.thresh = thresh
        self.kdtree_kwargs = kdtree_kwargs
        self.query_kwargs = query_kwargs
        self.logistic_kwargs = logistic_kwargs
        self.lr_kwargs = lr_kwargs

    def predict(self, X):
        
        return_df = isinstance(X, pd.DataFrame)
        check_is_fitted(self)
        X = check_array(X)

        logistic_kwargs = default_none_kwargs(self.logistic_kwargs)
        logistic_model = LogisticRegression(**logistic_kwargs) if self.thresh is not None else None

        lr_kwargs = default_none_kwargs(self.lr_kwargs)
        lr_model = LinearRegression(**lr_kwargs)

        out = np.empty((len(X), self.n_outputs), dtype=np.float64)
        for i in range(len(X)):
            # predict for this time step
            out[i] = self._predict_one_step(
                logistic_model,
                lr_model,
                X[None, i],
            )

        if return_df:
            return pd.DataFrame(out, columns=self.output_names)
        return out

    def _predict_one_step(self, logistic_model, lr_model, X):
        query_kwargs = default_none_kwargs(self.query_kwargs)
        inds = self.kdtree_.query(X, k=self.k_, return_distance=False, **query_kwargs).squeeze()

        x = np.asarray(self.kdtree_.data)[inds]
        y = self.y_[inds]

        if self.thresh is not None:
            exceed_ind = y > self.thresh
        else:
            exceed_ind = np.ones(len(y), dtype=bool)

        binary_y = exceed_ind.astype(np.int8)
        if not np.all(binary_y == 1):
            logistic_model.fit(x, binary_y)
            exceedance_prob = logistic_model.predict_proba(X)[0, 0]
        else:
            exceedance_prob = 1.0

        lr_model.fit(x[exceed_ind], y[exceed_ind])

        y_hat = lr_model.predict(x[exceed_ind])
        error = mean_squared_error(y[exceed_ind], y_hat, squared=False)

        predicted = lr_model.predict(X)

        return [predicted, exceedance_prob, error]


class PureAnalog(AnalogBase):

    n_outputs = 3
    output_names = ['pred', 'exceedance_prob', 'prediction_error']

    def __init__(
        self,
        n_analogs=200,
        kind='best_analog',
        thresh=None,
        kdtree_kwargs=None,
        query_kwargs=None,
    ):
        self.n_analogs = n_analogs
        self.kind = kind
        self.thresh = thresh
        self.kdtree_kwargs = kdtree_kwargs
        self.query_kwargs = query_kwargs

    def predict(self, X):
        
        return_df = isinstance(X, pd.DataFrame)
        check_is_fitted(self)
        X = check_array(X)

        if self.kind == 'best_analog' or self.n_analogs == 1:
            k = 1
            kind = 'best_analog'
        else:
            k = self.k_
            kind = self.kind

        query_kwargs = default_none_kwargs(self.query_kwargs)
        dist, inds = self.kdtree_.query(X, k=k, **query_kwargs)

        analogs = np.take(self.y_, inds, axis=0)

        if self.thresh is not None:
            analog_mask = analogs > self.thresh
            masked_analogs = np.where(analog_mask, analogs, np.nan)

        if kind == 'best_analog':
            predicted = analogs[:, 0]

        elif kind == 'sample_analogs':
            rand_inds = np.random.randint(low=0, high=k, size=len(X))
            predicted = select_analogs(analogs, rand_inds)

        elif kind == 'weight_analogs':
            tiny = 1e-20
            weights = 1.0 / np.where(dist == 0, tiny, dist)
            if self.thresh is not None:
                predicted = np.average(masked_analogs, weights=weights, axis=1)
            else:
                predicted = np.average(analogs.squeeze(), weights=weights, axis=1)

        elif kind == 'mean_analogs':
            if self.thresh is not None:
                predicted = masked_analogs.mean(axis=1)
            else:
                predicted = analogs.mean(axis=1)

        else:
            raise ValueError('got unexpected kind %s' % kind)

        if self.thresh is not None:
            predicted = np.nan_to_num(predicted, nan=0.0)
            prediction_error = masked_analogs.std(axis=1)
            exceedance_prob = np.where(analog_mask, 1, 0).mean(axis=1)
        else:
            prediction_error = analogs.std(axis=1)
            exceedance_prob = np.ones(len(X), dtype=np.float64)

        if return_df:
            out = pd.DataFrame(
                {
                    'pred': predicted,
                    'exceedance_prob': exceedance_prob,
                    'prediction_error': prediction_error,
                }
            )
            return out[self.output_names]
        else:
            predicted = predicted.reshape(-1, 1)
            exceedance_prob = exceedance_prob.reshape(-1, 1)
            prediction_error = prediction_error.reshape(-1, 1)
            return np.hstack((predicted, exceedance_prob, prediction_error))
