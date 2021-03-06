#!/usr/bin/env python3
import argparse
import sys

import numpy as np
import sklearn.datasets
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
import sklearn.pipeline
import sklearn.preprocessing

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument("--test_ratio", default=0.5, type=float, help="Test set size ratio")
    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)

    # Load digit dataset
    dataset = sklearn.datasets.load_digits()
    print(dataset.DESCR, file=sys.stderr)

    # Split the data randomly to train and test using `sklearn.model_selection.train_test_split`,
    # with `test_size=args.test_ratio` and `random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        dataset.data, dataset.target, test_size=args.test_ratio, random_state=args.seed)

    # TODO: Create a pipeline, which
    # 1. performs sklearn.preprocessing.MinMaxScaler()
    # 2. performs sklearn.preprocessing.PolynomialFeatures()
    # 3. performs sklearn.linear_model.LogisticRegression(multi_class="multinomial", random_state=args.seed)
    #
    # Then, using sklearn.model_selection.StratifiedKFold(5), evaluate crossvalidated
    # train performance of all combinations of the the following parameters:
    # - polynomial degree: 1, 2
    # - LogisticRegression regularization C: 0.01, 1, 100
    # - LogisticRegression solver: lbfgs, sag
    #
    # For the best combination of parameters, compute the test set accuracy.
    #
    # The easiest way is to use `sklearn.model_selection.GridSearchCV`.

    min_max = sklearn.preprocessing.MinMaxScaler()
    poly = sklearn.preprocessing.PolynomialFeatures()
    logi = sklearn.linear_model.LogisticRegression(multi_class="multinomial", random_state=args.seed)

    methods = sklearn.pipeline.Pipeline(steps=[('min_max', min_max),
                                               ('poly', poly),
                                               ('logi', logi)])

    params = {'poly__degree': [1, 2],
              'logi__C': [0.01, 1, 100],
              'logi__solver': ['lbfgs', 'sag']}

    grid = sklearn.model_selection.GridSearchCV(methods, params, cv=5)

    grid.fit(train_data, train_target)

    params = grid.best_params_

    C, degree, solver = params['logi__C'], params['poly__degree'], params['logi__solver']

    min_max = sklearn.preprocessing.MinMaxScaler()
    poly = sklearn.preprocessing.PolynomialFeatures(degree=degree)
    logi = sklearn.linear_model.LogisticRegression(multi_class="multinomial",
                                                   random_state=args.seed,
                                                   C=C, solver=solver)

    train_data = min_max.fit_transform(train_data)
    train_data = poly.fit_transform(train_data)
    test_data = min_max.transform(test_data)
    test_data = poly.transform(test_data)

    model = logi.fit(train_data,train_target)
    preds = model.predict(test_data)



    test_accuracy = np.mean(test_target == preds)
    print("{:.2f}".format(100 * test_accuracy))
