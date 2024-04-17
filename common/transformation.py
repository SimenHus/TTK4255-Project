


def transform(X, R, t):
    return R.T@(X - t)