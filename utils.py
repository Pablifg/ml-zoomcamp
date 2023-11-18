from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

def split_train_val_test():
    """ full train, train, validation and test"""
    pass

def train(df_train, y_train, solver='liblinear', C=1.0):
    """Train function"""
    dicts = df_train.to_dict(orient="records")

    dv = DictVectorizer()
    X_train = dv.fit_transform(dicts)

    model = LogisticRegression(solver=solver, C=C, max_iter=1000)
    model.fit(X_train, y_train)

    return dv, model

def predict(df, dv, model):
    """ Predict function """
    dicts = df.to_dict(orient="records")

    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:,1]

    return y_pred