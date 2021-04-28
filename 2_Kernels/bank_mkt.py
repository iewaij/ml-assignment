import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

def import_dataset(filename):
    """
    Import the dataset from the path.
    
    Parameters
    ----------
        filename : str
            filename with path
    Returns
    -------
        data : DataFrame

    Examples
    --------
        bank_mkt = import_dataset("BankMarketing.csv")
    """
    bank_mkt = pd.read_csv(
        filename,
        na_values=["unknown", "nonexistent"],
        true_values=["yes", "success"],
        false_values=["no", "failure"],
    )
    # Treat pdays = 999 as missing values
    bank_mkt["pdays"] = bank_mkt["pdays"].replace(999, pd.NA)
    # `month` will be encoded to the corresponding number, e.g. "mar" -> 3
    month_map = {
        "mar": 3,
        "apr": 4,
        "may": 5,
        "jun": 6,
        "jul": 7,
        "aug": 8,
        "sep": 9,
        "oct": 10,
        "nov": 11,
        "dec": 12,
    }
    bank_mkt["month"] = bank_mkt["month"].replace(month_map)
    # `day_of_week` will be encoded to the corresponding number, e.g. "wed" -> 3
    dow_map = {"mon": 1, "tue": 2, "wed": 3, "thu": 4, "fri": 5}
    bank_mkt["day_of_week"] = bank_mkt["day_of_week"].replace(dow_map)
    # Convert types, "Int64" is nullable integer data type in pandas
    bank_mkt = bank_mkt.astype(
        dtype={
            "contact": "category",
            "month": "Int64",
            "day_of_week": "Int64",
            "campaign": "Int64",
            "pdays": "Int64",
            "previous": "Int64",
            "poutcome": "boolean",
            "y": "boolean",
        }
    )
    # Drop 12 duplicated rows
    bank_mkt = bank_mkt.drop_duplicates().reset_index(drop=True)
    # Drop demographic data to improve model performace
    bank_mkt = bank_mkt.drop(columns=["age", "job", "marital", "education", "housing", "loan", "default"])
    return bank_mkt

def split_dataset(data, preprocessor=None, random_state=42):
    """
    Split dataset into train, test and validation sets using preprocessor.
    Because the random state of validation set is not specified, the validation set will be different each time when the function is called.
    
    Parameters
    ----------
        data : DataFrame
        preprocessor : Pipeline
        random_state : int
    
    Returns
    -------
        datasets : tuple

    Examples
    --------
        from sklearn.preprocessing import OrdinalEncoder
        data = import_dataset("../data/BankMarketing.csv").interpolate(method="pad").loc[:, ["job", "education", "y"]]
        # To unpack all train, test, and validation sets
        X_train, y_train, X_test, y_test, X_ttrain, y_ttrain, X_validate, y_validate = split_dataset(data, OrdinalEncoder())
        # To unpack train and test sets.
        X_train, y_train, X_test, y_test, *other_sets = split_dataset(data, OrdinalEncoder())
        # To unpack test and validation set
        *other_sets, X_test, y_test, X_ttrain, y_ttrain, X_validate, y_validate = split_dataset(data, OrdinalEncoder())
        # To unpack only train set.
        X_train, y_train, *other_sets = split_dataset(data, OneHotEncoder())
    """
    train_test_split = StratifiedShuffleSplit(
        n_splits=1, test_size=0.2, random_state=random_state
    )
    for train_index, test_index in train_test_split.split(
        data.drop("y", axis=1), data["y"]
    ):
        train_set = data.iloc[train_index]
        test_set = data.iloc[test_index]

    X_train = train_set.drop(["duration", "y"], axis=1)
    y_train = train_set["y"].astype("int").to_numpy()
    X_test = test_set.drop(["duration", "y"], axis=1)
    y_test = test_set["y"].astype("int").to_numpy()

    train_validate_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
    for ttrain_index, validate_index in train_validate_split.split(X_train, y_train):
        ttrain_set = train_set.iloc[ttrain_index]
        validate_set = train_set.iloc[validate_index]

    X_ttrain = ttrain_set.drop(["duration", "y"], axis=1)
    y_ttrain = ttrain_set["y"].astype("int").to_numpy()
    X_validate = validate_set.drop(["duration", "y"], axis=1)
    y_validate = validate_set["y"].astype("int").to_numpy()

    if preprocessor != None:
        X_train = preprocessor.fit_transform(X_train, y_train)
        X_test = preprocessor.transform(X_test)
        X_ttrain = preprocessor.fit_transform(X_ttrain, y_ttrain)
        X_validate = preprocessor.transform(X_validate)

    return (
        X_train,
        y_train,
        X_test,
        y_test,
        X_ttrain,
        y_ttrain,
        X_validate,
        y_validate,
    )

def transform(
    X,
    cut=None,
    gen=None,
    cyclic=None,
    target=None,
    fillna=True,
    to_float=False,
):
    """
    Encode, transform, and generate categorical data in the dataframe.
    Parameters
    ----------
        X : DataFrame
        gen : list, default = None
        cut : list, default = None
        external : list, default = None
        cyclic : list, default = None
        fillna : boolean, default = True
        to_float : boolean, default = False
    Returns
    -------
        X : DataFrame
    Examples
    --------
    bank_mkt = import_dataset("../data/BankMarketing.csv")
    X = dftransform(bank_mkt)
    """
    X = X.copy()

    if gen != None:
        if "year" in gen or "days" in gen:
            X.loc[X.index < 27682, "year"] = 2008
            X.loc[(27682 <= X.index) & (X.index < 39118), "year"] = 2009
            X.loc[39118 <= X.index, "year"] = 2010
            X["year"] = X["year"].astype("int")
        if "days" in gen:
            X["date"] = pd.to_datetime(X[["month", "year"]].assign(day=1))
            X["lehman"] = pd.to_datetime("2008-09-15")
            X["days"] = X["date"] - X["lehman"]
            X["days"] = X["days"].dt.days
            X = X.drop(["lehman", "year", "date"], axis=1)
        if "has_previous" in gen:
            X["has_previous"] = X["previous"] > 0
        if "has_default" in gen:
            X["has_default"] = X["default"].notna()
        if "has_marital" in gen:
            X["has_marital"] = X["marital"].notna()

    if cut != None:
        if "pdays" in cut:
            # Cut pdays into categories
            X["pdays"] = pd.cut(
                X["pdays"],
                [0, 3, 5, 10, 15, 30, 1000],
                labels=[3, 5, 10, 15, 30, 1000],
                include_lowest=True,
            ).astype("Int64")

    if cyclic != None:
        if "month" in cyclic:
            X["month_sin"] = np.sin(2 * np.pi * X["month"] / 12)
            X["month_cos"] = np.cos(2 * np.pi * X["month"] / 12)
            X = X.drop("month", axis=1)
        if "day_of_week" in cyclic:
            X["day_sin"] = np.sin(2 * np.pi * X["day_of_week"] / 5)
            X["day_cos"] = np.cos(2 * np.pi * X["day_of_week"] / 5)
            X = X.drop("day_of_week", axis=1)

    # Transform target encoded feature as str
    if target != None:
        X[target] = X[target].astype("str")

    # Other categorical features will be coded as its order in pandas categorical index
    X = X.apply(
        lambda x: x.cat.codes
        if pd.api.types.is_categorical_dtype(x)
        else (x.astype("Int64") if pd.api.types.is_bool_dtype(x) else x)
    )

    if fillna:
        # Clients who have been contacted but do not have pdays record should be encoded as 999
        # Clients who have not been contacted should be encoded as -999
        X.loc[X["pdays"].isna() & X["poutcome"].notna(), "pdays"] = 999
        X["pdays"] = X["pdays"].fillna(-999)
        # Fill other missing values as -1
        X = X.fillna(-1)
    else:
        X = X.astype("float")

    if to_float:
        X = X.astype("float")

    return X