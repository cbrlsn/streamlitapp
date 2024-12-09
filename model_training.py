import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
import joblib
import os

def train_and_save_model(df, model_path="best_catboost_model.pkl", metadata_path="model_metadata.pkl"):
    """
    Train, tune, and save a CatBoost model. If the model already exists, it loads the pre-trained model.
    Saves both the model and metadata.
    """

    if os.path.exists(model_path) and os.path.exists(metadata_path):
        # Load the pre-trained model and metadata if they exist
        print("Loading pre-trained model and metadata...")
        model = joblib.load(model_path)
        metadata = joblib.load(metadata_path)
        print("Model and metadata loaded successfully!")
        return model, metadata

    # Separating target and features
    X = df.drop(columns=['Price', 'Table', 'Depth'])  # Exclude unnecessary columns
    y = df['Price']  # Target variable

    # Encode categorical variables
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Define the CatBoost regressor
    catboost_model = CatBoostRegressor(cat_features=categorical_features, random_state=42, verbose=0)

    # Define the parameter grid for tuning
    param_grid = {
        'iterations': [500, 1000, 1500],
        'learning_rate': [0.01, 0.05, 0.1],
        'depth': [4, 6, 8, 10],
        'l2_leaf_reg': [1, 3, 5, 7],
        'bagging_temperature': [0, 1, 3],
        'random_strength': [0.5, 1, 1.5],
        'border_count': [32, 64, 128],
    }

    # Use RandomizedSearchCV for tuning
    random_search = RandomizedSearchCV(
        estimator=catboost_model,
        param_distributions=param_grid,
        n_iter=20,
        scoring='neg_mean_squared_error',
        cv=3,
        verbose=2,
        n_jobs=-1,
        random_state=42
    )

    print("Tuning the model... This may take a while!")
    random_search.fit(X_train, y_train)

    # Extract the best parameters and best model
    best_model = random_search.best_estimator_
    best_params = random_search.best_params_
    print("Best Parameters:", best_params)

    # Evaluate the best model on the test set
    best_model_predictions = best_model.predict(X_test)
    best_mse = mean_squared_error(y_test, best_model_predictions)
    best_rmse = np.sqrt(best_mse)
    print(f"Best Model RMSE on Test Data: {best_rmse}")

    # Save the tuned model and metadata to files
    metadata = {
        'columns': X.columns.tolist(),
        'categorical_features': categorical_features,
    }
    joblib.dump(best_model, model_path)
    joblib.dump(metadata, metadata_path)
    print(f"Model and metadata saved to {model_path} and {metadata_path}")

    return best_model, metadata
