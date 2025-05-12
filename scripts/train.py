
import argparse
import os
import numpy as np
from sklearn.metrics import accuracy_score
import joblib
from sklearn.model_selection import train_test_split
import pandas as pd
from azureml.core.run import Run
import xgboost as xgb
from xgboost import XGBClassifier
from azureml.core import Run, Dataset, Workspace
from sklearn.preprocessing import StandardScaler

# Save model for current iteration
run = Run.get_context()

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str)
    parser.add_argument('--learning_rate', type=float, default=0.1, help="Step size shrinkage to prevent overfitting")
    parser.add_argument('--n_estimators', type=int, default=100, help="Number of boosting rounds")
    parser.add_argument('--max_depth', type=int, default=3, help="Maximum depth of a tree")
    parser.add_argument('--random_state', type=int, default=1, help="Random seed for reproducibility")

    args = parser.parse_args()

    run.log("Learning Rate:", np.float(args.learning_rate))
    run.log("Number of Estimators:", np.int(args.n_estimators))
    run.log("Max Depth:", np.int(args.max_depth))
    run.log("Random State:", np.int(args.random_state))

    workspace = run.experiment.workspace  # Get the workspace
    dataset_name = args.data # Get dataset name from arguments

    # Get the Tabular Dataset and convert to DataFrame
    tabular_dataset = Dataset.get_by_name(workspace, name=dataset_name)
    x_df = tabular_dataset.to_pandas_dataframe()

    y_df = x_df.pop("loan_status")

    # Identify numeric and categorical columns
    numeric_cols = x_df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = x_df.select_dtypes(include=['object', 'category']).columns.tolist()

# Scale numeric columns
    if numeric_cols:  # Check if there are any numeric columns
        scaler = StandardScaler()
        x_df[numeric_cols] = scaler.fit_transform(x_df[numeric_cols])
  
    # One-hot encode categorical columns
    if categorical_cols: #check if there are categorical columns
        x_df = pd.get_dummies(x_df, columns=categorical_cols, drop_first=True)
  


    # Split data into train and test sets.
    x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.3, random_state=args.random_state)

    model = xgb.XGBClassifier(
        learning_rate=args.learning_rate,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state
    ).fit(x_train, y_train)

    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    run.log("Accuracy", np.float(accuracy))

    # Save model for current iteration, also include the hyperparameters in filename
    os.makedirs('outputs', exist_ok=True)
    joblib.dump(model, f'outputs/xgboost_lr_{args.learning_rate}_nestimators_{args.n_estimators}_depth_{args.max_depth}_rs_{args.random_state}.joblib')

if __name__ == '__main__':
    main()
