# Credit Loan Approval Prediction using Azure ML

The ability to accurately predict the approval status of a loan application is crucial for financial institutions to mitigate risk and streamline operations. This project focuses on developing a classification model to predict loan approval (Loan_Status), where 0 indicates 'not approved' and 1 indicates 'approved'.

The dataset utilized in this study is a synthetic dataset inspired by the original Credit Risk dataset from Kaggle and further enriched with variables drawn from Financial Risk for Loan Approval data. This enhanced dataset provides a robust foundation for building a predictive model.

The primary objective of this project is twofold: first, to leverage the capabilities of Azure Automated ML to rapidly identify a high-performing classification model for loan approval prediction. Second, we will explore the use of Azure HyperDrive to conduct a more granular hyperparameter tuning process, aiming to further optimize model performance.

Ultimately, the best-performing model from either Automated ML or HyperDrive training will be deployed as an endpoint, demonstrating a practical application of machine learning in a financial context. This report details the process of data exploration (as needed), model development using both Azure services, evaluation of the resulting models, and the deployment of the chosen model.


## Dataset

This project utilizes a synthetic dataset comprising 45,000 entries and 14 features relevant to credit risk assessment and loan approval. A summary of the dataset's structure and variable types is as follows:

* **Number of Instances:** 45,000
* **Number of Features:** 14

The features included in the dataset are:

* `person_age`: (float64) Age of the loan applicant.
* `person_gender`: (object) Gender of the loan applicant.
* `person_education`: (object) Education level of the loan applicant.
* `person_income`: (float64) Annual income of the loan applicant.
* `person_emp_exp`: (int64) Years of employment experience of the loan applicant.
* `person_home_ownership`: (object) Home ownership status of the loan applicant.
* `loan_amnt`: (float64) The amount of the loan requested.
* `loan_intent`: (object) The stated purpose of the loan.
* `loan_int_rate`: (float64) The interest rate of the loan.
* `loan_percent_income`: (float64) Loan amount as a percentage of the applicant's income.
* `cb_person_cred_hist_length`: (float64) Length of the applicant's credit history (in years).
* `credit_score`: (int64) Credit score of the applicant.
* `previous_loan_defaults_on_file`: (bool) Indicates if the applicant has a history of previous loan defaults.
* `loan_status`: (int64) The target variable indicating loan approval status (0 = not approved, 1 = approved).

This dataset offers a rich set of features, encompassing demographic information, financial details, and credit history, making it suitable for developing a robust loan approval prediction model.



### Task

This project aims to build a predictive model to classify loan applications as either approved or not approved (loan_status). To achieve this, we will leverage a variety of features within the dataset, including the applicant's age (person_age), gender (person_gender), education level (person_education), income (person_income), employment experience (person_emp_exp), home ownership status (person_home_ownership), the requested loan amount (loan_amnt), the stated purpose of the loan (loan_intent), the loan's interest rate (loan_int_rate), the loan amount as a percentage of income (loan_percent_income), the length of their credit history (cb_person_cred_hist_length), their credit score (credit_score), and whether they have a history of previous loan defaults (previous_loan_defaults_on_file). By analyzing these features, the classification models will learn the underlying patterns that determine loan approval, enabling us to predict the loan_status for new applications.


## Automated ML

### AutoML Configuration

| Parameter                     | Value                   | Description                                                                 |
|-------------------------------|-------------------------|-----------------------------------------------------------------------------|
| `experiment_timeout_minutes`  | 20                      | Maximum time in minutes that the experiment should run.                       |
| `max_concurrent_iterations` | 5                       | Maximum number of iterations to run in parallel.                              |
| `primary_metric`            | `AUC_weighted`          | The primary metric to optimize for during model selection.                   |
| `task`                        | `"classification"`        | The type of machine learning task.                                          |
| `label_column_name`         | `"loan_status"`         | The name of the column containing the target variable.                        |
| `enable_early_stopping`     | `True`                  | Flag to enable early termination of poorly performing runs.                   |
| `featurization`               | `'auto'`                | Specifies that featurization should be performed automatically.               |             |


### Results
The Automated ML run's best model was a voting ensemble that included several pipelines. Among them were pipelines featuring a StandardScalerWrapper with an XGBoostClassifier, a MaxAbsScaler with LightGBM, and a StandardScalerWrapper with a RandomForestClassifier. This ensemble approach yielded a strong performance with an AUC score of 0.94612.

![Best Model](images\bestautomlrunnotebook.png) Best AutoML Run

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
