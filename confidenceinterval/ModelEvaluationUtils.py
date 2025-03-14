import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import sys
import os
from scipy.stats import norm

current_dir = os.path.abspath(os.getcwd())
sys.path.append(os.path.abspath(os.path.join(current_dir, '..')))

def select_data(data_type, n, path_train, path_test):
    """
    Selects the first n columns based on the specified data type from the training and test datasets.
    
    This function loads the training and test datasets from Excel files, prepares the features (X) and targets (y),
    and selects columns based on the specified data type ('A' or 'B'). It ensures that no more columns than available
    are selected and returns the filtered datasets for training and testing.
    
    Parameters:
    - data_type (str): Specifies the type of data to select. 'A' for a comprehensive set of features, 'B' for a focused set.
    - n (int): The number of columns to select from the specified data type.
    
    Returns:
    - Tuple: (x_train, x_test, y_train, y_test), where:
        - x_train: The training dataset with the first n columns based on data_type.
        - x_test: The test dataset with the first n columns based on data_type.
        - y_train: The target variable for the training dataset.
        - y_test: The target variable for the test dataset.
    """
    # Load the datasets
    #train_db = pd.read_excel('datasets/mimic_train.xlsx')
    train_db = pd.read_csv(path_train)
    test_db = pd.read_csv(path_test)   
    #test_db = pd.read_excel('datasets/eicu_test.xlsx') 
    
    # Prepare X and y for training and testing
    X = train_db.drop('hospital_expire_flag', axis=1)
    y_train = train_db['hospital_expire_flag']
    x_test_ = test_db.drop('hospital_expire_flag', axis=1)
    y_test = test_db['hospital_expire_flag']
     
    if data_type == 'johan_variables':
        columns = ['sofa', 'urineoutput', 'vasopressin', 'phenylephrine', 'dobutamine', 'dopamine',
                   'epinephrine', 'norepinephrine', 'gcs_eyes', 'gcs_verbal', 'gcs_motor', 'gcs_min',
                   'Vancomycin', 'Quinolone', 'Penicillin', 'Metronidazole', 'Meropenem', 'Macrolide',
                   'Cefalosporine', 'calcium_max', 'calcium_min', 'wbc_max', 'wbc_min', 'bun_max', 'bun_min',
                   'sodium_max', 'sodium_min', 'potassium_max', 'potassium_min', 'platelet_max', 'platelet_min',
                    'hemoglobin_max', 'hemoglobin_min', 'hematocrit_max',
                   'hematocrit_min', 'glucose_max', 'glucose_min', 'chloride_max', 'chloride_min',
                   'creatinine_max', 'creatinine_min', 'bicarbonate_max', 
                   'bicarbonate_min', 'mbp_mean', 
                   'mbp_min', 'mbp_max', 'dbp_mean', 'dbp_min', 'dbp_max', 'sbp_mean', 'sbp_min', 'sbp_max', 
                   'resp_rate_mean', 'resp_rate_min', 'resp_rate_max', 'heart_rate_mean', 'heart_rate_min', 
                   'heart_rate_max', 'spo2_mean', 'spo2_min', 'spo2_max', 'temperature_mean', 'temperature_min', 
                   'temperature_max', 'los_icu', 'race', 'sex', 'age'] 
         
    elif data_type == 'FCV_variables':
        columns = ['heart_rate_min', 'heart_rate_max', 'heart_rate_mean', 'sbp_min', 'sbp_max', 
                   'sbp_mean', 'dbp_min', 'dbp_max', 'dbp_mean', 'mbp_min', 'mbp_max', 'mbp_mean', 
                   'resp_rate_min', 'resp_rate_max', 'resp_rate_mean', 'temperature_min', 
                   'temperature_max', 'temperature_mean', 'spo2_min', 'spo2_max', 'spo2_mean', 'age', 'los_icu', 
                   'race', 'sex']     
        
    elif data_type == 'Subjective_variables':
        columns = ['gcs_motor', 'gcs_eyes', 'gcs_verbal', 'gcs_min']
        
    elif data_type == 'Non_Subjective_variables':
        columns = ['Cefalosporine', 'Macrolide', 'Meropenem', 'Metronidazole', 'Penicillin', 'Quinolone', 'Vancomycin',
                   'dopamine', 'dobutamine', 'epinephrine', 'milrinone', 'norepinephrine', 'phenylephrine', 'vasopressin',
                   'lactate_min', 'lactate_max', 'hematocrit_min', 'hematocrit_max', 'hemoglobin_min', 'hemoglobin_max',
                   'platelets_min', 'platelets_max', 'wbc_min', 'wbc_max', 'albumin_min', 'albumin_max', 'aniongap_min',
                   'aniongap_max', 'bicarbonate_min', 'bicarbonate_max', 'bun_min', 'bun_max', 'calcium_min', 'calcium_max',
                   'chloride_min', 'chloride_max', 'creatinine_min', 'creatinine_max', 'glucose_min', 'glucose_max',
                   'sodium_min', 'sodium_max', 'potassium_min', 'potassium_max', 'bilirubin_min', 'bilirubin_max','urineoutput',
                   'heart_rate_min', 'heart_rate_max', 'heart_rate_mean', 'sbp_min', 'sbp_max', 'sbp_mean','dbp_min', 'dbp_max',
                   'dbp_mean', 'mbp_min', 'mbp_max', 'mbp_mean', 'resp_rate_min','resp_rate_max', 'resp_rate_mean', 
                   'temperature_min', 'temperature_max', 'temperature_mean','spo2_min', 'spo2_max', 'spo2_mean','age', 'los_icu',
                   'race', 'sex'] 
       
    elif data_type == 'Therapeutic_variables':
        columns = ['Cefalosporine', 'Macrolide', 'Meropenem', 'Metronidazole', 'Penicillin', 'Quinolone', 'Vancomycin',
                   'dopamine', 'dobutamine', 'epinephrine', 'milrinone', 'norepinephrine', 'phenylephrine', 'vasopressin']
        
    elif data_type == 'Laboratory_variables':
        columns = ['lactate_min', 'lactate_max', 'hematocrit_min', 'hematocrit_max', 'hemoglobin_min', 'hemoglobin_max',
                   'platelets_min', 'platelets_max', 'wbc_min', 'wbc_max', 'albumin_min', 'albumin_max', 'aniongap_min',
                   'aniongap_max', 'bicarbonate_min', 'bicarbonate_max', 'bun_min', 'bun_max', 'calcium_min', 'calcium_max',
                   'chloride_min', 'chloride_max', 'creatinine_min', 'creatinine_max', 'glucose_min', 'glucose_max',
                   'sodium_min', 'sodium_max', 'potassium_min', 'potassium_max', 'bilirubin_min', 'bilirubin_max','urineoutput']

    elif data_type == 'Vital_signs_variables':
        columns = ['heart_rate_min', 'heart_rate_max', 'heart_rate_mean', 'sbp_min', 'sbp_max', 'sbp_mean','dbp_min', 'dbp_max',
                   'dbp_mean', 'mbp_min', 'mbp_max', 'mbp_mean', 'resp_rate_min','resp_rate_max', 'resp_rate_mean', 
                   'temperature_min', 'temperature_max', 'temperature_mean','spo2_min', 'spo2_max', 'spo2_mean']   
          
    elif data_type == 'Demographics_variables':
        columns = ['age', 'los_icu', 'race', 'sex']

    elif data_type == 'VitalSigns_Demographics':
        columns = ['age', 'los_icu', 'race', 'sex', 'heart_rate_min', 'heart_rate_max', 'heart_rate_mean', 'sbp_min', 'sbp_max', 
                   'sbp_mean','dbp_min', 'dbp_max', 'dbp_mean', 'mbp_min', 'mbp_max', 'mbp_mean', 'resp_rate_min','resp_rate_max',
                   'resp_rate_mean', 'temperature_min', 'temperature_max', 'temperature_mean','spo2_min', 'spo2_max', 'spo2_mean' ]     
        
    elif data_type == 'VitalSigns_Laboratory':
        columns = ['heart_rate_min', 'heart_rate_max', 'heart_rate_mean', 'sbp_min', 'sbp_max', 'sbp_mean','dbp_min', 'dbp_max',
                   'dbp_mean', 'mbp_min', 'mbp_max', 'mbp_mean', 'resp_rate_min','resp_rate_max', 'resp_rate_mean', 
                   'temperature_min', 'temperature_max', 'temperature_mean','spo2_min', 'spo2_max', 'spo2_mean', 'lactate_min', 
                   'lactate_max', 'hematocrit_min', 'hematocrit_max', 'hemoglobin_min', 'hemoglobin_max', 'platelets_min', 
                   'platelets_max', 'wbc_min', 'wbc_max', 'albumin_min', 'albumin_max', 'aniongap_min', 'aniongap_max', 
                   'bicarbonate_min', 'bicarbonate_max', 'bun_min', 'bun_max', 'calcium_min', 'calcium_max', 'chloride_min', 
                   'chloride_max', 'creatinine_min', 'creatinine_max', 'glucose_min', 'glucose_max', 'sodium_min', 'sodium_max', 
                   'potassium_min', 'potassium_max', 'bilirubin_min', 'bilirubin_max','urineoutput']   
        
    elif data_type == 'VitalSigns_Therapeutic':
        columns = ['heart_rate_min', 'heart_rate_max', 'heart_rate_mean', 'sbp_min', 'sbp_max', 'sbp_mean','dbp_min', 'dbp_max',
                   'dbp_mean', 'mbp_min', 'mbp_max', 'mbp_mean', 'resp_rate_min','resp_rate_max', 'resp_rate_mean', 
                   'temperature_min', 'temperature_max', 'temperature_mean','spo2_min', 'spo2_max', 'spo2_mean', 'Cefalosporine',
                   'Macrolide', 'Meropenem', 'Metronidazole', 'Penicillin', 'Quinolone', 'Vancomycin', 'dopamine', 'dobutamine', 
                   'epinephrine', 'milrinone', 'norepinephrine', 'phenylephrine', 'vasopressin']   
    elif data_type == 'SOFA':
        columns = ['sofa']
  
    # Ensure not to select more columns than available
    n_columns = min(n, len(columns))
    selected_columns = columns[:n_columns]
    
    # Select the first n columns from the filtered datasets
    x_train = X[selected_columns]
    x_test = x_test_[selected_columns]
    
    return x_train, x_test, y_train, y_test

def z_score(x_train, x_test):
    """
    Scales the training and test datasets using Z-score normalization.
    
    Parameters:
    - x_train: Training dataset (pandas DataFrame).
    - x_test: Test dataset (pandas DataFrame).
    
    Returns:
    - Tuple of DataFrames: (scaled training dataset, scaled test dataset).
    """
    # Initialize the StandardScaler
    scaler = StandardScaler()
    
    # Fit the scaler using only the training data
    scaler.fit(x_train)
    
    # Transform both training and test data with the fitted scaler
    x_train_scaled = scaler.transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    
    # Convert the scaled arrays back to pandas DataFrames with the original column names
    x_train_scaled_df = pd.DataFrame(x_train_scaled, columns=x_train.columns)
    x_test_scaled_df = pd.DataFrame(x_test_scaled, columns=x_test.columns)
    
    return x_train_scaled_df, x_test_scaled_df

def train_and_evaluate_models(x_train, y_train, x_test, y_test, skf, models_list):
    """
    Trains multiple models and collects predictions for validation and test sets using cross-validation.
    
    Parameters:
    - x_train: Training features (pandas DataFrame).
    - y_train: Training target (pandas Series).
    - x_test: Test features (pandas DataFrame).
    - y_test: Test target (pandas Series).
    - skf: Stratified K-Folds cross-validator.
    - models_list: List of ML models to train and evaluate.
    
    Returns:
    - Dictionary containing validation and test predictions and truths for each model.
    """
    results = {}
    
    # Loop over each model in the list
    for model in models_list:
        model_name = type(model).__name__
        print(f"Training and evaluating model: {model_name}")
        
        val_preds, val_truths, test_preds, test_truths = [], [], [], []
        
        for i, (train_index, val_index) in enumerate(skf.split(x_train, y_train), 1):
            # Splitting the data into training and validation folds
            x_train_fold, x_val_fold = x_train.iloc[train_index], x_train.iloc[val_index]
            y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]
            
            # Training the model on the training fold
            model.fit(x_train_fold, y_train_fold)
            
            # Predicting probabilities or class labels on the validation and test sets
            if hasattr(model, 'predict_proba'):
                val_probs = model.predict_proba(x_val_fold)[:, 1]
                test_probs = model.predict_proba(x_test)[:, 1]
            else:
                val_probs = model.predict(x_val_fold)
                test_probs = model.predict(x_test)
            
            # Print AUC if applicable
            if hasattr(model, 'predict_proba'):  # Only calculate AUC for models that support probability predictions
                train_probs = model.predict_proba(x_train_fold)[:, 1]
                train_auc = roc_auc_score(y_train_fold, train_probs)
                print(f"Fold {i}: Train AUC: {train_auc:.4f}")
                
                val_auc = roc_auc_score(y_val_fold, val_probs)
                print(f" Validation AUC: {val_auc:.4f}")
                
                test_auc = roc_auc_score(y_test, test_probs)
                print(f" Test AUC: {test_auc:.4f}")
            else:
                print(f"Fold {i}: Model doesn't support AUC calculation.")
            
            # Collecting predictions and truths for validation and test sets
            val_preds.extend(val_probs)
            val_truths.extend(y_val_fold)
            test_preds.extend(test_probs)
            test_truths.extend(y_test)
        
        # Save results for the current model
        results[model_name] = {
            'val_preds': val_preds,
            'val_truths': val_truths,
            'test_preds': test_preds,
            'test_truths': test_truths,
        }

    return results

def calculate_ci_models(results, truth_key, preds_key, alpha=0.95):
    """
    Calculate AUC and confidence intervals for multiple models.

    Parameters:
    - results: Dictionary containing model predictions and truths.
    - truth_key: Key for the true values in the results dictionary.
    - preds_key: Key for the predicted values in the results dictionary.
    - alpha: Confidence level (default 0.95 for 95% confidence interval).

    Returns:
    - List of tuples containing model name, AUC, and confidence interval.
    """
    auc_results = []
    z = norm.ppf(1 - (1 - alpha) / 2)  # Critical value for the desired confidence level
    for model_name, data in results.items():
        truths = data[truth_key]
        preds = data[preds_key]
        
        # Calculate AUC
        auc = roc_auc_score(truths, preds)

        # Number of samples
        n = len(preds)
        
        # Calculate the variance and the standard error
        variance = auc * (1 - auc) / n
        se = np.sqrt(variance)
        
        # Calculate the margin of error
        margin_of_error = z * se
        
        # Calculate the confidence interval
        ci_lower = auc - margin_of_error
        ci_upper = auc + margin_of_error
        

        auc_results.append((model_name, auc, (ci_lower, ci_upper)))
        
    return auc_results

def plot_roc_auc(val_truths, val_preds, test_truths, test_preds, low_auc, upper_auc, low_auc_test, upper_auc_test,title):
    """
    Plots ROC AUC values.
    """
    fpr_val, tpr_val, th_val = roc_curve(val_truths, val_preds)
    auc_val = roc_auc_score(val_truths, val_preds)
    fpr_test, tpr_test, th_test = roc_curve(test_truths, test_preds)
    auc_test = roc_auc_score(test_truths, test_preds)

    # Plot ROC curves
    plt.figure(figsize=(7, 5))
    plt.plot(fpr_val, tpr_val, label=f'MIMIC AUC {auc_val:.2f} (95% CI {low_auc:.2f}-{upper_auc:.2f})')
    plt.plot(fpr_test, tpr_test, label=f'eICU-CRD AUC {auc_test:.2f} (95% CI {low_auc_test:.2f}-{upper_auc_test:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{title}')
    plt.legend(loc='lower right')
    plt.grid(True, which='major', linestyle='-', linewidth='0.5')
    plt.minorticks_on()
    plt.grid(True, which='minor', linestyle=':', linewidth='0.5')
    plt.savefig(f"imagenes/{title}.pdf", format="pdf", bbox_inches="tight")
    plt.show()

    return fpr_val, tpr_val, th_val, fpr_test, tpr_test, th_test
    
def plot_roc_auc_models(results, truth_key, preds_key, auc_results,title):
    """
    Plots ROC AUC values for different models.

    Parameters:
        results (dict): Dictionary containing model names as keys and 
                        another dictionary with 'val_preds' and 'val_truths'.
        truth_key (str): The key to access the true labels in results (e.g., 'val_truths' or 'test_truths').
        preds_key (str): The key to access the predicted probabilities in results (e.g., 'val_preds' or 'test_preds').
        auc_results (list): List of tuples containing model name, AUC, and confidence intervals.
    """
    plt.figure(figsize=(7, 7))

    # Loop through each model and plot the ROC curve
    for model_name, auc_score, (lower_ci, upper_ci) in auc_results:
        # Extract truth and predicted probabilities for the current model
        truth = results[model_name][truth_key]  # Actual labels
        preds = results[model_name][preds_key]    # Predicted probabilities

        # Calculate FPR and TPR for the current model
        fpr, tpr, _ = roc_curve(truth, preds)
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.2f} (95% CI {lower_ci:.2f} - {upper_ci:.2f}))')

    # Plot the diagonal line representing a random classifier
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')

    # Labels and title
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{title}')
    plt.legend(loc='lower right')
    plt.grid(True, which='major', linestyle='-', linewidth='0.5')
    plt.minorticks_on()
    plt.grid(True, which='minor', linestyle=':', linewidth='0.5')
    plt.axis('square')  # Ensures x and y axes are on the same scale
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.savefig(f"imagenes/{title}.pdf", format="pdf", bbox_inches="tight")
    # Show the plot
    plt.show()

def predict_mortality(sofa_score):
    if sofa_score <= 1:
        return 0.0  # No risk of mortality
    elif 2 <= sofa_score <= 3:
        return 0.015  # Adjust based on mortality in higher score
    elif 4 <= sofa_score <= 5:
        return 0.067  # Mortality probability for score 4-5
    elif 6 <= sofa_score <= 7:
        return 0.182  # Mortality probability for score 6-7
    elif 8 <= sofa_score <= 9:
        return 0.263  # Mortality probability for score 8-9
    elif 10 <= sofa_score <= 11:
        return 0.458  # Mortality probability for score 10-11
    elif 12 <= sofa_score <= 14:
        return 0.80   # Mortality probability for score 12-14
    else:  # sofa_score > 14
        return 0.897  # High risk of mortality

def train_and_evaluate_sofa(x_train, y_train, x_test, y_test):
    """
    Train and evaluate the rule-based SOFA mortality prediction model.
    
    Parameters:
    - x_train: Training features (pandas DataFrame).
    - y_train: Training target (pandas Series).
    - x_test: Test features (pandas DataFrame).
    - y_test: Test target (pandas Series).

    
    Returns:
    - Tuple of lists: (validation predictions, validation truths, test predictions, test truths).
    """
    
    # Prepare DataFrames for training and testing
    data_train = pd.DataFrame({
        'sofa': x_train['sofa'],  
        'y_true': y_train 
    })
    
    data_test = pd.DataFrame({
        'sofa': x_test['sofa'], 
        'y_true': y_test 
    })

    # Apply the rule-based model to get predicted probabilities for training data
    data_train['y_pred'] = data_train['sofa'].apply(predict_mortality)

    # Apply the rule-based model to get predicted probabilities for testing data
    data_test['y_pred'] = data_test['sofa'].apply(predict_mortality)

    # Return the predictions and true values
    results = {}
    results['SOFA'] = { 
        'val_preds': data_train['y_pred'],
        'val_truths': data_train['y_true'],
        'test_preds': data_test['y_pred'],
        'test_truths': data_test['y_true'],    
    }
    return results

