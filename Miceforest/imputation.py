import pandas as pd 
import miceforest as mf
from miceforest import mean_match_default
import os
import matplotlib.pyplot as plt

# Load the dataset
db_signs = pd.read_csv('/HomeGroup/csantos_cps/sepsis/Dataset_to_impute/ssinflag_eICU_12_36.csv')

labs = db_signs[['Cefalosporine', 'Macrolide', 'Meropenem', 'Metronidazole',
       'Penicillin', 'Quinolone', 'Vancomycin', 'gcs_min', 'gcs_motor',
       'gcs_verbal', 'gcs_eyes', 'age', 'sex', 'race', 'los_icu',
       'bicarbonate_min', 'bicarbonate_max',
       'creatinine_min', 'creatinine_max', 'chloride_min', 'chloride_max',
       'glucose_min', 'glucose_max', 'hematocrit_min', 'hematocrit_max',
       'hemoglobin_min', 'hemoglobin_max',
       'platelet_min', 'platelet_max', 'potassium_min', 'potassium_max',
       'sodium_min', 'sodium_max', 'bun_min', 'bun_max', 'wbc_min', 'wbc_max',
       'calcium_min', 'calcium_max', 'sofa', 'urineoutput', 'norepinephrine',
       'epinephrine', 'dopamine', 'dobutamine', 'phenylephrine', 'vasopressin',
        'temperature_max', 'temperature_min', 'temperature_mean',
       'spo2_max', 'spo2_min', 'spo2_mean', 'heart_rate_max', 'heart_rate_min',
       'heart_rate_mean', 'resp_rate_max', 'resp_rate_min', 'resp_rate_mean',
       'sbp_max', 'sbp_min', 'sbp_mean', 'dbp_max', 'dbp_min', 'dbp_mean',
       'mbp_max', 'mbp_min', 'mbp_mean']]

# Define the imputation kernels
kds = mf.ImputationKernel(
    labs,  # The dataset with missing data
    datasets=1,  
    train_nonmissing=False,
    save_all_iterations=True,
    imputation_order='ascending',
    random_state=1991,
    mean_match_scheme=mean_match_default)

# Run the MICE algorithm 
kds.mice(iterations=10, compile_candidates=True, verbose=True, save_models=1)

# Get the optimal parameters and losses
optimal_parameters, losses = kds.tune_parameters(
  dataset=0, #chose a dataset to tune the parameters
  nfold=5, # Number of folds for cross-validation
  optimization_steps=5 # Number of optimization steps	    
  )

# Run the MICE algorithm for 1 iteration with the optimal parameters
kds.mice(iterations=1, variable_parameters=optimal_parameters, verbose=True, save_models=1)

# Plot the imputed distributions
kds.plot_imputed_distributions(wspace=1, hspace=1.5)
plt.gcf().set_size_inches(12,14)
plt.savefig(f'{'Figures'}/imputed_distributions_e1236.pdf')
plt.close()  

# Plot the feature importance
kds.plot_feature_importance(dataset=0, annot=True, cmap="YlGnBu", vmin=0, vmax=1)
plt.gcf().set_size_inches(150,154)
plt.savefig(f'{'Figures'}/feature_importance_e1236.pdf')
plt.close()

# Plot the mean convergence
kds.plot_mean_convergence(wspace=0.9, hspace=1)
plt.gcf().set_size_inches(12, 14)
plt.savefig(f'{'Figures'}/mean_convergence_e1236.pdf')
plt.close()

#  Get the completed data
data_complete = kds.complete_data(inplace=False) 
# Save the completed data
data_complete.to_csv('Data_complete/eicu1236_complete.csv', index=False)
