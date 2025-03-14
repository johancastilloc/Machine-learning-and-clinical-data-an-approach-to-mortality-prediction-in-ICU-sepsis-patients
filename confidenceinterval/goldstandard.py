import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import os

current_dir = os.path.abspath(os.getcwd())
sys.path.append(os.path.abspath(os.path.join(current_dir, '..')))
def confusion_matrix_2(th, truths_val, preds_val, truths_test, preds_test,title1,title2):
    # Convert predicted probabilities into binary classes based on the selected threshold
    predicted_classes_val = (preds_val >= th).astype(int)
    
    # Calculate TP, FP, FN
    TP = np.sum((truths_val == 1) & (predicted_classes_val == 1))  # True Positives
    FP = np.sum((truths_val == 0) & (predicted_classes_val == 1))  # False Positives
    FN = np.sum((truths_val == 1) & (predicted_classes_val == 0))  # False Negatives
    
    # Calculate metrics
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Print results
    print("Validation results:")
    print(f"Threshold: {th:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Calculate confusion matrix
    cm = confusion_matrix(truths_val, predicted_classes_val)

    # Convert the confusion matrix to percentages
    cm_percentage = cm.astype('float') / cm.sum() * 100
    
    # Create the annotation text with the percentage symbol
    annotations = np.array([["{:.2f}%".format(value) for value in row] for row in cm_percentage])
    # Plot the confusion matrix with percentages
    plt.figure(figsize=(7, 5))  # Set figure size
    sns.heatmap(cm_percentage, annot=annotations, fmt='', cmap='Blues', cbar=True,  # Display the percentages with two decimal points
                xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])  # Set class labels
    plt.ylabel('True label')  # Label for the y-axis
    plt.xlabel('Predicted label')  # Label for the x-axis
    plt.title(f'{title1}')  # Title of the plot
    plt.savefig(f"imagenes/{title1}.pdf", format="pdf", bbox_inches="tight")
    plt.show()  # Display the plot
    #Test
    # Convert predicted probabilities into binary classes based on the selected threshold
    predicted_classes_test = (preds_test >= th).astype(int)
    
    # Calculate TP, FP, FN
    TP2 = np.sum((truths_test == 1) & (predicted_classes_test == 1))  # True Positives
    FP2 = np.sum((truths_test == 0) & (predicted_classes_test == 1))  # False Positives
    FN2 = np.sum((truths_test == 1) & (predicted_classes_test == 0))  # False Negatives

    # Calculate metrics
    precision2 = TP2 / (TP2 + FP2) if (TP2 + FP2) > 0 else 0
    recall2 = TP2 / (TP2 + FN2) if (TP2 + FN2) > 0 else 0
    f12 = 2 * (precision2 * recall2) / (precision2 + recall2) if (precision2 + recall2) > 0 else 0
    
    # Print results
    print("Test results:")
    print(f"With the validation threshold: {th:.4f}")
    print(f"Precision: {precision2:.4f}")
    print(f"Recall: {recall2:.4f}")
    print(f"F1 Score: {f12:.4f}")
    
    # Calculate confusion matrix
    cm2= confusion_matrix(truths_test, predicted_classes_test)

    # Convert the confusion matrix to percentages
    cm_percentage2 = cm2.astype('float') / cm2.sum() * 100
    
    # Create the annotation text with the percentage symbol
    annotations2 = np.array([["{:.2f}%".format(value) for value in row] for row in cm_percentage2])
    # Plot the confusion matrix with percentages
    plt.figure(figsize=(7, 5))  # Set figure size
    sns.heatmap(cm_percentage2, annot=annotations2, fmt='', cmap='Blues', cbar=True,  # Display the percentages with two decimal points
                xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])  # Set class labels
    plt.ylabel('True label')  # Label for the y-axis
    plt.xlabel('Predicted label')  # Label for the x-axis
    plt.title(f'{title2}')  # Title of the plot
    plt.savefig(f"imagenes/{title2}.pdf", format="pdf", bbox_inches="tight")
    plt.show()  # Display the plot
    
def evaluation_metrics(desired_fpr, fpr, th, truths_val, preds_val, truths_test, preds_test,title1,title2):
    # Ensure truths and preds are NumPy arrays
    truths_val = np.array(truths_val)
    preds_val = np.array(preds_val)
    truths_test = np.array(truths_test)
    preds_test = np.array(preds_test)
    
    # Find the index of the FPR value that is closest to the desired FPR
    index = np.abs(fpr - desired_fpr).argmin()
    
    # Retrieve the threshold corresponding to the closest FPR
    threshold_fpr = th[index]
    confusion_matrix_2(threshold_fpr, truths_val, preds_val, truths_test, preds_test,title1,title2)

    
def max_th_f1(th, truths_val, preds_val,truths_test, preds_test,title1,title2,title3): 
    # Ensure truths and preds are NumPy arrays
    truths_val = np.array(truths_val)
    preds_val = np.array(preds_val)
    truths_test = np.array(truths_test)
    preds_test = np.array(preds_test)    
# Generate a range of thresholds from 0 to 1
    thresholds_range = np.arange(0, 1, 0.0001)
    f1_scores = []

    # Initialize variables to find the threshold that maximizes F1 Score
    max_f1 = 0
    optimal_threshold = 0

    for th in thresholds_range:
        # Predict classes using the current threshold
        predicted_classes_val = (preds_val >= th).astype(int)
        
        # Calculate TP, FP, FN
        TP = np.sum((truths_val == 1) & (predicted_classes_val == 1))  # True Positives
        FP = np.sum((truths_val == 0) & (predicted_classes_val == 1))  # False Positives
        FN = np.sum((truths_val == 1) & (predicted_classes_val == 0))  # False Negatives
        
        # Calculate precision and recall
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        
        # Calculate F1 Score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Store F1 Score
        f1_scores.append(f1)
        
        # Update the maximum F1 Score and the optimal threshold
        if f1 > max_f1:
            max_f1 = f1
            optimal_threshold = th

    # Print the optimal threshold and corresponding F1 Score
    print(f"Optimal Threshold for Maximum F1 Score: {optimal_threshold:.4f}")
    print(f"Maximum F1 Score: {max_f1:.4f}")
    
    # Plot F1 Score vs Threshold
    plt.figure(figsize=(7, 5))
    plt.plot(thresholds_range, f1_scores, color='purple', label='F1 Score')  
    plt.axvline(optimal_threshold, color='red', linestyle='--', label='Optimal Threshold')  # Included optimal threshold value in the legend
    plt.plot(optimal_threshold, max_f1, 'r*', markersize=10,label = f'{optimal_threshold:.4f}')
    plt.title(f'{title3}')
    plt.xlabel('Threshold')
    plt.ylabel('F1 Score')
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, which='major', linestyle='-', linewidth='0.5')
    plt.minorticks_on()
    plt.grid(True, which='minor', linestyle=':', linewidth='0.5')
    plt.savefig(f"imagenes/{title3}.pdf", format="pdf", bbox_inches="tight")
    plt.show()
    
    confusion_matrix_2(optimal_threshold, truths_val, preds_val, truths_test, preds_test,title1,title2)
    