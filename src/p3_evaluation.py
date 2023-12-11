import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def main():
    st.title("Training Classification Report")

    # Load data from the CSV file
    log_data = pd.read_csv("Model_BERT_2/training_log_BERT2_2023-12-07_02-51-28.txt")

    # Display the report
    display_report(log_data)

    # Plot training metrics
    plot_metrics(log_data)

def display_report(log_data):
    # Display raw log data
    st.subheader("Raw Log Data")
    st.dataframe(log_data)

def plot_metrics(log_data):
    st.subheader("Training Metrics Over Epochs")

    # Plot training loss
    if 'Epoch' in log_data.columns and 'Train_Loss' in log_data.columns:
        plt.figure(figsize=(10, 5))
        plt.plot(log_data['Epoch'], log_data['Train_Loss'], label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Epochs')
        plt.legend()
        st.pyplot()

    # Plot training precision
    if 'Epoch' in log_data.columns and 'Train_Precision' in log_data.columns:
        plt.figure(figsize=(10, 5))
        plt.plot(log_data['Epoch'], log_data['Train_Precision'], label='Training Precision')
        plt.xlabel('Epoch')
        plt.ylabel('Precision')
        plt.title('Training Precision Over Epochs')
        plt.legend()
        st.pyplot()





def evaluation():
    st.image("image/3_evaluation.png")
    main()
     

#Step 1: Metrics evaluation
#Step 2: Training and validation loss 
#Step 3: Training and validation accuracy 
#Step 4: Confusion matrix
#Step 5: Classification report
#Step 6: ROC curve
#Step 7: Precision-recall curve
