import os
import streamlit as st  
import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
import os
from datetime import datetime
import torch.nn as nn
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")

device = "cpu"
labels = ["false", "effect", "mechanism", "advise", "int"] 

class BertSentimentClassifier(nn.Module):
    def __init__(self, bert_model_name, num_classes):
        super(BertSentimentClassifier, self).__init__()

        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout1 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(self.bert.config.hidden_size, 256)
        self.dropout2 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(256, num_classes)


    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout1(pooled_output)
        output = self.fc1(pooled_output)
        output = self.dropout2(output)
        logits = self.fc2(output)

        return logits

def load_bert_model(weight_path, bert_model_name):
  
    num_classes = 6
    model_predict = BertSentimentClassifier(bert_model_name, num_classes)
    model_predict.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu')))
    tokenizer = BertTokenizer.from_pretrained(bert_model_name) 
    return model_predict, tokenizer

def inference():
    st.image("image/4_inference.png")
    st.subheader("Step 1: Choose the BERT architecture and pretraining.", divider='rainbow') 
    choice_model = st.selectbox("Choose the BERT architecture and pretraining model.", 
    ["None","BERT Model 1", "BERT Model 2"]) 
    if choice_model == "BERT Model 1":
        st.image("image/Model1.png")
    elif choice_model == "BERT Model 2": 
        st.image("image/Model2.png")

    choice_pretraining_model = st.selectbox("Choose the BERT architecture and pretraining model.", 
    ["bert-base-uncased", "alvaroalon2/biobert_diseases_ner"]) 
    
    if choice_model == "BERT Model 1":
        if choice_pretraining_model == "bert-base-uncased":
            weight_path = "comming soon"
        elif choice_pretraining_model == "alvaroalon2/biobert_diseases_ner":
            weight_path = "Model_BERT_1_270/best_model_BERT1 (1).pt"
        max_len = 270 

    elif choice_model == "BERT Model 2": 
        if choice_pretraining_model == "bert-base-uncased":
            weight_path = "comming soon"
        elif choice_pretraining_model == "alvaroalon2/biobert_diseases_ner":
            weight_path = "Model_BERT_2/best_model_BERT2.pt"
        max_len = 30


    st.subheader("Step 2: Input the sentence.", divider='rainbow') 

    col1, col2 = st.columns(2)  # Using beta_columns to create two columns

    with col1:
        e1 = st.text_input("Enter Entity e1:")

    with col2:
        e2 = st.text_input("Enter Entity e2:")
    
    if choice_model == "BERT Model 1":
        full_sentence = st.text_input("Enter Full Sentence:") 
        input_text = f"{e1} [SEP] {e2} [SEP] {full_sentence}"
    else: 
        input_text = f"{e1} [SEP] {e2}"
    
    st.subheader("Step 3: Output the result.", divider='rainbow') 
    if st.button("Run"): 
        bert_model_name = choice_pretraining_model 
        
        model_predict, tokenizer = load_bert_model(weight_path, bert_model_name)  

        encoded_input = tokenizer.encode_plus(
            input_text, 
            add_special_tokens=True,
            truncation=True,
            max_length= max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        ).to(device)

        predict_id = torch.argmax(model_predict(**encoded_input), dim=1).item()
        print(labels[predict_id]) 
        st.success(f"Predicted Relation: {labels[predict_id]}")
        # Get the predicted label probabilities
        probs = torch.nn.functional.softmax(model_predict(**encoded_input), dim=1)[0].tolist()

        # Filter out class 5
        filtered_probs = [prob for i, prob in enumerate(probs) if i != 5]

        # Create a list of labels excluding class 5
        filtered_labels = labels

        # Plot the distribution using seaborn
        plt.figure(figsize=(8, 6))
        sns.barplot(x=filtered_labels, y=filtered_probs, palette='viridis')
        plt.xlabel('Labels')
        plt.ylabel('Probability')
        plt.title('Predicted Label Distribution')

        # Display percentages on top of each bar
        for i, prob in enumerate(filtered_probs):
            plt.text(i, prob + 0.01, f'{prob * 100:.2f}%', ha='center')
        st.pyplot(plt) 

        st.balloons()




