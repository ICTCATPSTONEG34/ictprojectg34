import torch
from transformers import BertTokenizer, BertForSequenceClassification


model_path = 'bert_local_model'


tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)


model.eval()

def classify_text(input_text):
    try:
        
        input_ids = tokenizer.encode(input_text, add_special_tokens=True)

        
        input_ids = torch.tensor(input_ids).unsqueeze(0)  # Add batch dimension

        
        with torch.no_grad():
            outputs = model(input_ids)

        
        logits = outputs.logits

        
        probs = torch.softmax(logits, dim=1)

        
        predicted_class = torch.argmax(probs, dim=1).item()

        
        class_labels = model.config.id2label

        
        predicted_label = class_labels[predicted_class]

        if predicted_label == 'LABEL_0':
            predicted_label = 'Access control'
        elif predicted_label == 'LABEL_1':
            predicted_label = 'Accountability'
        elif predicted_label == 'LABEL_2':
            predicted_label = 'Availability'
        elif predicted_label == 'LABEL_3':
            predicted_label = 'Confidentiality'
        elif predicted_label == 'LABEL_4':
            predicted_label = 'Functional'
        elif predicted_label == 'LABEL_5':
            predicted_label = 'Integrity'
        elif predicted_label == 'LABEL_6':
            predicted_label = 'Operational'
        else:
            predicted_label = 'Other'

        
        return predicted_label, probs
    except Exception as e:
        
        error_message = str(e)
        return error_message, None
