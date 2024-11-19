import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

num_classes = 5
model = models.resnet50(pretrained=False)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 512),
    nn.BatchNorm1d(512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, num_classes)
)
model.load_state_dict(torch.load('models/enhanced_resnet50_cassava.pth', map_location='cpu'))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

class_names = ['cbb', 'cbsd', 'cgm', 'cmd', 'healthy']

full_class_names = {
    'cbb': 'Cassava Bacterial Blight',
    'cbsd': 'Cassava Brown Streak Disease',
    'cgm': 'Cassava Green Mottle',
    'cmd': 'Cassava Mosaic Disease',
    'healthy': 'Healthy'
}

st.title('Plant Disease Detection')

uploaded_file = st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption='Uploaded Image', use_column_width=True)
    except Exception as e:
        st.error(f"Error loading image: {e}")
        st.stop()
    
    input_tensor = transform(image).unsqueeze(0)
    
    with st.spinner('Making prediction...'):
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            prob_values, pred = torch.max(probabilities, 0)
    st.success('Prediction completed!')
    
    predicted_class_abbr = class_names[pred.item()]
    predicted_class_full = full_class_names.get(predicted_class_abbr, "Unknown")
    
    st.write(f'**Predicted Class:** {predicted_class_abbr} - {predicted_class_full} ({prob_values.item()*100:.2f}%)')
    
    prob_dict = {class_names[i]: float(probabilities[i]) for i in range(num_classes)}
    prob_df = pd.DataFrame(list(prob_dict.items()), columns=['Class', 'Probability'])
    
    prob_df['Full Class Name'] = prob_df['Class'].map(full_class_names)
    
    st.write("**Class Probabilities:**")
    st.dataframe(prob_df[['Class', 'Full Class Name', 'Probability']].style.format({'Probability': "{:.2%}"}))
    
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x='Probability', y='Full Class Name', data=prob_df, palette='viridis', ax=ax)
    ax.set_xlim(0, 1)
    ax.set_xlabel('Probability')
    ax.set_ylabel('Class')
    ax.set_title('Class Probabilities')
    st.pyplot(fig)
