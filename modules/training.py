import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.localization import get_text
import time

def generate_shape_data(num_samples=200, size=28):
    data = []
    labels = []
    for _ in range(num_samples):
        img = np.zeros((size, size), dtype=np.float32)
        label = np.random.randint(0, 2)

        r = np.random.randint(5, 10)
        cx = np.random.randint(r, size - r)
        cy = np.random.randint(r, size - r)

        if label == 0:
            img[cy-r:cy+r, cx-r:cx+r] = 1.0
        else:
            y, x = np.ogrid[:size, :size]
            mask = (x - cx)**2 + (y - cy)**2 <= r**2
            img[mask] = 1.0

        noise = np.random.normal(0, 0.1, (size, size))
        img = img + noise
        img = np.clip(img, 0, 1)

        data.append(img)
        labels.append(label)

    return torch.tensor(np.array(data)).unsqueeze(1), torch.tensor(np.array(labels))

class TinyNet(nn.Module):
    def __init__(self):
        super(TinyNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, 3)
        self.fc1 = nn.Linear(16 * 5 * 5, 32)
        self.fc2 = nn.Linear(32, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def render(lang):
    st.header(get_text('module_training', lang))

    st.write(get_text('training_intro', lang))
    st.info(get_text('math_backprop', lang))
    st.latex(r"\theta_{new} = \theta_{old} - \eta \cdot \nabla_\theta J(\theta)")

    col1, col2 = st.columns(2)
    with col1:
        lr = st.select_slider(get_text('lr', lang), options=[0.0001, 0.001, 0.01, 0.1], value=0.001)
    with col2:
        epochs = st.slider(get_text('epochs', lang), 5, 50, 20)

    if st.button(get_text('start_training', lang)):
        X, y = generate_shape_data(num_samples=300)

        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        model = TinyNet()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        chart_loss = st.empty()
        chart_acc = st.empty()

        history = {'loss': [], 'acc': []}

        progress_bar = st.progress(0)

        st.write(get_text('training_progress', lang))

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                test_out = model(X_test)
                _, predicted = torch.max(test_out.data, 1)
                accuracy = (predicted == y_test).sum().item() / len(y_test)

            history['loss'].append(loss.item())
            history['acc'].append(accuracy)

            df_hist = pd.DataFrame(history)

            fig_loss = px.line(df_hist, y='loss', title="Loss Curve")
            chart_loss.plotly_chart(fig_loss, use_container_width=True)

            fig_acc = px.line(df_hist, y='acc', title="Accuracy Curve", range_y=[0, 1])
            chart_acc.plotly_chart(fig_acc, use_container_width=True)

            progress_bar.progress((epoch + 1) / epochs)
            time.sleep(0.1)

        st.success(get_text('training_finished', lang))

        st.subheader(get_text('evaluation', lang))
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test.numpy(), predicted.numpy())

        fig_cm = px.imshow(cm, text_auto=True,
                           labels=dict(x="Predicted", y="True", color="Count"),
                           x=['Square', 'Circle'], y=['Square', 'Circle'])
        st.plotly_chart(fig_cm)

        st.write(get_text('test_examples', lang))
        c1, c2, c3, c4 = st.columns(4)
        indices = np.random.choice(len(X_test), 4)
        cols = [c1, c2, c3, c4]
        for i, idx in enumerate(indices):
            img = X_test[idx][0].numpy()
            true_lbl = "Square" if y_test[idx]==0 else "Circle"
            pred_lbl = "Square" if predicted[idx]==0 else "Circle"
            color = "green" if true_lbl == pred_lbl else "red"

            with cols[i]:
                st.image(img, clamp=True, width=100)
                st.markdown(f"True: {true_lbl}<br>Pred: <span style='color:{color}'>{pred_lbl}</span>", unsafe_allow_html=True)
