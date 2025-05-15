import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim

# Sample data for visualization
homework_results = {'Topic': ['A', 'B', 'C'], 'Score': [85, 90, 78]}
recommendations = {'Topic': ['A', 'B', 'C'], 'Recommendation': [4.5, 4.7, 4.2]}

# Create DataFrames
df_homework = pd.DataFrame(homework_results)
df_recommendations = pd.DataFrame(recommendations)

# Merge DataFrames
df_combined = pd.merge(df_homework, df_recommendations, on='Topic')

# Visualization
plt.figure(figsize=(10, 6))
sns.barplot(x='Topic', y='Score', data=df_combined, color='blue', label='Score')
sns.lineplot(x='Topic', y='Recommendation', data=df_combined, color='red', marker='o', label='Recommendation')

plt.title('Homework Scores and Recommendations')
plt.xlabel('Topic')
plt.ylabel('Values')
plt.legend()
plt.show()

# Example hybrid model in PyTorch
class HybridModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, lstm_hidden_dim, feature_dim, output_dim):
        super(HybridModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, lstm_hidden_dim, batch_first=True)
        self.fc1 = nn.Linear(feature_dim, 64)
        self.fc2 = nn.Linear(lstm_hidden_dim + 64, output_dim)

    def forward(self, text, features):
        embedded_text = self.embedding(text)
        lstm_out, _ = self.lstm(embedded_text)
        lstm_out = lstm_out[:, -1, :]  # Get the last output of the LSTM
        dense_features = torch.relu(self.fc1(features))
        combined = torch.cat((lstm_out, dense_features), dim=1)
        output = torch.sigmoid(self.fc2(combined))
        return output

# Hyperparameters
vocab_size = 10000
embedding_dim = 128
lstm_hidden_dim = 64
feature_dim = 10
output_dim = 1

# Initialize model, loss function, and optimizer
model = HybridModel(vocab_size, embedding_dim, lstm_hidden_dim, feature_dim, output_dim)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Summary of the model
print(model)