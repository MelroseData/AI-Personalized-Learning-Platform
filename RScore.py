import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from enviroment import RecommendationEnv
from nlp_model import NLPModel

class QNetwork(nn.Module):
    def __init__(self, input_dim, n_actions):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, n_actions)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class RLAgent:
    def __init__(self, n_actions, input_dim):
        self.model = QNetwork(input_dim, n_actions)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

    def act(self, state):
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()

    def train(self, state, action, reward, next_state):
        target = reward + 0.95 * torch.max(self.model(next_state)).item()
        target_f = self.model(state)
        target_f[0][action] = target
        self.optimizer.zero_grad()
        loss = self.criterion(target_f, self.model(state))
        loss.backward()
        self.optimizer.step()

def main():
    # Sample interactions data
    interactions = [
        {"student_id": 1, "knowledge_point_id": 101, "interaction": 1, "time_on_page": 5, "clicks_on_knowledge_points": 2, "reading_time": 30, "weight": 0.5},
        {"student_id": 1, "knowledge_point_id": 102, "interaction": 1, "time_on_page": 10, "clicks_on_knowledge_points": 3, "reading_time": 40, "weight": 0.6},
        {"student_id": 1, "knowledge_point_id": 103, "interaction": 1, "time_on_page": 15, "clicks_on_knowledge_points": 4, "reading_time": 50, "weight": 0.7},
    ]

    # Initialize NLP Model
    nlp_model = NLPModel()

    # Initialize Environment and Agent
    env = RecommendationEnv(n_students=10, n_knowledge_points=5, interactions=interactions, nlp_model=nlp_model)
    agent = RLAgent(n_actions=env.n_knowledge_points, input_dim=env.n_students)

    # Training Loop
    episodes = 10
    for e in range(episodes):
        state = env.reset()
        state = torch.FloatTensor(state).unsqueeze(0)
        for time in range(500):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(1, action)  # Assuming student_id = 1 for simplicity
            next_state = torch.FloatTensor(next_state).unsqueeze(0)
            agent.train(state, action, reward, next_state)
            state = next_state
            if done:
                break
        print(f"Episode {e+1}/{episodes}, Reward: {env.reward}")

    # Example grading
    student_answer = "The capital of France is Paris."
    professor_answer = "Paris is the capital of France."
    grade = env.grade_answer(student_answer, professor_answer)
    print(f"Grading result: {grade}")

if __name__ == "__main__":
    main()