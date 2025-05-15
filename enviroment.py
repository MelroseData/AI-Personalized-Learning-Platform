import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

class RecommendationEnv:
    def __init__(self, n_students, n_knowledge_points, interactions, nlp_model):
        self.n_students = n_students
        self.n_knowledge_points = n_knowledge_points
        self.interactions = pd.DataFrame(interactions)
        self.state = np.zeros(n_students)
        self.action_space = np.arange(n_knowledge_points)
        self.reward = 0
        self.nlp_model = nlp_model
        self.similarity_matrix = self.calculate_similarity()

    def calculate_similarity(self):
        pivot_table = self.interactions.pivot_table(index='student_id', columns='knowledge_point_id', values='interaction', fill_value=0)
        similarity_matrix = cosine_similarity(pivot_table)
        return pd.DataFrame(similarity_matrix, index=pivot_table.index, columns=pivot_table.index)

    def reset(self):
        self.state = np.zeros(self.n_students)
        return self.state

    def step(self, student_id, action):
        similar_students = self.similarity_matrix[student_id].sort_values(ascending=False).index[1:4]
        similar_students_data = self.interactions[self.interactions['student_id'].isin(similar_students)]
        recommended_points = similar_students_data['knowledge_point_id'].value_counts().index
        reward = 1 if action in recommended_points else 0
        self.reward += reward * self.interactions.loc[self.interactions['student_id'] == student_id, 'weight'].mean()  # Incorporate weight
        self.state[action] = reward

        # Update new factors (illustrative)
        self.interactions.loc[self.interactions['student_id'] == student_id, 'time_on_page'] += np.random.rand()  # Random increment for illustration
        self.interactions.loc[self.interactions['student_id'] == student_id, 'clicks_on_knowledge_points'] += np.random.randint(1, 5)  # Random increment for illustration
        self.interactions.loc[self.interactions['student_id'] == student_id, 'reading_time'] += np.random.rand()  # Random increment for illustration

        return self.state, reward, False, {}

    def grade_answer(self, student_answer, professor_answer):
        return self.nlp_model.predict(student_answer, professor_answer)