from enviroment import RecommendationEnv
from nlp_model import NLPModel

# Sample interactions data
interactions = [
    {"student_id": 1, "knowledge_point_id": 101, "interaction": 1, "time_on_page": 5, "clicks_on_knowledge_points": 2, "reading_time": 30, "weight": 0.5},
    {"student_id": 1, "knowledge_point_id": 102, "interaction": 1, "time_on_page": 10, "clicks_on_knowledge_points": 3, "reading_time": 40, "weight": 0.6},
    {"student_id": 1, "knowledge_point_id": 103, "interaction": 1, "time_on_page": 15, "clicks_on_knowledge_points": 4, "reading_time": 50, "weight": 0.7},
]

# Initialize NLP Model
nlp_model = NLPModel()

# Initialize Environment
env = RecommendationEnv(n_students=10, n_knowledge_points=5, interactions=interactions, nlp_model=nlp_model)

# Sample usage
state, reward, done, _ = env.step(1, 2)
print(state, reward)

student_answer = "The capital of France is Paris."
professor_answer = "Paris is the capital of France."
grade = env.grade_answer(student_answer, professor_answer)
print(f"Grading result: {grade}")