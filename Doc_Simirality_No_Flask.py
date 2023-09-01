from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# File paths
text1_path = "F:\Company_Data\CHAPTER_Script\Chapters\Class_8\Hindi\sections\section_.txt"
text2_path = "F:\Company_Data\CHAPTER_Script\Chapters\Class_8\science\sections\section_23.txt"

# Read the files
with open(text1_path, 'r', encoding='utf-8') as file:
    text1 = file.read().replace('\n', '')
with open(text2_path, 'r', encoding='utf-8') as file:
    text2 = file.read().replace('\n', '')

texts = [text1, text2]

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# Get embeddings
embeddings = model.encode(texts)

# Calculate cosine similarity
cos_sim = cosine_similarity([embeddings[0]], [embeddings[1]])

print('Cosine similarity: ', cos_sim[0][0])
