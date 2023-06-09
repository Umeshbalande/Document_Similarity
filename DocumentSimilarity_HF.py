from sentence_transformers import SentenceTransformer, util
from flask import Flask, request

app = Flask(__name__)

class DocumentSimilarity:
    def __init__(self, documents, model_name="all-mpnet-base-v2", threshold=0.7):
        self.documents = documents
        self.model_name = model_name
        self.threshold = threshold
        self.model = SentenceTransformer(self.model_name)
        self.similar_documents = self._calculate_similarities()

    def _calculate_similarities(self):
        embeddings = self.model.encode(self.documents, convert_to_tensor=True)
        cosine_scores = util.pytorch_cos_sim(embeddings, embeddings)
        similar_documents = []
        num_docs = len(self.documents)
        for i in range(num_docs):
            for j in range(i+1, num_docs):
                if cosine_scores[i][j] > self.threshold:
                    similar_documents.append((self.documents[i], self.documents[j], cosine_scores[i][j].item()))
        similar_documents.sort(key=lambda x: x[2], reverse=True)
        return similar_documents

    def get_similar_documents(self, top_n):
        similar_docs = []
        for doc in self.similar_documents[:top_n]:
            similar_docs.append({
                'document1': doc[0],
                'document2': doc[1],
                'similarity': doc[2]
            })
        return similar_docs

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the texts from the user
        text1 = request.form.get('text1')
        text2 = request.form.get('text2')

        # Create a list of the texts
        texts = [text1, text2]

        # Create an instance of the DocumentSimilarity class
        ds = DocumentSimilarity(texts)

        # Get the top 2 similar documents
        similar_docs = ds.get_similar_documents(2)

        # Format and return the response
        response = ""
        for doc in similar_docs:
            response += f"{doc['document1']} is similar to {doc['document2']} with a cosine similarity of {doc['similarity']}<br>"
        return response

    # Render the HTML form for input
    return '''
        <form method="POST">
            <label for="text1">Text 1:</label><br>
            <input type="text" id="text1" name="text1"><br><br>
            <label for="text2">Text 2:</label><br>
            <input type="text" id="text2" name="text2"><br><br>
            <input type="submit" value="Submit">
        </form>
    '''

if __name__ == '__main__':
    app.run(debug=True)
    #app.run(host='0.0.0.0', port=5000)
