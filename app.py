from flask import Flask, request, jsonify
from flask_cors import CORS
from Bio import Entrez, Medline
import os
import re
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
import faiss
from openai import OpenAI
from dotenv import load_dotenv

from waitress import serve
app = Flask(__name__)
# Enable CORS for all routes and origins
CORS(app, origins=["http://localhost:3000", 'https://metacare.ai'])

# Load environment variables from .env file
load_dotenv()

# Environment variable configurations
Entrez.email = os.getenv('ENTREZ_EMAIL')
openai_api_key = os.getenv('OPENAI_API_KEY')
if not Entrez.email or not openai_api_key:
    raise ValueError("Environment variables for Entrez email or OpenAI API key are not set.")

# Initialize the OpenAI client
client = OpenAI(api_key=openai_api_key)

# BERT initialization
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def chat(message):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": f"{message}"},
        ],
        temperature=0.1
    )
    return response.choices[0].message.content


@app.route('/pubmedsummary', methods=['POST'])
def pubmed_summary():
    if request.content_type == 'application/x-www-form-urlencoded':
        question = request.form.get('question')
    else:
        return jsonify({"error": "Unsupported Media Type. Please use application/x-www-form-urlencoded"}), 415

    if not question:
        return jsonify({"error": "Please provide a clinical question."}), 400
    
    pico_res = chat('rewrite the following clinical question according to the PICO model using (P), (I) , (C), (O) notation to the right of the clause:' + question )
    #print(pico_res)

    import re

    # Example input strings
    pico_res_new_format = "P: 49-year-old white male\nI: smoking tobacco\nC: not smoking tobacco\nO: risks"
    pico_res_original_format = "In a 49-year-old white male (P), is smoking tobacco (I) compared with not smoking tobacco (C) associated with risks (O)?"

    # Combined regular expression to capture PICO components from both formats
    pattern = r"(?:In\s+(?P<Patient>.*?)\s+\(P\),\s+(?P<Intervention>.*?)\s+\(I\)\s+(?P<Comparison>.*?)\s+\(C\)\s+(?P<Outcome>.*?)\s+\(O\)\?)|" \
            r"(?:P:\s*(?P<Patient2>[^\n]+)\nI:\s*(?P<Intervention2>[^\n]+)\nC:\s*(?P<Comparison2>[^\n]+)\nO:\s*(?P<Outcome2>.+))"

    # Function to match and merge groupdict results
    def match_and_merge(pico_string, pattern):
        match = re.match(pattern, pico_string, re.DOTALL)
        if match:
            # Merge matched groups from both formats, prioritizing non-None values
            pico_variables = {k.rstrip("2"): v or match.group(k + "2") for k, v in match.groupdict().items() if not k.endswith("2")}
            return pico_variables
        else:
            return "No match found!"

    # Testing the regex pattern with both formats
    pico_variables = match_and_merge(pico_res_new_format, pattern)
    if pico_variables == "No match found!":
        pico_variables = match_and_merge(pico_res_original_format, pattern)

    idList = []
    handle = Entrez.esearch(db="mesh", term=pico_variables['Patient'])
    record = Entrez.read(handle)
    handle.close()
    mesh_terms = []
    for translation in record['TranslationSet']:
        terms = translation['To'].split(' OR ')
        for term in terms:
            if '[MeSH Terms]' in term:
                mesh_terms.append(term.replace('[MeSH Terms]', '').replace('"', '').strip())
    query_terms = [f"{term}" for term in mesh_terms]
    query = " AND ".join(query_terms)
    p_query = query
    #print(p_query)

    handle = Entrez.esearch(db="mesh", term=pico_variables['Intervention'])
    record = Entrez.read(handle)
    handle.close()
    # Extract MeSH terms from the result
    mesh_terms = []
    for translation in record['TranslationSet']:
        terms = translation['To'].split(' OR ')
        for term in terms:
            if '[MeSH Terms]' in term:
                mesh_terms.append(term.replace('[MeSH Terms]', '').replace('"', '').strip())

    query_terms = [f"{term}" for term in mesh_terms]
    query = " OR ".join(query_terms)
    i_query = query
    #print(i_query)

    handle = Entrez.esearch(db="mesh", term=pico_variables['Comparison'])
    record = Entrez.read(handle)
    handle.close()
    mesh_terms = []
    for translation in record['TranslationSet']:
        terms = translation['To'].split(' OR ')
        for term in terms:
            if '[MeSH Terms]' in term:
                mesh_terms.append(term.replace('[MeSH Terms]', '').replace('"', '').strip())
    query_terms = [f"{term}" for term in mesh_terms]
    query = " OR ".join(query_terms)
    c_query = query
    #print(c_query)

    handle = Entrez.esearch(db="mesh", term=pico_variables['Outcome'])
    record = Entrez.read(handle)
    handle.close()
    mesh_terms = []
    for translation in record['TranslationSet']:
        terms = translation['To'].split(' OR ')
        for term in terms:
            if '[MeSH Terms]' in term:
                mesh_terms.append(term.replace('[MeSH Terms]', '').replace('"', '').strip())
    query_terms = [f"{term}" for term in mesh_terms]
    query = " OR ".join(query_terms)
    o_query = query
    #print(o_query)

    final_query = f"({p_query}) AND ({i_query}) AND ({c_query}) AND ({o_query})"
    #print(final_query)

    handle = Entrez.esearch(db="pubmed", term=final_query)
    record = Entrez.read(handle)
    handle.close()
    idlist = record['IdList']
    #print(idlist)
    #print(record['Count'])

    from Bio import Medline
    handle = Entrez.efetch(db="pubmed", id=idlist, rettype="medline",retmode="text")
    records = Medline.parse(handle)
    records = list(records)
    handle.close()

    articles = []

    for record in records:
        title = record.get("TI", "?")
        author = record.get("AU", "?")
        journal = record.get("TA", "?")
        date_of_publication = record.get("DP", "?")
        abstract = record.get("AB", "?")
        keywords = record.get("OT", "?")
        mesh_terms =record.get("MH", "?")
        articles.append((title, abstract, journal, author, date_of_publication, keywords, mesh_terms))
    
    #print(articles.__len__())
    from transformers import BertTokenizer, BertModel
    import torch

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    def embed_text(text):
        if not text:
            return None  # or return a zero vector or another placeholder
        
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs['pooler_output'].numpy()

        
    vectors = [embed_text(article[1]) for article in articles if article[1]]
    vectors = [v for v in vectors if v is not None]
    #print(f"Number of vectors: {len(vectors)}")

    import faiss
    import numpy as np

    # Convert vectors list to a 2D numpy array
    vectors_matrix = np.vstack(vectors)

    # Build the index
    index = faiss.IndexFlatL2(vectors_matrix.shape[1])
    index.add(vectors_matrix)

    query_text = pico_res
    query_vector = embed_text(query_text)
    #print(query_vector.shape)
    #print(pico_res)

    # Define the number of nearest neighbors you want to retrieve
    if len(vectors) >= 5: k = 5
    else: k = len(vectors)

    # Search the index for the k-nearest vectors
    D, I = index.search(query_vector, k)

    # D contains the distances, and I contains the indices of the nearest vectors
    nearest_articles = [articles[i] for i in I[0]]  # I[0] because I is a 2D array
    #print(nearest_articles)
    # Now, print the nearest articles:
    s = ""
    for idx, article in enumerate(nearest_articles):
        title, abstract, journal, authors, date_of_publication, keywords, mesh_terms = article

        # Convert lists to strings
        authors_str = ', '.join(authors) if authors else "N/A"
        keywords_str = ', '.join(keywords) if keywords else "N/A"
        mesh_terms_str = ', '.join(mesh_terms) if mesh_terms else "N/A"

        s += f"Title: {title}\n"
        s += f"Abstract: {abstract}\n"
        s += f"Journal: {journal}\n"
        s += f"Authors: {authors_str}\n"
        s += f"Date of Publication: {date_of_publication}\n"
        s += f"Keywords: {keywords_str}\n"
        s += f"Mesh Terms: {mesh_terms_str}\n\n"

    # Use the GPT model to generate a summary
    research_res = chat_with_openai("Act as an evidenced-based clinical researcher. Using only the following PubMed Abstracts to guide your content (" + s + "), create an evidence based medicine report usig the PICO framework that answers the following PICO question: " + pico_res)
    summary_response = research_res
    # summary_prompt = "Summarize the key findings of the following PubMed articles:\n" + s
    # summary_response = chat_with_openai(summary_prompt)
    

    return jsonify({
        "pico_question": pico_res,
        "articles": nearest_articles,
        "summary": summary_response
    })

def chat_with_openai(message):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": message}],
        temperature=0.5
    )
    return response.choices[0].message.content.strip()

#if __name__ == '__main__':
#    app.run(debug=True)

if __name__ == '__main__':
    serve(app, host='0.0.0.0', port=8080)
