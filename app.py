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
import faiss
import numpy as np
from waitress import serve
from Bio import Medline
import re


app = Flask(__name__)
# Enable CORS for all routes and origins
#CORS(app, origins=["https://metacare.ai"])
CORS(app, origins=["http://localhost:3000"])

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
@app.route('/pico', methods=['POST'])
def pico():   

    if request.content_type == 'application/x-www-form-urlencoded':
        question = request.form.get('question')
    else:
        return jsonify({"error": "Unsupported Media Type. Please use application/x-www-form-urlencoded"}), 415
    if not question:
        return jsonify({"error": "Please provide a clinical question that is answerable with an evidence based medicine report."}), 400
    print('transforming pico')
    pico_res = chat('Rewrite the following clinical question according to the PICO model using (P), (I) , (C), (O) notation like P:  <data>/lnI: <data>/nC: <data>/nO: <data>:' + question )
    print(pico_res)
    return jsonify({
            "pico_question": pico_res
        })

@app.route('/pubmedsummary', methods=['POST'])
def pubmed_summary():
    
    if request.content_type == 'application/x-www-form-urlencoded':
        question = request.form.get('question')
    else:
        return jsonify({"error": "Unsupported Media Type. Please use application/x-www-form-urlencoded"}), 415

    if not question:
        return jsonify({"error": "Please provide a clinical question that is answerable with an evidence based medicine report."}), 400
    print('transforming pico')
    pico_res = chat('Rewrite the following clinical question according to the PICO model using (P), (I) , (C), (O) notation like P:  <data>/lnI: <data>/nC: <data>/nO: <data>:' + question )
    print(pico_res)
    print('transformed pico')
    # Define the Regular Expression pattern to match the sections
    pattern = r'^P: (.+)$\n^I: (.+)$\n^C: (.+)$\n^O: (.+)$'

    # Compile the regular expression for performance if it's going to be used multiple times
    compiled_pattern = re.compile(pattern, re.MULTILINE)
    # Attempt to match the pattern against the provided text
    match = compiled_pattern.search(pico_res)
    if match:
        # Extract the groups from the match
        patient, intervention, comparison, outcome = match.groups()
        # Format the results into a string with key/value pairs
        result_string = {"Patient": {patient}, 
                         "Intervention" : {intervention}, 
                         "Comparison": {comparison}, 
                         "Outcome" : {outcome},
                        }  
        
        print(result_string)

        print("searching Entrez")    
        handle = Entrez.esearch(db="mesh", term=result_string['Patient'])
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
        print(p_query)

        handle = Entrez.esearch(db="mesh", term=result_string['Intervention'])
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
        print(i_query)

        handle = Entrez.esearch(db="mesh", term=result_string['Comparison'])
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
        print(c_query)

        handle = Entrez.esearch(db="mesh", term=result_string['Outcome'])
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
        print(o_query)

        final_query = f"({p_query}) AND ({i_query}) AND ({c_query}) AND ({o_query})"
        print(final_query)

        handle = Entrez.esearch(db="pubmed", term=final_query)
        record = Entrez.read(handle)
        handle.close()
        idlist = record['IdList']
        print(idlist)
        print(record['Count'])

        handle = Entrez.efetch(db="pubmed", id=idlist, rettype="medline",retmode="text")
        records = Medline.parse(handle)
        records = list(records)
        handle.close()
        print("searched Entrez")  
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

        print(len(articles))
          
        print("embedding vectors")      
        vectors = [embed_text(article[1]) for article in articles if article[1]]
        vectors = [v for v in vectors if v is not None]
        print(f"Number of vectors: {len(vectors)}")

        # Convert vectors list to a 2D numpy array
        vectors_matrix = np.vstack(vectors)
        print(vectors_matrix)
        # Build the index
        index = faiss.IndexFlatL2(vectors_matrix.shape[1])
        index.add(vectors_matrix)

        query_text = pico_res
        query_vector = embed_text(query_text)
        print(query_vector.shape)

        # Define the number of nearest neighbors you want to retrieve
        if len(vectors) >= 7: k = 7
        else: k = len(vectors)

        # Search the index for the k-nearest vectors
        D, I = index.search(query_vector, k)
        print(I, D)

        # D contains the distances, and I contains the indices of the nearest vectors
        nearest_articles = [articles[i] for i in I[0]]  # I[0] because I is a 2D array
        print("vector search complete")  
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
        print("creating report")      
        research_res = chat_with_openai("Act as an evidenced-based clinical researcher. Using only the following PubMed Abstracts to guide your content (" + s + "), create an evidence based medicine report usig the PICO framework that answers the following PICO question: " + pico_res)
        print("report complete")  
        summary_response = research_res
        # summary_prompt = "Summarize the key findings of the following PubMed articles:\n" + s
        # summary_response = chat_with_openai(summary_prompt)
        return jsonify({
            "pico_question": pico_res,
            "articles": nearest_articles,
            "summary": summary_response
        })
    else:
        return jsonify({
            "pico_question": pico_res
        })
def chat_with_openai(message):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": message}],
        temperature=0.5
    )
    return response.choices[0].message.content.strip()
def embed_text(text):
    if not text:
        return None  # or return a zero vector or another placeholder
    # BERT initialization
    # model = BertModel.from_pretrained('bert-base-uncased')
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs['pooler_output'].numpy()

if __name__ == '__main__':
    app.run(debug=True)

#if __name__ == '__main__':
#    serve(app, host='0.0.0.0', port=8080)
