from django.shortcuts import render, redirect
from django.http import JsonResponse
import os
from dotenv import load_dotenv
load_dotenv()
from django.conf import settings
import tempfile

from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.csv_loader import UnstructuredCSVLoader
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader


from langchain_mistralai.chat_models import ChatMistralAI
from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain


import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.data.path.append('/tmp/nltk_data')

# Create your views here.
MISTRAL_API_KEY = os.environ["app"]

def doc(request):
    if request.method == 'POST':
        uploaded_file = request.FILES.get('file')
        if not uploaded_file:
            return JsonResponse({'error': 'Please upload a file'}, status=400)
            
         # Create a temporary file to hold the uploaded content
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name
        
        # Determine file type and load accordingly
        if uploaded_file.name.endswith('.pdf'):
            loader = PyPDFLoader(temp_file_path)
        elif uploaded_file.name.endswith('.txt'):
            loader = TextLoader(temp_file_path)
        elif uploaded_file.name.endswith('.xlsx') or uploaded_file.name.endswith('.xls'):
            loader = UnstructuredExcelLoader(temp_file_path, mode="elements")
        elif uploaded_file.name.endswith('.csv'):
            loader = UnstructuredCSVLoader(temp_file_path, mode="elements")
        elif uploaded_file.name.endswith('.docx'):
            loader = UnstructuredWordDocumentLoader(temp_file_path, mode="elements")
        elif uploaded_file.name.endswith('.md'):
            loader = UnstructuredMarkdownLoader(temp_file_path, mode="elements")

        else:
            os.remove(temp_file_path)  # Clean up the temporary file
            return JsonResponse({'error': 'Unsupported file type'}, status=400) 
        
        docs = loader.load_and_split()

        # Split text into chunks 
        text_splitter = RecursiveCharacterTextSplitter()
        documents = text_splitter.split_documents(docs)
        print("documents", documents)
        # Define the embedding model
        embeddings = MistralAIEmbeddings(model="mistral-embed", mistral_api_key=MISTRAL_API_KEY)
        # Create the vector store 
        vector = FAISS.from_documents(documents, embeddings)
        
        
        global retriever
        retriever = vector.as_retriever()
        
        os.remove(temp_file_path)
            
        return redirect('chatbot')

    return render(request, 'doc.html')
    

def ask_mistralai(message):
    
    # Define LLM
    model = ChatMistralAI(model="mistral-large-latest", mistral_api_key=MISTRAL_API_KEY)
    # Define prompt template
    prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context and if you don't know the answer, just say you don't know the answer and don't make up the answer and don't assume the answer.:

    <context>
    {context}
    </context>

    Question: {input}""")

    # Create a retrieval chain to answer questions
    document_chain = create_stuff_documents_chain(model, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    response = retrieval_chain.invoke({"input": message})
    print(response)
    return response["answer"]
   

def chatbot(request):

    if request.method == 'POST':
        message = request.POST.get('message')
        response = ask_mistralai(message)

        return JsonResponse({'message': message, 'response': response})
    return render(request, 'chat.html',)
