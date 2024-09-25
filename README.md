# LocalClubHouse AI Assistant
## Description
LocalClubhouse AI Assistant is a multimodal chatbot backend designed to provide intelligent responses to user queries about a WordPress website. It uses web scraping, content indexing, and a Large Language Model (LLM) to generate relevant and concise answers.
## Features
 - Web scraping of WordPress sites
 - Content indexing using FAISS
 - Integration with Hugging Face transformers for LLM functionality
 - Flask-based API with CORS support
 - Ability to refresh content index on demand
## Installation
1. Clone the Repo
git clone https://github.com/Zalasyu/RAG_ChatBot_App.git
cd rag_chatbot_app

2. Install dependencies
pip install poetry
poetry install

## Configuration
1. Set up your Hugging Face token in llm_model.py:
login(token='your_huggingface_token_here')
2. Configure the WordPress URL in app.py:
WORDPRESS_URL = 'https://yourwordpresssite.com'

## Usage
1. Start the Flask Server
python app.py

2.The server will run on http://localhost:5000 by default.

3.Use the following API endpoints:

POST /chat: Send a query to get an AI-generated response
POST /refresh_index: Refresh the content index
GET /index_status: Check the status of the content index