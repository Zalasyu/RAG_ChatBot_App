"""
Module: app.py
Author: Alec Moldovan
Created: 2024-09-14
Last Modified: 2024-09-14
Version: 1.0

Description:
This module serves as the main entry point for the multimodal chatbot backend.
It sets up a Flask server with CORS support and provides endpoints for chat 
functionality. The module integrates a question-answering pipeline and a 
content indexer for similarity search.

Dependencies:
- Flask
- flask_cors
- transformers
- sentence_transformers
- faiss
- numpy
- indexing (local module)
"""

import os
from typing import Any, Dict

from flask import Flask, jsonify, request
from flask_cors import CORS

from .indexer import WordPressScraperIndexer
from .llm_model import LLMIntegrator

# Constant Variables
# WORDPRESS_URL = 'https://localclubhouse.com'
WORDPRESS_URL = "https://traviscountyappliancerepair.com"
INDEX_PATH = "wordpress_index"

app = Flask(import_name=__name__)
CORS(app=app)  # Enable CORS for all routes during DEV


# Initialize Content Indexer
scraper_indexer = WordPressScraperIndexer(WORDPRESS_URL)

# Initialize the LLM Integrator
llm = LLMIntegrator()


@app.route("/chat", methods=["POST"])
def chat() -> Dict[str, Any]:
    """
    Handle chat requests from the frontend.

    This function receives a query from the frontend, performs a similarity
    search to find relevant context, and then uses the LLM to generate a response.

    Returns:
        Dict[str, Any]: A dictionary containing the answer and confidence score.

    Raises:
        400 Bad Request: If the request doesn't contain a 'query' field.
    """

    data: Dict[str, str] = request.json

    # Ensure the request contains a query
    if "query" not in data:
        return jsonify({"error": "No query provided"}), 400

    query: str = data["query"]

    try:
        # Perform Similarity Search to find relevant context
        search_results = scraper_indexer.search(query, k=20)

        # Prepare context for the LLM
        context: str = "\n".join([result["content"] for result in search_results])

        # Use the LLM to generate an answer
        llm_result: Dict[str, Any] = llm.answer_question(query, context)

        print(llm_result)

        # Prepare the response
        response = {
            "answer": llm_result["answer"],
            "confidence": llm_result["confidence_score"],
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/refresh_index", methods=["POST"])
def refresh_index() -> Dict[str, str]:
    """Refresh the contennt index by re-scraping the WordPress site

    Returns:
        Dict[str, str]: A dictionary containing a success message or error message.
    """
    try:
        scraper_indexer.scrape_n_index()
        scraper_indexer.save_index(INDEX_PATH)
        return jsonify({"message": "Content index refreshed successfully"})
    except Exception as e:
        return jsonify({"error": f"Failed to refresh index: {str(e)}"}), 500


@app.route("/index_status", methods=["GET"])
def index_status() -> Dict[str, Any]:
    """Get the status of the content index.

    Returns:
        Dict[str, Any]: A dictionary containing the index status info
    """
    INDEX_PATH = "wordpress_index"
    try:
        # TODO: Implement more comprehensive method in WordPressScraperIndexer class
        index_exists = os.path.exists(INDEX_PATH)

        return jsonify({"index_exists": index_exists, "index_path": INDEX_PATH})
    except Exception as e:
        return jsonify({"error": f"Failed to get index status: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
