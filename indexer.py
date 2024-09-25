"""
Module: indexing.py
Author: Alec Sebastian Moldovan
Created: 2024-09-14
Last Modified: 2024-09-15
Version: 1.0

Description:
Module for web scraping WordPress sites and indexing content using LangChain and FAISS.

This module provides functionality to scrape all pages of a WordPress site,
extract text and image content, and index it using FAISS for efficient retrieval.

"""

import os
from parser import SitemapParser
from typing import Any, Dict, List

from bs4 import BeautifulSoup
from langchain.document_loaders import AsyncHtmlLoader
from langchain.document_transformers import BeautifulSoupTransformer
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from langchain.vectorstores import FAISS


class WordPressScraperIndexer:

    def __init__(self, base_url: str, model_name: str = "BAAI/bge-base-en-v1.5"):
        """Initialize the WordPressScraperIndexer

        Args:
            base_url (str): The base URL of the WordPress site to scrape.
        """

        self.base_url = base_url
        self.embeddings = HuggingFaceBgeEmbeddings(model_name=model_name, model_kwargs = {"device": "cpu"}, encode_kwargs = {"normalize_embeddings": True})
        self.vector_store = None

    def create_sitemap_url(self, base_url: str) -> str:
        """Create the sitemap url from the base url of the WordPress site

        Args:
            base_url (str): WordPress base URL

        Returns:
            str: Sitemap URL
        """
        return f"{base_url}/sitemap.xml"

    def get_all_pages(self) -> List[str]:
        """Get all page URLs from the WordPress site's sitemap

        Returns:
            List[str]: List of all page URLs
        """
        sitemap_url = self.create_sitemap_url(base_url=self.base_url)
        parser = SitemapParser()
        page_urls = parser.get_all_page_urls(sitemap_url)
        return list(page_urls)

    def load_html_content(self, urls: List[str]) -> List[Any]:
        """Load HTML content from the given URLs

        Args:
            urls (List[str]): List of URLs to load.

        Returns:
            List[Any]: List of loaded HTML documents.
        """

        # Use AsyncHtmlLoader to efficiently load HTML content from multiple URLs
        loader = AsyncHtmlLoader(urls)
        return loader.load()

    def transform_html_documents(self, docs: List[Any]) -> List[Any]:
        """Transform HTML documents to extract relevant content.

        Args:
            docs (List[Any]): List of HTML documents

        Returns:
            List[Any]: List of transformed HTML documents
        """

        # Use BeautifulSoupTransformer to extract relevant content from HTML
        bs_transformer = BeautifulSoupTransformer()
        return bs_transformer.transform_documents(docs)

    def extract_text_n_images(self, doc: Any) -> str:
        """Extract text and image information from document.

        Args:
            doc (Any): The document to process

        Returns:
            str: Extracted text and image info.
        """

        # Parse the document content with BeautifulSoup
        soup = BeautifulSoup(doc.page_content, "html.parser")

        # Extract all text content
        text_content = soup.get_text(separator="\n", strip=True)

        # Extract info. about all images
        images = []

        for img in soup.find_all("img"):
            src = img.get("src", "")
            alt = img.get("alt", "")
            if src:
                images.append(f"Image: {src} (Alt: {alt})")

        return text_content + "\n" + "\n".join(images)

    def process_documents(self, docs: List[Any]) -> List[Any]:
        """
        Process documents to extract text and image info.

        Args:
            docs (List[Any]): List of documents to process

        Returns:
            List[Any]: Processed documents
        """
        processed_docs = []

        for doc in docs:

            # Extract text and image info from each document
            doc.page_content = self.extract_text_n_images(doc)
            processed_docs.append(doc)

        return processed_docs

    def split_documents(self, docs: List[Any]) -> List[Any]:
        """Split documents to smaller chunks. Due to context length limitations for LLMs.
        Each LLM has a different context length max.

        Args:
            docs (List[Any]): Documents to split

        Returns:
            List[Any]: Split documents
        """

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.HTML,
            chunk_size=1250, chunk_overlap=500
        )
        return text_splitter.split_documents(docs)

    def create_faiss_index(self, docs: List[Any]):
        """Create a FAISS index from the given documents

        Args:
            docs (List[Any]): Documents to index
        """

        if not docs:
            raise ValueError("No docments to index")

        # Create the FAISS index from the documents using the specified embeddings
        self.vector_store = FAISS.from_documents(docs, self.embeddings)

    def scrape_n_index(self):
        """Scrape all pages and index th content using FAISS"""

        # Get all page URLs
        urls = self.get_all_pages()
        print(f"Total URLs fetched: {len(urls)}")
        print(urls)

        # Load HTML content
        docs = self.load_html_content(urls)
        print(f"Number of documents loaded: {len(docs)}")

        # Transform HTML to extract relevant content
        docs_transformed = self.transform_html_documents(docs)
        print(f"Number of documents after transformation: {len(docs_transformed)}")

        # Process documents to extract text and image info.
        processed_docs = self.process_documents(docs_transformed)
        print(f"Number of documents after processing: {len(processed_docs)}")

        # Split docs into chunks
        splits = self.split_documents(processed_docs)
        print(f"Number of documents after splitting: {len(splits)}")

        # Create FAISS index
        self.create_faiss_index(splits)

    def save_index(self, path: str):
        """Save the FAISS index to a file

        Args:
            path (str): The path to save the index
        """
        self.vector_store.save_local(path)

    def load_index(self, path: str):
        """Load the FAISS index from a file.

        Args:
            path (str): The path to load the index from.
        """
        self.vector_store = FAISS.load_local(
            path, self.embeddings, allow_dangerous_deserialization=True
        )

    def load_or_create_index(self, path: str = "wordpress_index"):
        """
        Load the existing index or create a new one if it doesn't exist.

        Args:
            path (str, optional): Folder where index relevant info stored.. Defaults to 'wordpress_index'.


        Returns:
            _type_: _description_
        """
        if os.path.exists(path):
            self.load_index(path)
        else:
            self.scrape_and_index()
            self.save_index(path)

    def search(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        """Search the indexed content for relevant info.

        Args:
            query (str): The search query
            k (int, optional): The humber of results to return. Defaults to 4.

        Returns:
            List[Dict[str, Any]]: _description_
        """
        self.load_or_create_index()
        print(self.vector_store)

        if not self.vector_store:
            raise ValueError("Index not loaded. Call load_index() first")

        # Perform similarity search using FAISS index
        results = self.vector_store.similarity_search_with_score(query, k)
        return [{"content": doc.page_content, "score": score} for doc, score in results]


if __name__ == "__main__":
    scraper_indexer = WordPressScraperIndexer("https://traviscountyappliancerepair.com")
    scraper_indexer.scrape_n_index()
    scraper_indexer.save_index('wordpress_index')
    results = scraper_indexer.search("How to repair dryer?")
    print(results)
