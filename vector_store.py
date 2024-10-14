"""
Module: vector_store.py
Author: Alec Sebastian Moldovan
Created: 2024-09-25
Last Modified: 2024-09-25
Version: 1.0

Description:


"""

import os
from typing import Any, Dict, List

from bs4 import BeautifulSoup
from langchain.document_loaders import AsyncHtmlLoader
from langchain.document_transformers import BeautifulSoupTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.retrievers import ParentDocumentRetriever
from langchain_community.document_transformers import LongContextReorder
from langchain_chroma import Chroma
from langchain.storage._lc_store import create_kv_docstore
from langchain.storage import LocalFileStore
import torch
from langsmith import traceable

from parser import SitemapParser
from dotenv import load_dotenv, find_dotenv

# Load environment variables from .env
load_dotenv(find_dotenv())

from langsmith import Client

langsmith_client = Client(api_key=os.getenv("LANGCHAIN_API_KEY"), api_url=os.getenv("LANGCHAIN_ENDPOINT"), )


class VectorStore:

    def __init__(
        self,
        base_url: str,
        model_name: str = "WhereIsAI/UAE-Large-V1",
        # model_name: str = "intfloat/e5-large-v2",
        collection_name: str = "full_website",
    ):

        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.base_url = base_url
        self.embeddings = HuggingFaceBgeEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        self.vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory="./chroma_db",
        )
        fs = LocalFileStore("./store_location")

        # This text splitter is used to create the parent documents
        self.parent_splitter = RecursiveCharacterTextSplitter.from_language(
            chunk_size=4096,
            chunk_overlap=1024,
            add_start_index=True,
            language=Language.HTML,
        )

        # This text splitter is ued to create the child documents
        self.child_splitter = RecursiveCharacterTextSplitter.from_language(
            chunk_size=512,
            chunk_overlap=128,
            add_start_index=True,
            language=Language.HTML,
        )

        # The storage layer for the parent documents
        self.store = create_kv_docstore(fs)
        self.retriever = ParentDocumentRetriever(
            vectorstore=self.vector_store,
            docstore=self.store,
            child_splitter=self.child_splitter,
            parent_splitter=self.parent_splitter,
        )

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

        # Filter out URLs that likely contain reviews
        filtered_urls = [
            url
            for url in page_urls
            if "reviews" not in url.lower() or "google" not in url.lower()
        ]

        return filtered_urls

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

        # Exclude specific sections or classes that contain reviews
        for review_section in soup.find_all(
            "div",
            {
                "class": [
                    "review",
                    "reviews",
                    "testimonial",
                    "testimonials",
                    "comment",
                    "comments",
                    "google-review",
                    "google-reviews",
                    "customer-review",
                    "customer-reviews",
                ]
            },
        ):
            review_section.decompose()

        # Remove Google reviews or any other embedded review sections
        for review in soup.find_all(
            "iframe", src=lambda x: "google.com" in x if x else False
        ):
            review.decompose()

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

            # If the document contains known review patterns, exclude it
            if "google.com/local/reviews" in doc.page_content or "review" in doc.page_content.lower():
                continue

            # Extract text and image info from each document
            doc.page_content = self.extract_text_n_images(doc)
            processed_docs.append(doc)

        return processed_docs

    def scrape_n_index(self):
        """Scrape all pages and index th content using FAISS"""

        # Get all page URLs
        urls = self.get_all_pages()
        print(f"Total URLs fetched: {len(urls)}")

        # Additional filter to remove any review-related URLs after fetching
        urls = [url for url in urls if "google.com/local/reviews" not in url]

        print(f"Total URLs kept: {len(urls)}")

        # Load HTML content
        docs = self.load_html_content(urls)
        print(f"Number of documents loaded: {len(docs)}")

        # Transform HTML to extract relevant content
        docs_transformed = self.transform_html_documents(docs)
        print(f"Number of documents after transformation: {len(docs_transformed)}")

        # Process documents to extract text and image info.
        processed_docs = self.process_documents(docs_transformed)
        print(f"Number of documents after processing: {len(processed_docs)}")

        # Add Documents
        self.add_documents(processed_docs)

    def add_documents(self, documents: List[Any]):
        """Add documents to the vector store and implement Parent Document Retrieever

        Args:
            documents (List[Any]): A List of Documents to add to the vector store.
        """
        self.retriever.add_documents(documents)

    def list_docs(self) -> None:
        """Lists the documents store in the memory store."""
        print(list(self.store.yield_keys()))

    def load_retriever(self) -> ParentDocumentRetriever:
        """Loads the vector store and the document store, initializing a retriever."""

        self.vector_store = Chroma(
            collection_name="full_website",
            embedding_function=self.embeddings,
            persist_directory="./chroma_db",
        )
        # Verify vector store content
        vector_store_count = self.vector_store._collection.count()
        print(f"Vector store contains {vector_store_count} documents.")

        fs = LocalFileStore("./store_location")

        self.store = create_kv_docstore(fs)
        # Verify document store content
        doc_keys = list(self.store.yield_keys())
        print(f"Document store contains {len(doc_keys)} documents.")

        self.retriever = ParentDocumentRetriever(
            vectorstore=self.vector_store,
            docstore=self.store,
            child_splitter=self.child_splitter,
            parent_splitter=self.parent_splitter,
        )

        return self.retriever
    
    def reorder_documents(self, docs: List[Any]) -> List[Any]:
        """Reorders the documents in the list based on the parent document.

        Args:
            docs (List[Any]): The list of documents to reorder.

        Returns:
            List[Any]: The reordered list of documents.
        """
        reordering = LongContextReorder()
        reordered_docs = reordering.transform_documents(docs)
        return reordered_docs
    

    # TODO: Determine why it keeps returning only 4 documents when I specified 10.
    @traceable(run_type="retriever", name="search", project_name="RAG-CHATBOT")
    def search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """Search the indexed content for relevant info.

        Args:
            query (str): The search query
            k (int, optional): The humber of results to return. Defaults to 4.

        Returns:
            List[Dict[str, Any]]: _description_
        """

        if os.path.exists("./chroma_db") and os.path.exists("./store_location"):
            print(f"Loading retriever...")
            self.load_retriever()
        else:
            print(f"Scraping and indexing...")
            self.scrape_n_index()

        # Retrieve from overall retriever
        retrieved_docs = self.retriever.invoke(query, k=k)
        print(f"Number of documents retrieved: {len(retrieved_docs)}")

        reordered_docs = self.reorder_documents(retrieved_docs)

        return reordered_docs[: k]


if __name__ == "__main__":
    vector_store = VectorStore("https://traviscountyappliancerepair.com")
    # vector_store = VectorStore("https://localclubhouse.com/")
    vector_store.scrape_n_index()
    results = vector_store.search(query="What is your dryer repair process?", k=10)
    # results = vector_store.search("Where to sign up?")
    print(results)
