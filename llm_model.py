"""
Module for integrating a Large Language Model (LLM) into the chatbot.

This module provides functionality to load and use a pre-trained language
model for generating responses in the chatbot application.
"""
import os
from typing import Any, Dict
from regex import D

import torch
from dotenv import load_dotenv, find_dotenv
from langsmith import traceable, Client
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain.memory import ConversationBufferMemory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFacePipeline
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)
from accelerate import infer_auto_device_map

from huggingface_hub import login

from vector_store import VectorStore

load_dotenv(find_dotenv())  # Load environment variables from .env file


login(token=os.getenv("HUGGINGFACE_TOKEN"))

langsmith_client = Client(api_key=os.getenv("LANGCHAIN_API_KEY"), api_url=os.getenv("LANGCHAIN_ENDPOINT"))


class LLMIntegrator:
    """
    A class for integrating and using a Large Language Model.

    This class handles loading the model and tokenizer, and provides
    methods for generating responses.

    Attributes:
        model (AutoModelForCausalLM): The loaded language model.
        tokenizer (AutoTokenizer): The tokenizer for the model.
        device (torch.device): The device (CPU/GPU) on which the model is loaded.
    """

    def __init__(self, model_name: str = "unsloth/Llama-3.2-1B"):
        """Initialize the LLMIntegrator

        Args:
            model_name (str, optional): The name of the pre-trained model to use.
        """

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.tokenizer.clean_up_tokenization_spaces = True
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Configure quantization
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
        )

        # config = AutoConfig.from_pretrained(model_name)

        # Initialize your model
        # self.model = AutoModel.from_config(config=config)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=quantization_config
        )

        # Infer a device map base don available memory
        device_map = infer_auto_device_map(
            self.model, max_memory={0: "4GB", "cpu": "6GB"}
        )
        print(device_map)

        # Create a HuggingFacePipeline
        pipe = pipeline(
            task="text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=250,
            pad_token_id=self.tokenizer.eos_token_id,
            device_map=device_map,
            return_full_text=False,
            do_sample=True,
            top_p=0.95,
        )
        # Wrap the pipeline in a LangChain LLM
        self.llm = HuggingFacePipeline(pipeline=pipe)

        # Set up conversation memory
        self.memory = ConversationBufferMemory(
            input_key="question", memory_key="chat_history"
        )

        system_prompt = (    
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )

        self.prompt = ChatPromptTemplate(
            [
                ("system", system_prompt),
                ("human", "{question}"),
            ]
        )


    # TODO: Determine why LLM model does not consider context when generating answer.
    @traceable(run_type="chain")
    def setup_retriaval_qa(self, retriever: Any) -> None:
        """Set up the retrieval QA Chain.

        Args:
            vector_store (Any): The vector store to use for retrieval.
        """
        # Define a runnable that retrieves documents based on the question
        retrieved_docs = RunnableLambda(lambda question: retriever.search(question, k=10))

        # Convert documents to a single string
        docs_to_text_with_sources = RunnableLambda(
            lambda docs:  {
                "context": "\n\n".join([doc.page_content for doc in docs]),
                "sources": [doc.metadata["source"] for doc in docs]
                }
        )

        # Runnable to print context
        print_context = RunnableLambda(
            lambda context: (print(f"Context:\n{context['context']}\n"), context)[1]
        )

        # Runnable to print prompt
        print_prompt = RunnableLambda(
            lambda prompt: (print(f"Prompt:\n{prompt}\n"), prompt)[1]
        )

        # Create the RunnableSequence
        self.qa_chain = (
            {
                "context": retrieved_docs | docs_to_text_with_sources,
                "question": RunnablePassthrough(),
            }
            | self.prompt
            | print_prompt
            | self.llm
            | StrOutputParser()
        )


    @traceable(run_type="llm")
    def answer_question(self, question: str) -> Dict[str, Any]:
        """Generate an answer to a question given some context

        Args:
            question (str): The question to answer
            context (str): The context information for answering the question.

        Returns:
            Dict[str, Any]: A dictionary containing the answer and confidence score.
        """
        response = self.qa_chain.invoke(question)

        
        # Free up GPU memory
        torch.cuda.empty_cache()  # Clear cache to free memory

        return {"answer": response}


if __name__ == "__main__":
    llm = LLMIntegrator()
    vector_store_instance = VectorStore("https://traviscountyappliancerepair.com")
    llm.setup_retriaval_qa(retriever=vector_store_instance)
    result = llm.answer_question("What could cause my dishwasher to break?")
    print(f"The result: {result}")
