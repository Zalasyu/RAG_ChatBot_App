"""
Module for integrating a Large Language Model (LLM) into the chatbot.

This module provides functionality to load and use a pre-trained language
model for generating responses in the chatbot application.
"""

import os
from typing import Any, Dict

import torch
from dotenv import load_dotenv
from langsmith import traceable
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.memory import ConversationBufferMemory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

from huggingface_hub import login

load_dotenv()  # Load environment variables from .env file


login(token=os.getenv("HUGGINGFACE_TOKEN"))


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

    def __init__(self, vector_store: Any, model_name: str = "mistralai/Mistral-7B-v0.1" ):
        """Initialize the LLMIntegrator

        Args:
            model_name (str, optional): The name of the pre-trained model to use. Defaults to 'gpt2'.
        """

        self.vector_store = vector_store

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.tokenizer.clean_up_tokenization_spaces = True
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Configure quantization
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=quantization_config, device_map="auto"
        )

        # Create a HuggingFacePipeline
        pipe = pipeline(
            task="text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=100,
            pad_token_id=self.tokenizer.eos_token_id,
            device_map="auto",
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

        template = """You are a helpful AI assistant for a website. Your task is to provide a direct and concise response to the user's question. Follow these rules strictly:

        1. If the question is about signing up or accessing a specific page, provide ONLY a direct link or clear, brief instructions.
        2. Your response should be no more than 30 words.
        3. Do not mention the context or rephrase the question in your answer.
        4. Start your response with a relevant action verb (e.g., "Visit", "Go to", "Click", etc.) when appropriate.
        5. If the context does not contain relevant information to answer the question, respond with "I don't have enough information to answer that question."
        6. If you provide a URL, ensure it's complete and correct based on the context.

        Use this context to inform your answer, but do not repeat it verbatim:
        Context:{context}
        Question: {question}
        Answer: """
        self.prompt = PromptTemplate.from_template(
            template=template, input_variables=["context", "question"]
        )

    def setup_retriaval_qa(self, vector_store: Any) -> None:
        """Set up the retrieval QA Chain.

        Args:
            vector_store (Any): The vector store to use for retrieval.
        """

        # Create the RunnableSequence
        self.qa_chain = (
            {
                "context": vector_store.as_retriever(seatch_kwargs={"k": 3}),
                "question": RunnablePassthrough(),
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
    @traceable()
    def answer_question(self, question: str) -> Dict[str, Any]:
        """Generate an answer to a question given some context

        Args:
            question (str): The question to answer
            context (str): The context information for answering the question.

        Returns:
            Dict[str, Any]: A dictionary containing the answer and confidence score.
        """
        response = self.qa_chain.invoke(question)

        # TODO: Implement a method to calculate the confidence score
        confidence_score = 0.8  # Placeholder

        return {"answer": response, "confidence_score": confidence_score}


if __name__ == "__main__":
    llm = LLMIntegrator("mistralai/Mistral-7B-v0.1")
    result = llm.answer_question(
        "What is Python?",
        "Python is a high-level programming language. It is used for making apps",
    )
    print(f"The result: {result}")
