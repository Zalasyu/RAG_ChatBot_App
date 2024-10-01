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
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFacePipeline
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)
from accelerate import init_empty_weights, load_checkpoint_and_dispatch, infer_auto_device_map
from airllm import AutoModel

from huggingface_hub import login, hf_hub_download

from vector_store import VectorStore

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

    def __init__(self, model_name: str = "openbmb/MiniCPM-V-2_6-int4"):
        """Initialize the LLMIntegrator

        Args:
            model_name (str, optional): The name of the pre-trained model to use.
        """

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", trust_remote_code=True)
        self.tokenizer.clean_up_tokenization_spaces = True
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Configure quantization
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
        )
        
        with init_empty_weights():
            self.model = AutoModel.from_pretrained(
                model_name)

        # Infer a device map base don available memory
        device_map = infer_auto_device_map(self.model, max_memory={0: "5GB", "cpu": "8GB"})

        # Load the model and dispatch it according to the inferred device map
        self.model = load_checkpoint_and_dispatch(self.model, model_name, device_map, offload_folder="offload")


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
        self.prompt = ChatPromptTemplate.from_template(template)

    def setup_retriaval_qa(self, retriever: Any) -> None:
        """Set up the retrieval QA Chain.

        Args:
            vector_store (Any): The vector store to use for retrieval.
        """

        # Create the RunnableSequence
        self.qa_chain = (
            {
                "context": retriever,
                "question": RunnablePassthrough(),
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )


    # @traceable()
    def answer_question(self, question: str) -> Dict[str, Any]:
        """Generate an answer to a question given some context

        Args:
            question (str): The question to answer
            context (str): The context information for answering the question.

        Returns:
            Dict[str, Any]: A dictionary containing the answer and confidence score.
        """
        response = self.qa_chain.invoke(question)
        torch.cuda.empty_cache()  # Clear cache to free memory

        return {"answer": response}


if __name__ == "__main__":
    llm = LLMIntegrator()
    vector_store_instance = VectorStore("https://traviscountyappliancerepair.com")
    retriever_instance = vector_store_instance.load_retriever()
    llm.setup_retriaval_qa(retriever=retriever_instance)
    result = llm.answer_question(
        "How can I fix a Samsung dryer?"
    )
    print(f"The result: {result}")
