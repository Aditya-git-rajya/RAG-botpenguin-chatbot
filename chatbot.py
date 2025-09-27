#!/usr/bin/env python3

# =============================================================================
# ADVANCED WEBSITE CHATBOT WITH RAG USING GROQ API
#
# Author: Aditya Chauhan
# Date: September 7, 2025
# Description: Optimized chatbot using a fast, cloud-based LLM (Groq) for
#              intelligent, contextual responses without local model downloads.
# =============================================================================

import requests
from bs4 import BeautifulSoup
import warnings
import os
from groq import Groq
from dotenv import load_dotenv
load_dotenv()  # This loads the .env file

# --- RAG-specific imports ---
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# --- Async scraping imports ---
import asyncio
import aiohttp

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# --- CORE RAG COMPONENTS ---

class KnowledgeBase:
    """
    Manages the website knowledge base for Retrieval-Augmented Generation (RAG).
    It processes text into embeddings and provides a fast similarity search.
    """
    def __init__(self):
        """Initializes the embedding model and FAISS index."""
        print("📥 Initializing embedding model...")
        # The embedding model is still run locally and is a small download.
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = None
        self.chunks = []
        print("✅ Embedding model loaded.")

    def create_from_scraped_data(self, website_content_dict):
        """
        Processes scraped content, chunks it, and builds a searchable index.
        """
        print("🏗️ Building knowledge base from scraped data...")
        all_text = []
        for section, content in website_content_dict.items():
            if isinstance(content, list):
                all_text.extend(content)

        self.chunks = []
        current_chunk = ""
        for item in all_text:
            if len(current_chunk) + len(item) < 500:
                current_chunk += " " + item
            else:
                self.chunks.append(current_chunk.strip())
                current_chunk = item
        if current_chunk:
            self.chunks.append(current_chunk.strip())

        if not self.chunks:
            print("⚠️ No suitable text chunks found to build knowledge base.")
            return

        print(f"📄 Found {len(self.chunks)} text chunks. Creating embeddings...")

        self.embeddings = self.embedding_model.encode(self.chunks, convert_to_tensor=False)
        print("✅ Knowledge base built and indexed.")

    def retrieve_relevant_chunks(self, query, top_k=5):
        """
        Finds the most relevant text chunks from the knowledge base using cosine similarity.
        """
        if self.embeddings is None or len(self.chunks) == 0:
            return []

        query_embedding = self.embedding_model.encode([query], convert_to_tensor=False)
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Get indices of top_k most similar chunks
        indices = np.argsort(similarities)[-top_k:][::-1]
        
        relevant_chunks = [self.chunks[i] for i in indices]
        return relevant_chunks

class LLMResponseGenerator:
    """
    Handles text generation using the Groq API.
    """
    def __init__(self, model_name="llama-3.1-8b-instant"):
        """Initializes the Groq client."""
        print("🤖 Initializing LLM via Groq API...")
        
        # Use environment variable for API key
        api_key = os.environ.get("GROQ_API_KEY")
    
        if not api_key:
            raise ValueError("Please set the GROQ_API_KEY environment variable.")
        
        self.client = Groq(api_key=api_key)
        self.model_name = model_name
        print(f"✅ LLM client initialized with model: {self.model_name}.")

    def generate_response(self, user_query, context, chat_history):
        """
        Generates a response using the Groq API based on context and history.
        """
        # Create the full prompt with context and history
        full_context = "\n\n".join(context)
        
        # Groq's API uses a list of messages for history
        messages = [
            {"role": "system", "content": "You are a helpful chatbot for the BotPenguin website. Answer user questions based ONLY on the provided context."}
        ]
        
        for turn in chat_history:
            messages.append({"role": "user", "content": turn['user']})
            messages.append({"role": "assistant", "content": turn['bot']})
            
        messages.append({"role": "user", "content": f"Context: {full_context}\n\nQuestion: {user_query}"})

        try:
            chat_completion = self.client.chat.completions.create(
                messages=messages,
                model=self.model_name,
                temperature=0.7,
                max_tokens=200,
            )
            response = chat_completion.choices[0].message.content.strip()
            return response
        except Exception as e:
            print(f"❌ Error generating response from Groq API: {e}")
            return "I am unable to generate a response at this time. Please try again later."

class AdvancedWebsiteChatbot:
    """
    Core chatbot class that orchestrates web scraping, RAG, and user interaction.
    """
    def __init__(self, website_url):
        print("Initializing Advanced RAG Chatbot...")
        self.website_url = website_url
        self.scraped_data = None
        self.knowledge_base = KnowledgeBase()
        self.llm_generator = LLMResponseGenerator()
        print("✅ Chatbot initialized.")

    async def fetch_page(self, session, url):
        """Asynchronously fetches a single page."""
        try:
            async with session.get(url, timeout=10) as response:
                response.raise_for_status()
                return await response.text(), None
        except aiohttp.ClientError as e:
            return None, f"Error fetching {url}: {e}"
        except asyncio.TimeoutError:
            return None, f"Timeout fetching {url}"

    async def scrape_website_comprehensive(self):
        """
        Optimized web scraper using async/await to fetch content faster.
        """
        structured_content = {
            'headers': [],
            'paragraphs': [],
            'lists': [],
        }
        async with aiohttp.ClientSession() as session:
            print(f"📡 Fetching content from: {self.website_url}...")
            html, error = await self.fetch_page(session, self.website_url)

            if error:
                print(f"❌ Error during web scraping: {error}")
                return None

            soup = BeautifulSoup(html, 'html.parser')
            print("✅ Content fetched successfully.")

            for header in soup.find_all(['h1', 'h2', 'h3']):
                structured_content['headers'].append(' '.join(header.get_text(strip=True).split()))
            for p in soup.find_all('p'):
                structured_content['paragraphs'].append(' '.join(p.get_text(strip=True).split()))
            for ul in soup.find_all(['ul', 'ol']):
                structured_content['lists'].extend([
                    ' '.join(li.get_text(strip=True).split())
                    for li in ul.find_all('li')
                ])

            title = soup.find('title')
            structured_content['metadata'] = {
                'title': title.string if title else 'No Title'
            }
        return structured_content

    def answer_question_with_rag(self, query, chat_history):
        """
        Answers a user query using the RAG pipeline.
        """
        relevant_chunks = self.knowledge_base.retrieve_relevant_chunks(query, top_k=5)
        if not relevant_chunks:
            return "I'm sorry, I couldn't find any relevant information on the website to answer your question."

        response = self.llm_generator.generate_response(query, relevant_chunks, chat_history)
        return response

    def start_advanced_chat(self):
        """Starts the console-based chat session."""
        print("==========================================================")
        print(f"      ADVANCED RAG CHATBOT for: {self.website_url}")
        print("==========================================================")
        print("A.I. is processing the website content. Please wait...")

        self.scraped_data = asyncio.run(self.scrape_website_comprehensive())

        if not self.scraped_data:
            print("❌ Failed to scrape the website. Cannot start chat.")
            return

        self.knowledge_base.create_from_scraped_data(self.scraped_data)

        if not self.knowledge_base.chunks:
            print("❌ Knowledge base is empty. Cannot start chat.")
            return

        print("\nHello! I have analyzed the website. How can I help you?")
        print("Type 'quit' or 'exit' to end the chat.")

        chat_history = []
        while True:
            user_input = input("\nYou: ").strip()
            if user_input.lower() in ['quit', 'exit']:
                print("Goodbye!")
                break
            
            response = self.answer_question_with_rag(user_input, chat_history)
            
            print(f"Bot: {response}")
            chat_history.append({"user": user_input, "bot": response})
            # Keep history to a manageable size
            if len(chat_history) > 5:
                chat_history.pop(0)

# Example Usage:
if __name__ == "__main__":
    url_to_analyze = "https://botpenguin.com"
    chatbot = AdvancedWebsiteChatbot(url_to_analyze)
    chatbot.start_advanced_chat()