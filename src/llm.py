import ollama

from sentence_transformers import CrossEncoder

class LLMInterface:

    def __init__(self, model_name: str = 'llama3.2:3b'):
        """Handles interactions with the language model.
    
        Attributes:
            model_name (str): Name of the LLM model
            system_prompt (str): System prompt for the LLM
        """
        self.model_name = model_name
        self.system_prompt = """
                You are an AI assistant tasked with providing detailed answers based solely on the given context. Your goal is to analyze the information provided and formulate a comprehensive, well-structured response to the question.

                context will be passed as "Context:"
                user question will be passed as "Question:"

                To answer the question:
                1. Thoroughly analyze the context, identifying key information relevant to the question.
                2. Organize your thoughts and plan your response to ensure a logical flow of information.
                3. Formulate a detailed answer that directly addresses the question, using only the information provided in the context.
                4. Ensure your answer is comprehensive, covering all relevant aspects found in the context.
                5. If the context doesn't contain sufficient information to fully answer the question, state this clearly in your response.

                Format your response as follows:
                1. Use clear, concise language.
                2. Organize your answer into paragraphs for readability.
                3. Use bullet points or numbered lists where appropriate to break down complex information.
                4. If relevant, include any headings or subheadings to structure your response.
                5. Ensure proper grammar, punctuation, and spelling throughout your answer.

                Important: Base your entire response solely on the information provided in the context. Do not include any external knowledge or assumptions not present in the given text.
                """

    def re_rank_crossencoder(self, document: list[str], prompt) -> tuple[str, list[int]]:
        """Re-ranks documents using a cross-encoder model for more accurate relevance scoring.

        Uses the MS MARCO MiniLM cross-encoder model to re-rank the input documents based on
        their relevance to the query prompt. Returns the concatenated text of the top 3 most
        relevant documents along with their indices.

        Args:
            documents: List of document strings to be re-ranked.

        Returns:
            tuple: A tuple containing:
                - relevant_text (str): Concatenated text from the top 3 ranked documents
                - relevant_text_ids (list[int]): List of indices for the top ranked documents

        Raises:
            ValueError: If documents list is empty
            RuntimeError: If cross-encoder model fails to load or rank documents
        """
        try:
            relevant_text = ""
            relevant_text_ids = []

            encoder_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

            ranks = encoder_model.rank(prompt, document, top_k=3)
            for rank in ranks:
                relevant_text += document[rank['corpus_id']]
                relevant_text_ids.append(rank['corpus_id'])

            return relevant_text, relevant_text_ids
        except Exception as e:
            raise (f"Error happened while Re-ranks documents using a cross-encoder model - {e}")   
        
    def call_llm(self, context: str, prompt: str):
        """Calls a language model (LLM) to generate responses based on the provided context and prompt.

        This method interacts with an LLM using the Ollama API to obtain a response. It sends a system prompt along with user-defined context and question, and yields the generated response in chunks.

        Args:
            context: A string providing contextual information that informs the LLM's response.
            prompt: A string containing the user's question or query for which a response is sought.

        Yields:
            str: The generated content from the LLM in chunks. Each chunk is yielded as it is received.

        Raises:
            Exception: If there are issues during the interaction with the LLM, such as network errors or invalid parameters.

        Notes:
            - The method uses streaming to receive responses incrementally (stream = True), allowing for real-time updates.
        """
        try:
            response = ollama.chat(
                model = self.model_name,
                stream = True,
                messages = [
                    {
                        "role": "system",
                        "content": self.system_prompt,
                    },
                    {
                        "role": "user",
                        "content": f"Context: {context}, Question: {prompt}",
                    }
                ],
            )

            for chunk in response:
                if chunk['done'] is False:
                    yield chunk['message']['content']
                else:
                    break
        except Exception as e:
            raise (f"Error happened while calling the LLM model to generate responses - {e}")   