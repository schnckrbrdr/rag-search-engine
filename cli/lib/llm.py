import os
from dotenv import load_dotenv
from google import genai

class LLM():

    def __init__(self, model_name="gemini-2.5-flash"):
        load_dotenv()
        self.api_key = os.environ.get("GEMINI_API_KEY")
        self.model_name = model_name

        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY environment variable not set")
    
        self.client = genai.Client(api_key=self.api_key)

    def enhance_query(self, query: str, enhance_type: str) -> str:

        prompt = ""

        if enhance_type == "spell":
            prompt = f"""Fix any spelling errors in the user-provided movie search query below.
                        Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
                        Preserve punctuation and capitalization unless a change is required for a typo fix.
                        If there are no spelling errors, or if you're unsure, output the original query unchanged.
                        Output only the final query text, nothing else.
                        User query: "{query}"
                        """
        elif enhance_type == "rewrite":
            prompt = f"""Rewrite the user-provided movie search query below to be more specific and searchable.

                        Consider:
                        - Common movie knowledge (famous actors, popular films)
                        - Genre conventions (horror = scary, animation = cartoon)
                        - Keep the rewritten query concise (under 10 words)
                        - It should be a Google-style search query, specific enough to yield relevant results
                        - Don't use boolean logic

                        Examples:
                        - "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
                        - "movie about bear in london with marmalade" -> "Paddington London marmalade"
                        - "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

                        If you cannot improve the query, output the original unchanged.
                        Output only the rewritten query text, nothing else.

                        User query: "{query}"
                        """
        elif enhance_type == "expand":
            prompt = f"""Expand the user-provided movie search query below with related terms.

                        Add synonyms and related concepts that might appear in movie descriptions.
                        Keep expansions relevant and focused.
                        Output only the additional terms; they will be appended to the original query.

                        Examples:
                        - "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
                        - "action movie with bear" -> "action thriller bear chase fight adventure"
                        - "comedy with bear" -> "comedy funny bear humor lighthearted"

                        User query: "{query}"
                        """
        else:
            return query
        
        response = self.client.models.generate_content(
            #model = "gemma-3-27b-it", 
            model = self.model_name,
            contents = prompt
        )

        if response:
            return response.text            
        else:
            raise RuntimeError("Error retrieving LLM-Response")
        
    def rerank_request(self, query: str, movie_details: dict) -> int:

        prompt = rerank_prompt = f"""Rate how well this movie matches the search query.

                                    Query: "{query}"
                                    Movie: {movie_details['document']['title']} - {movie_details['document']['description']}

                                    Consider:
                                    - Direct relevance to query
                                    - User intent (what they're looking for)
                                    - Content appropriateness

                                    Rate 0-10 (10 = perfect match).
                                    Output ONLY the number in your response, no other text or explanation.

                                    Score:"""
        
        response = self.client.models.generate_content(
            #model = "gemma-3-27b-it", 
            model = self.model_name,
            contents = prompt
        )

        if response and response.text.isdigit():
            return int(response.text)
        else:
            return 0
        
    def rerank_batch(self, query, movie_details: list) -> list:
        
        doc_list_str = ""

        for entry in movie_details:
            doc_list_str += f"{entry[1]['document']['id']}: {entry[1]['document']['title']} - {entry[1]['document']['description']}\n\n"

        prompt = f"""Rank the movies listed below by relevance to the following search query.

                    Query: "{query}"

                    The following list of movies is formatted as: "<Movie-ID>: <Movie-Title> - <Movie-Description>" with two consecutive linebreaks between each movie in the list

                    Movies:
                    {doc_list_str}

                    Return ONLY the movie IDs in order of relevance (best match first). Return a valid JSON list, nothing else.

                    Rank ALL movies in the list

                    For example:
                    [75, 12, 34, 2, 1]

                    Ranking:"""            

        response = self.client.models.generate_content(
            #model = "gemma-3-27b-it", 
            model = self.model_name,
            contents = prompt
        )

        try:
            response_list = list(response.text[1:-1].split(','))
        except:
            response_list = []

        return response_list
    
    def evaluation_request(self, query: str, to_evaluate:list[str]):

        prompt = f"""Rate how relevant each result is to this query on a 0-3 scale:

                    Query: "{query}"

                    Results:
                    {chr(10).join(to_evaluate)}

                    Scale:
                    - 3: Highly relevant
                    - 2: Relevant
                    - 1: Marginally relevant
                    - 0: Not relevant

                    Do NOT give any numbers other than 0, 1, 2, or 3.

                    Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

                    [2, 0, 3, 2, 0, 1]"""
        
        response = self.client.models.generate_content(
            model = self.model_name,
            contents = prompt
        )

        try:
            response_list = list(response.text[1:-1].split(','))
        except:
            response_list = []

        return response_list
    
    def rag_request(self, query: str, documents: list[str]) -> str:

        prompt = f"""Answer the question or provide information based on the provided documents. This should be tailored to Hoopla users. Hoopla is a movie streaming service.

                    Query: {query}

                    Documents:
                    {chr(10).join(documents)}

                    Provide a comprehensive answer that addresses the query:"""
        
        response = self.client.models.generate_content(
            model = self.model_name,
            contents = prompt
        )

        if response:
            return response.text            
        else:
            raise RuntimeError("Error retrieving LLM-Response")
        
    def rag_summarize(self, query:str, documents: list[str]) -> str:

        prompt = f"""
                    Provide information useful to this query by synthesizing information from multiple search results in detail.
                    The goal is to provide comprehensive information so that users know what their options are.
                    Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.
                    This should be tailored to Hoopla users. Hoopla is a movie streaming service.
                    Query: {query}
                    Search Results:
                    {chr(10).join(documents)}
                    Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:
                    """
        response = self.client.models.generate_content(
            model = self.model_name,
            contents = prompt
        )

        if response:
            return response.text            
        else:
            raise RuntimeError("Error retrieving LLM-Response")

    def rag_citations(self, query: str, documents: list[str]):

        prompt = f"""Answer the question or provide information based on the provided documents.

                    This should be tailored to Hoopla users. Hoopla is a movie streaming service.

                    If not enough information is available to give a good answer, say so but give as good of an answer as you can while citing the sources you have.

                    Query: {query}

                    Documents:
                    {chr(10).join(documents)}

                    Instructions:
                    - Provide a comprehensive answer that addresses the query
                    - Cite sources using [1], [2], etc. format when referencing information
                    - If sources disagree, mention the different viewpoints
                    - If the answer isn't in the documents, say "I don't have enough information"
                    - Be direct and informative

                    Answer:"""    
        
        response = self.client.models.generate_content(
            model = self.model_name,
            contents = prompt
        )

        if response:
            return response.text            
        else:
            raise RuntimeError("Error retrieving LLM-Response")
        
    def rag_question(self, question: str, documents: list[str]):

        prompt = f"""Answer the user's question based on the provided movies that are available on Hoopla.

                    This should be tailored to Hoopla users. Hoopla is a movie streaming service.

                    Question: {question}

                    Documents:
                    {chr(10).join(documents)}

                    Instructions:
                    - Answer questions directly and concisely
                    - Be casual and conversational
                    - Don't be cringe or hype-y
                    - Talk like a normal person would in a chat conversation

                    Answer:"""   
        
        response = self.client.models.generate_content(
            model = self.model_name,
            contents = prompt
        )

        if response:
            return response.text            
        else:
            raise RuntimeError("Error retrieving LLM-Response")