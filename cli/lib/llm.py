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