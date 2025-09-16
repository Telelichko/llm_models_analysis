from openai import OpenAI
import json
import config


class LLMJudge:
    def __init__(self):
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.judge_model = "gpt-4-turbo"

    def evaluate_response(self, prompt: str, reference: str, response: str, model_name: str) -> dict:
        """Evaluates model response using LLM judge"""

        evaluation_prompt = f"""
        You are an experienced AI model response evaluator. Evaluate the model's response to the following query:

        QUERY: {prompt}

        REFERENCE ANSWER: {reference}

        MODEL RESPONSE ({model_name}): {response}

        Evaluate the model response on the following criteria (from 1 to 10):
        1. Accuracy: how factually correct the response is and how well it matches the reference
        2. Completeness: how fully the response addresses the query
        3. Relevance: how well the response follows instructions in the query
        4. Clarity: how clear and understandable the response is

        Return the response in JSON format:
        {{
            "accuracy_score": number,
            "completeness_score": number,
            "relevance_score": number,
            "clarity_score": number,
            "overall_score": number,
            "explanation": string
        }}
        """

        try:
            response = self.client.chat.completions.create(
                model=self.judge_model,
                messages=[{"role": "user", "content": evaluation_prompt}],
                temperature=0
            )

            evaluation = json.loads(response.choices[0].message.content)
            return evaluation

        except Exception as e:
            return {
                "accuracy_score": 0,
                "completeness_score": 0,
                "relevance_score": 0,
                "clarity_score": 0,
                "overall_score": 0,
                "explanation": f"Evaluation failed: {str(e)}"
            }