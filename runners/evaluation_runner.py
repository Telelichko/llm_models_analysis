import json
from adapters import get_adapter
from evaluators.llm_judge import LLMJudge


class EvaluationRunner:
    def __init__(self):
        self.judge = LLMJudge()

    def run_evaluation(self, models: list, dataset_path: str, results_path: str):
        """Runs comparative testing of models"""

        # Load golden dataset
        with open(dataset_path, 'r', encoding='utf-8') as f:
            test_cases = [json.loads(line) for line in f]

        results = []

        for model_name in models:
            print(f"Testing {model_name}...")
            adapter = get_adapter(model_name)

            for i, test_case in enumerate(test_cases):
                print(f"  Running test case {i + 1}/{len(test_cases)}")

                # Run model
                result = adapter.run_with_metrics(test_case["prompt"])

                # Evaluate response
                if result["success"]:
                    evaluation = self.judge.evaluate_response(
                        test_case["prompt"],
                        test_case["reference_answer"],
                        result["response"],
                        model_name
                    )
                else:
                    evaluation = {
                        "accuracy_score": 0,
                        "completeness_score": 0,
                        "relevance_score": 0,
                        "clarity_score": 0,
                        "overall_score": 0,
                        "explanation": f"Model failed: {result['error']}"
                    }

                # Save result
                record = {
                    "model": model_name,
                    "test_case_id": test_case["id"],
                    "prompt": test_case["prompt"],
                    "reference_answer": test_case["reference_answer"],
                    "response": result["response"],
                    "latency": result["latency"],
                    "success": result["success"],
                    "error": result["error"],
                    **evaluation
                }

                results.append(record)

                # Save after each test (in case of crash)
                with open(results_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')

        return results
