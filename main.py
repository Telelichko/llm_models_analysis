import json
import pandas as pd
from runners.evaluation_runner import EvaluationRunner
import config


def analyze_results(results_path: str):
    """Analyzes results and creates a summary table"""

    with open(results_path, 'r', encoding='utf-8') as f:
        results = [json.loads(line) for line in f]

    # Create DataFrame for analysis
    df = pd.DataFrame(results)

    # Group by model and calculate average metrics
    summary = df.groupby('model').agg({
        'overall_score': 'mean',
        'accuracy_score': 'mean',
        'completeness_score': 'mean',
        'relevance_score': 'mean',
        'clarity_score': 'mean',
        'latency': 'mean',
        'success': 'sum'
    }).round(3)

    # Rename columns
    summary = summary.rename(columns={
        'overall_score': 'Avg Overall Score',
        'accuracy_score': 'Avg Accuracy',
        'completeness_score': 'Avg Completeness',
        'relevance_score': 'Avg Relevance',
        'clarity_score': 'Avg Clarity',
        'latency': 'Avg Latency (s)',
        'success': 'Successful Tests'
    })

    # Save to CSV
    summary.to_csv('results/summary.csv')
    print(summary)

    # Save full report to Excel
    with pd.ExcelWriter('results/full_report.xlsx') as writer:
        summary.to_excel(writer, sheet_name='Summary')
        df.to_excel(writer, sheet_name='Full Data', index=False)

    return summary


def main():
    # Run testing
    runner = EvaluationRunner()
    results = runner.run_evaluation(
        config.MODELS_TO_TEST,
        config.GOLDEN_DATASET_PATH,
        config.RESULTS_PATH
    )

    # Analyze results
    summary = analyze_results(config.RESULTS_PATH)
    print("Evaluation completed! Results saved to results/")


if __name__ == "__main__":
    main()