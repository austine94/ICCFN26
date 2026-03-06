import json
import pandas as pd

def generate_delta_table(mr_json='experiment_2_KG_MR_results.json', kg_json='experiment_3_KG_Only_results.json'):
    #produces table showing the benefit of adding in map reduce 
    
    try:
        with open(mr_json, 'r') as f:
            mr_data = json.load(f)
        with open(kg_json, 'r') as f:
            kg_data = json.load(f)
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return

    def get_summary_df(data):
        rows = []
        for k_str, iterations in data.items():
            for entry in iterations:
                rows.append({
                    'k': int(k_str),
                    'model': entry['model'],
                    # Multiply by 100 here to get percentage straight away
                    'accuracy': int(entry['is_correct']) * 100 
                })
        df = pd.DataFrame(rows)
        # Calculate mean accuracy per model, per k
        return df.groupby(['model', 'k'])['accuracy'].mean()

    #calculate average accuracies
    mr_acc = get_summary_df(mr_data)
    kg_acc = get_summary_df(kg_data) # <-- Fixed the typo here!

    #calculate the Delta (Improvement)
    #delta = (Accuracy with Map-Reduce) - (Accuracy with Raw KG Only)
    delta_acc = mr_acc - kg_acc
    delta_df = delta_acc.reset_index()
    delta_df.rename(columns={'accuracy': 'Delta (%)'}, inplace=True)

    pivot_table = delta_df.pivot(index='model', columns='k', values='Delta (%)')

    print("\n### Accuracy Delta Table (Improvement from Map-Reduce) ###")
    print("Values represent: [MR Accuracy %] - [Baseline Accuracy %]")
    print("-" * 65)
    
    pivot_table.to_csv('accuracy_delta_analysis.csv')
    print("\nSuccess! Delta analysis saved to accuracy_delta_analysis.csv")

if __name__ == "__main__":
    generate_delta_table()