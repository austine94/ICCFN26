import json
import pandas as pd

def generate_delta_table(mr_json='experiment_3_results.json', NA_json='experiment_4_results.json'):
    #produces table showing the benefit of adding in map reduce 
    
    try:
        with open(mr_json, 'r') as f:
            mr_data = json.load(f)
        with open(NA_json, 'r') as f:
            NA_data = json.load(f)
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
    NA_acc = get_summary_df(NA_data) 

    #calculate the Delta (Improvement)
    #delta = (Accuracy with Map-Reduce) - (Accuracy with No KG or MR)
    delta_acc = mr_acc - NA_acc
    delta_df = delta_acc.reset_index()
    delta_df.rename(columns={'accuracy': 'Delta (%)'}, inplace=True)

    pivot_table = delta_df.pivot(index='model', columns='k', values='Delta (%)')

    print("\n### Accuracy Delta Table (Improvement Using Map-Reduce) ###")
    print("-" * 65)
    
    pivot_table.to_csv('accuracy_delta_analysis.csv')
    print("\nSuccess! Delta analysis saved to accuracy_delta_analysis.csv")

if __name__ == "__main__":
    generate_delta_table()