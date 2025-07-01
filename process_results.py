import os
import sys
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from itertools import combinations
from sklearn.metrics import cohen_kappa_score

import warnings
warnings.filterwarnings('ignore')


def generate_output_path(results_file_path, new_extension="png"):
    """Generate output path for PNG file based on the results JSON file path."""
    directory = os.path.dirname(results_file_path)
    filename = os.path.basename(results_file_path)
    name, ext = os.path.splitext(filename)
    output_path = os.path.join(directory, f"{name}.{new_extension}")
    return output_path

def load_and_process_data(file_path):
    """Load JSON data and process it into a structured format."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    processed_data = []
    for item in data:
        if item['status'] == 'completed' and 'label.responses' in item:
            responses = item['label.responses']
            users = item['label.responses.users']
            
            # Create a row for each annotation
            for i, (response, user) in enumerate(zip(responses, users)):
                processed_data.append({
                    'id': item['id'],
                    'prompt': item['prompt'],
                    'answer_a': item['answer_a'],
                    'answer_b': item['answer_b'],
                    'lang': item['lang'],
                    'model_A': item['model_A'],
                    'model_B': item['model_B'],
                    'response': response,
                    'user': user,
                    'annotator_index': i
                })
    
    return pd.DataFrame(processed_data)

def calculate_majority_vote(df):
    """Calculate majority vote for each item."""
    majority_votes = []
    
    for item_id in df['id'].unique():
        item_responses = df[df['id'] == item_id]['response'].tolist()
        
        # Count responses
        response_counts = Counter(item_responses)
        
        # Get majority vote (most common response)
        majority_response = response_counts.most_common(1)[0][0]
        majority_count = response_counts.most_common(1)[0][1]
        
        # Check if there's a tie
        is_tie = len([count for count in response_counts.values() if count == majority_count]) > 1
        
        majority_votes.append({
            'id': item_id,
            'majority_vote': majority_response,
            'majority_count': majority_count,
            'total_annotations': len(item_responses),
            'is_tie': is_tie,
            'agreement_ratio': majority_count / len(item_responses)
        })
    
    return pd.DataFrame(majority_votes)

def calculate_inter_annotator_agreement(df):
    """Calculate inter-annotator agreement metrics."""
    agreements = []
    
    for item_id in df['id'].unique():
        item_data = df[df['id'] == item_id]
        responses = item_data['response'].tolist()
        users = item_data['user'].tolist()
        
        if len(responses) < 2:
            continue
        
        # Calculate pairwise agreement
        total_pairs = 0
        agreeing_pairs = 0
        
        for i in range(len(responses)):
            for j in range(i + 1, len(responses)):
                total_pairs += 1
                if responses[i] == responses[j]:
                    agreeing_pairs += 1
        
        agreement_rate = agreeing_pairs / total_pairs if total_pairs > 0 else 0
        
        agreements.append({
            'id': item_id,
            'pairwise_agreement': agreement_rate,
            'num_annotators': len(responses),
            'unique_responses': len(set(responses))
        })
    
    return pd.DataFrame(agreements)

def calculate_cohen_kappa(df):
    """Calculate Cohen's kappa for each pair of annotators."""
    # Get items that have been annotated by multiple annotators
    annotator_pairs = []
    
    # Find common items between annotator pairs
    annotators = df['user'].unique()
    
    for ann1, ann2 in combinations(annotators, 2):
        ann1_data = df[df['user'] == ann1]
        ann2_data = df[df['user'] == ann2]
        
        # Find common items
        common_items = set(ann1_data['id']) & set(ann2_data['id'])
        
        if len(common_items) >= 5:  # Only calculate if enough common items
            ann1_responses = []
            ann2_responses = []
            
            for item_id in common_items:
                ann1_resp = ann1_data[ann1_data['id'] == item_id]['response'].iloc[0]
                ann2_resp = ann2_data[ann2_data['id'] == item_id]['response'].iloc[0]
                ann1_responses.append(ann1_resp)
                ann2_responses.append(ann2_resp)
            
            kappa = cohen_kappa_score(ann1_responses, ann2_responses)
            
            annotator_pairs.append({
                'annotator_1': ann1,
                'annotator_2': ann2,
                'kappa': kappa,
                'common_items': len(common_items)
            })
    
    return pd.DataFrame(annotator_pairs)

def create_visualizations(df, majority_df, agreement_df, kappa_df, out_plot_path):
    """Create comprehensive visualizations."""
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Overall Response Distribution
    ax1 = plt.subplot(2, 4, 1)
    response_counts = df['response'].value_counts()
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    plt.pie(response_counts.values, labels=response_counts.index, autopct='%1.1f%%', 
            colors=colors, startangle=90)
    plt.title('Overall Response Distribution\n(All Annotations)', fontsize=12, fontweight='bold')
    
    # 2. Majority Vote Distribution
    ax2 = plt.subplot(2, 4, 2)
    majority_counts = majority_df['majority_vote'].value_counts()
    plt.pie(majority_counts.values, labels=majority_counts.index, autopct='%1.1f%%', 
            colors=colors, startangle=90)
    plt.title('Majority Vote Distribution\n(Per Item)', fontsize=12, fontweight='bold')
    
    # 3. Agreement Rate Distribution
    ax3 = plt.subplot(2, 4, 3)
    plt.hist(majority_df['agreement_ratio'], bins=20, alpha=0.7, color='#45B7D1', edgecolor='black')
    plt.xlabel('Agreement Ratio')
    plt.ylabel('Number of Items')
    plt.title('Distribution of Agreement Ratios\n(Majority Vote Strength)', fontsize=12, fontweight='bold')
    plt.axvline(majority_df['agreement_ratio'].mean(), color='red', linestyle='--', 
                label=f'Mean: {majority_df["agreement_ratio"].mean():.2f}')
    plt.legend()
    
    # 4. Inter-annotator Agreement
    ax4 = plt.subplot(2, 4, 4)
    plt.hist(agreement_df['pairwise_agreement'], bins=15, alpha=0.7, color='#96CEB4', edgecolor='black')
    plt.xlabel('Pairwise Agreement Rate')
    plt.ylabel('Number of Items')
    plt.title('Pairwise Agreement Distribution', fontsize=12, fontweight='bold')
    plt.axvline(agreement_df['pairwise_agreement'].mean(), color='red', linestyle='--',
                label=f'Mean: {agreement_df["pairwise_agreement"].mean():.2f}')
    plt.legend()
    
    # 5. Response by Language (if multiple languages)
    ax5 = plt.subplot(2, 4, 5)
    if df['lang'].nunique() > 1:
        lang_response = pd.crosstab(df['lang'], df['response'], normalize='index')
        lang_response.plot(kind='bar', ax=ax5, color=colors)
        plt.title('Response Distribution by Language', fontsize=12, fontweight='bold')
        plt.xlabel('Language')
        plt.ylabel('Proportion')
        plt.xticks(rotation=45)
        plt.legend(title='Response', bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        plt.text(0.5, 0.5, 'Only one language\nin dataset', ha='center', va='center', 
                transform=ax5.transAxes, fontsize=12)
        plt.title('Language Distribution', fontsize=12, fontweight='bold')
        ax5.set_xticks([])
        ax5.set_yticks([])
    
    # 6. Agreement vs Number of Annotators
    ax6 = plt.subplot(2, 4, 6)
    if len(agreement_df) > 0:
        plt.scatter(agreement_df['num_annotators'], agreement_df['pairwise_agreement'], 
                   alpha=0.6, color='#FF6B6B')
        plt.xlabel('Number of Annotators')
        plt.ylabel('Pairwise Agreement')
        plt.title('Agreement vs Number of Annotators', fontsize=12, fontweight='bold')
        
        # Add trend line
        z = np.polyfit(agreement_df['num_annotators'], agreement_df['pairwise_agreement'], 1)
        p = np.poly1d(z)
        plt.plot(agreement_df['num_annotators'], p(agreement_df['num_annotators']), "r--", alpha=0.8)
    
    # 7. Cohen's Kappa Distribution
    ax7 = plt.subplot(2, 4, 7)
    if len(kappa_df) > 0:
        plt.hist(kappa_df['kappa'], bins=10, alpha=0.7, color='#FFD93D', edgecolor='black')
        plt.xlabel("Cohen's Kappa")
        plt.ylabel('Number of Annotator Pairs')
        plt.title("Cohen's Kappa Distribution", fontsize=12, fontweight='bold')
        plt.axvline(kappa_df['kappa'].mean(), color='red', linestyle='--',
                    label=f'Mean: {kappa_df["kappa"].mean():.2f}')
        plt.legend()
        
        # Add interpretation lines
        plt.axvline(0.2, color='orange', linestyle=':', alpha=0.7, label='Fair')
        plt.axvline(0.4, color='yellow', linestyle=':', alpha=0.7, label='Moderate')
        plt.axvline(0.6, color='lightgreen', linestyle=':', alpha=0.7, label='Substantial')
        plt.axvline(0.8, color='green', linestyle=':', alpha=0.7, label='Almost Perfect')
    else:
        plt.text(0.5, 0.5, 'Insufficient overlapping\nannotations for\nCohen\'s Kappa', 
                ha='center', va='center', transform=ax7.transAxes, fontsize=12)
        plt.title("Cohen's Kappa Distribution", fontsize=12, fontweight='bold')
    
    # 8. Model Performance Comparison (if model names are available)
    ax8 = plt.subplot(2, 4, 8)
    model_comparison = df.groupby(['response']).size().reset_index(name='count')
    
    # Calculate win rates for models
    total_responses = len(df)
    model_a_wins = len(df[df['response'] == 'answer_a'])
    model_b_wins = len(df[df['response'] == 'answer_b'])
    ties_both = len(df[df['response'] == 'both'])
    ties_none = len(df[df['response'] == 'none'])
    
    categories = ['Model A Wins', 'Model B Wins', 'Both Good', 'Both Wrong']
    values = [model_a_wins, model_b_wins, ties_both, ties_none]
    
    bars = plt.bar(categories, values, color=colors)
    plt.title('Model Performance Summary', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Annotations')
    plt.xticks(rotation=45)
    
    # Add percentages on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{value}\n({value/total_responses*100:.1f}%)',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(out_plot_path, dpi=300, bbox_inches='tight')
    plt.show()

def create_disagreement_table(df, majority_df, output_path_base):
    """
    Create a detailed table of samples with less than 100% agreement for manual inspection.
    
    Args:
        df: DataFrame with individual annotations
        majority_df: DataFrame with majority vote results
        output_path_base: Base path for output files (without extension)
    
    Returns:
        pd.DataFrame: Table with disagreement cases
    """
    # Get items with less than 100% agreement
    disagreement_items = majority_df[majority_df['agreement_ratio'] < 1.0]['id'].tolist()
    
    if not disagreement_items:
        print("🎉 All items have 100% agreement!")
        return pd.DataFrame()
    
    disagreement_data = []
    
    for item_id in disagreement_items:
        # Get all annotations for this item
        item_annotations = df[df['id'] == item_id].copy()
        
        # Get the first row for basic item info
        first_row = item_annotations.iloc[0]
        
        # Get all responses and users
        responses = item_annotations['response'].tolist()
        users = item_annotations['user'].tolist()
        
        # Get majority vote info
        majority_info = majority_df[majority_df['id'] == item_id].iloc[0]
        
        # Count each response type
        response_counts = item_annotations['response'].value_counts().to_dict()
        
        # Create response summary string
        response_summary = []
        for resp_type in ['answer_a', 'answer_b', 'both', 'none']:
            count = response_counts.get(resp_type, 0)
            if count > 0:
                response_summary.append(f"{resp_type}: {count}")
        response_summary_str = " | ".join(response_summary)
        
        # Create individual responses string
        individual_responses = []
        for i, (user, response) in enumerate(zip(users, responses)):
            individual_responses.append(f"Ann{i+1}: {response}")
        individual_responses_str = " | ".join(individual_responses)
        
        disagreement_data.append({
            'id': item_id,
            'prompt': first_row['prompt'],
            'answer_a': first_row['answer_a'],
            'answer_b': first_row['answer_b'],
            'model_a': first_row['model_A'],
            'model_b': first_row['model_B'],
            'language': first_row['lang'],
            'majority_vote': majority_info['majority_vote'],
            'agreement_ratio': f"{majority_info['agreement_ratio']:.1%}",
            'response_counts': response_summary_str,
            'individual_responses': individual_responses_str,
            'num_annotators': majority_info['total_annotations'],
            'is_tie': majority_info['is_tie']
        })
    
    disagreement_df = pd.DataFrame(disagreement_data)
    
    # Sort by agreement ratio (lowest first) and then by number of annotators
    disagreement_df = disagreement_df.sort_values(['agreement_ratio', 'num_annotators'])
    
    # Save to CSV for easy inspection
    csv_path = output_path_base.replace('.png', '_disagreements.csv')
    disagreement_df.to_csv(csv_path, index=False, encoding='utf-8')
    
    # Also save a more readable HTML version
    html_path = output_path_base.replace('.png', '_disagreements.html')
    
    # Create HTML with better formatting
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Disagreement Cases - Manual Inspection</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background-color: #f2f2f2; font-weight: bold; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .prompt {{ max-width: 300px; word-wrap: break-word; }}
            .answer {{ max-width: 400px; word-wrap: break-word; }}
            .disagreement {{ background-color: #ffe6e6; }}
            .tie {{ background-color: #fff3cd; }}
            h1 {{ color: #333; }}
            .summary {{ background-color: #e9f4ff; padding: 15px; margin-bottom: 20px; border-radius: 5px; }}
        </style>
    </head>
    <body>
        <h1>Manual Inspection: Cases with Disagreement</h1>
        <div class="summary">
            <strong>Summary:</strong> {len(disagreement_df)} items with less than 100% agreement out of {len(majority_df)} total items 
            ({len(disagreement_df)/len(majority_df)*100:.1f}% disagreement rate)
        </div>
    """
    
    # Add table
    html_content += "<table>"
    html_content += "<tr>"
    for col in ['ID', 'Prompt', 'Model A Answer', 'Model B Answer', 'Models', 'Language', 
                'Majority Vote', 'Agreement', 'Response Counts', 'Individual Responses']:
        html_content += f"<th>{col}</th>"
    html_content += "</tr>"
    
    for _, row in disagreement_df.iterrows():
        row_class = "tie" if row['is_tie'] else "disagreement"
        html_content += f'<tr class="{row_class}">'
        html_content += f'<td>{row["id"]}</td>'
        html_content += f'<td class="prompt">{row["prompt"]}</td>'
        html_content += f'<td class="answer">{row["answer_a"]}</td>'
        html_content += f'<td class="answer">{row["answer_b"]}</td>'
        html_content += f'<td><strong>A:</strong> {row["model_a"]}<br><strong>B:</strong> {row["model_b"]}</td>'
        html_content += f'<td>{row["language"]}</td>'
        html_content += f'<td><strong>{row["majority_vote"]}</strong></td>'
        html_content += f'<td>{row["agreement_ratio"]}</td>'
        html_content += f'<td>{row["response_counts"]}</td>'
        html_content += f'<td>{row["individual_responses"]}</td>'
        html_content += '</tr>'
    
    html_content += """
        </table>
        <div class="summary">
            <strong>Legend:</strong><br>
            • <span style="background-color: #ffe6e6; padding: 2px 5px;">Red rows</span>: Clear majority but with disagreement<br>
            • <span style="background-color: #fff3cd; padding: 2px 5px;">Yellow rows</span>: Tied votes (no clear majority)<br>
            • Agreement %: Percentage of annotators who agreed with the majority vote
        </div>
    </body>
    </html>
    """
    
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"\n📋 DISAGREEMENT ANALYSIS:")
    print(f"   • Items with disagreement: {len(disagreement_df)} out of {len(majority_df)} ({len(disagreement_df)/len(majority_df)*100:.1f}%)")
    print(f"   • Disagreement table saved as: {csv_path}")
    print(f"   • HTML report saved as: {html_path}")
    
    if len(disagreement_df) > 0:
        print(f"   • Worst agreement: {disagreement_df['agreement_ratio'].iloc[0]}")
        print(f"   • Items with ties: {disagreement_df['is_tie'].sum()}")
    
    return disagreement_df

def print_summary_statistics(df, majority_df, agreement_df, kappa_df):
    """Print summary statistics."""
    print("="*60)
    print("A/B TEST RESULTS SUMMARY")
    print("="*60)
    
    print(f"\n📊 DATASET OVERVIEW:")
    print(f"   • Total annotations: {len(df)}")
    print(f"   • Unique items: {df['id'].nunique()}")
    print(f"   • Unique annotators: {df['user'].nunique()}")
    print(f"   • Languages: {', '.join(df['lang'].unique())}")
    
    print(f"\n🎯 RESPONSE DISTRIBUTION:")
    response_dist = df['response'].value_counts(normalize=True)
    for response, pct in response_dist.items():
        print(f"   • {response}: {pct:.1%}")
    
    print(f"\n🤝 INTER-ANNOTATOR AGREEMENT:")
    if len(agreement_df) > 0:
        print(f"   • Mean pairwise agreement: {agreement_df['pairwise_agreement'].mean():.2%}")
        print(f"   • Median pairwise agreement: {agreement_df['pairwise_agreement'].median():.2%}")
        print(f"   • Items with perfect agreement: {len(agreement_df[agreement_df['pairwise_agreement'] == 1.0])}")
        print(f"   • Items with no agreement: {len(agreement_df[agreement_df['pairwise_agreement'] == 0.0])}")
    
    if len(kappa_df) > 0:
        print(f"   • Mean Cohen's Kappa: {kappa_df['kappa'].mean():.3f}")
        print(f"   • Median Cohen's Kappa: {kappa_df['kappa'].median():.3f}")
        
        # Interpret kappa values
        mean_kappa = kappa_df['kappa'].mean()
        if mean_kappa < 0:
            interpretation = "Poor (worse than chance)"
        elif mean_kappa < 0.2:
            interpretation = "Slight"
        elif mean_kappa < 0.4:
            interpretation = "Fair"
        elif mean_kappa < 0.6:
            interpretation = "Moderate"
        elif mean_kappa < 0.8:
            interpretation = "Substantial"
        else:
            interpretation = "Almost Perfect"
        
        print(f"   • Agreement interpretation: {interpretation}")
    
    print(f"\n🏆 MAJORITY VOTE RESULTS:")
    majority_dist = majority_df['majority_vote'].value_counts(normalize=True)
    for response, pct in majority_dist.items():
        print(f"   • {response}: {pct:.1%}")
    
    print(f"   • Mean agreement ratio: {majority_df['agreement_ratio'].mean():.2%}")
    print(f"   • Items with ties: {majority_df['is_tie'].sum()}")

def main():
    """Main function to run the analysis."""
    
    # results_file_path = "./output/records_veritasQA_en-es-ca_20250626_1614.json"
    if len(sys.argv) > 1:
        results_file_path = sys.argv[1]
    else:
        print("Usage: python process_results.py <path_to_json_file>")
        sys.exit(1)

    out_plot_path = generate_output_path(results_file_path)
    
    try:
        # Load and process data
        print(f"Loading data from '{results_file_path}'...")
        df = load_and_process_data(results_file_path)
        
        if len(df) == 0:
            print("No valid data found in the JSON file.")
            return
        
        print(f"Loaded {len(df)} annotations for {df['id'].nunique()} items")
        
        # Calculate various metrics
        print("Calculating majority votes...")
        majority_df = calculate_majority_vote(df)
        
        print("Calculating inter-annotator agreement...")
        agreement_df = calculate_inter_annotator_agreement(df)
        
        print("Calculating Cohen's kappa...")
        kappa_df = calculate_cohen_kappa(df)
        
        # Create visualizations
        print("Creating visualizations...")
        create_visualizations(df, majority_df, agreement_df, kappa_df, out_plot_path)

        # Create disagreement table for manual inspection
        print("Creating disagreement analysis...")
        disagreement_df = create_disagreement_table(df, majority_df, out_plot_path)
        
        # Print summary statistics
        print_summary_statistics(df, majority_df, agreement_df, kappa_df)
        
        print(f"\n✅ Analysis complete! Visualization saved as '{out_plot_path}'")
        
    except FileNotFoundError:
        print(f"❌ Error: '{results_file_path}' file not found in the current directory.")
        print("Please make sure the file exists and try again.")
    except json.JSONDecodeError:
        print(f"❌ Error: Invalid JSON format in '{results_file_path}'.")
        print("Please check the file format and try again.")
    except Exception as e:
        print(f"❌ An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
