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


def generate_output_path(results_file_path):
    """Generate output path for PNG file based on the results JSON file path."""
    directory = os.path.dirname(results_file_path)
    filename = os.path.basename(results_file_path)
    name, ext = os.path.splitext(filename)
    output_path = os.path.join(directory, f"{name}.png")
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
