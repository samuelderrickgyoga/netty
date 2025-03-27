import pandas as pd
import numpy as np
from random import choice, uniform, randint, random
import argparse

def generate_realistic_cow_data(num_cows=100, days_per_cow=6, mastitis_prevalence=0.3, output_file='cow_mastitis_data.csv'):
    """
    Generate synthetic cow data for mastitis detection with enhanced clinical realism
    
    Parameters:
    -----------
    num_cows : int
        Number of unique cows to generate
    days_per_cow : int
        Number of consecutive days of measurements per cow
    mastitis_prevalence : float
        The proportion of cows that will develop mastitis
    output_file : str
        Filename for the output CSV
    """
    breeds = ['Jersey', 'Holstein']
    cow_ids = [f'cow{i}' for i in range(1, num_cows+1)]
    
    data = []
    
    for cow_id in cow_ids:
        breed = choice(breeds)
        months_post_birth = randint(1, 12)
        prev_mastitis = 1 if random() < 0.15 else 0  # 15% chance of previous mastitis
        
        # Apply risk factors
        base_mastitis_risk = mastitis_prevalence
        # Older cows have 2x higher risk
        if months_post_birth > 6:
            base_mastitis_risk *= 2
        # Previous mastitis increases risk by 50%
        if prev_mastitis == 1:
            base_mastitis_risk *= 1.5
            
        # Cap the risk at 90%
        mastitis_risk = min(base_mastitis_risk, 0.9)
        
        # Determine if this cow will develop mastitis
        will_develop_mastitis = 1 if random() < mastitis_risk else 0
            
        # Determine on which day mastitis will start (if applicable)
        # More likely to start in the middle of the observation period
        if will_develop_mastitis:
            # Modify this part to match days_per_cow
            day_weights = [0.1, 0.15, 0.25, 0.25, 0.15, 0.1]
            # Truncate or pad weights to match days_per_cow
            if days_per_cow > len(day_weights):
                day_weights += [0.1] * (days_per_cow - len(day_weights))
            else:
                day_weights = day_weights[:days_per_cow]
            
            # Normalize weights
            day_weights = [w/sum(day_weights) for w in day_weights]
            
            mastitis_start_day = np.random.choice(range(1, days_per_cow+1), p=day_weights)
        else:
            mastitis_start_day = None
        
        # Rest of the function remains the same as in the original code
        # Set breed-specific baseline parameters
        if breed == 'Jersey':
            healthy_udder_range = (150, 250)
            temp_normal_range = (38.0, 39.5)
        else:  # Holstein
            healthy_udder_range = (250, 350)
            temp_normal_range = (38.0, 39.5)
        
        # Generate baseline values for this cow (individual variation)
        cow_baseline_temp = uniform(*temp_normal_range)
        cow_baseline_udder = [
            randint(healthy_udder_range[0], healthy_udder_range[1]) 
            for _ in range(8)
        ]
        
        # Generate data for each day
        for day in range(1, days_per_cow + 1):
            # Check if mastitis is active on this day
            has_mastitis_today = will_develop_mastitis and day >= mastitis_start_day
            
            # Add daily variation to baseline temperature (±0.3°C)
            daily_temp_variation = uniform(-0.3, 0.3)
            
            if has_mastitis_today:
                days_with_mastitis = day - mastitis_start_day + 1
                # More realistic progression curve - sigmoid-like
                severity_factor = min(1.0, days_with_mastitis / (days_per_cow - mastitis_start_day + 1) * 1.5)
                
                # Clinical progression based on severity
                # Temperature rises first
                temp_increase = severity_factor * uniform(1.0, 2.5)
                temp = round(cow_baseline_temp + temp_increase + daily_temp_variation, 1)
                
                # Then hardness appears (at ~30% severity)
                hardness = 1 if severity_factor > 0.3 else 0
                
                # Then pain develops (at ~50% severity)
                pain = 1 if severity_factor > 0.5 else 0
                
                # Finally milk changes become visible (at ~70% severity)
                milk_visible = 1 if severity_factor > 0.7 else 0
                
                # Inflamed udder measurements that increase with severity
                measurements = []
                for i, baseline in enumerate(cow_baseline_udder):
                    # Different quarters can be affected differently
                    quarter_severity = severity_factor * uniform(0.8, 1.2)
                    # Inflammation causes swelling (increase in measurements)
                    increase = int(baseline * quarter_severity * 0.5)  # Up to 50% increase
                    # Add some random variation (±10%)
                    variation = randint(int(-baseline * 0.1), int(baseline * 0.1))
                    measurements.append(baseline + increase + variation)
                
            else:
                # Healthy characteristics with slight day-to-day variation
                temp = round(cow_baseline_temp + daily_temp_variation, 1)
                hardness = 0
                pain = 0
                milk_visible = 0
                
                # Normal udder measurements with slight day-to-day variation (±10%)
                measurements = []
                for baseline in cow_baseline_udder:
                    variation = randint(int(-baseline * 0.1), int(baseline * 0.1))
                    measurements.append(baseline + variation)
            
            # Create the row
            row = [
                cow_id, day, breed, months_post_birth, prev_mastitis,
                *measurements,
                temp, hardness, pain, milk_visible, 1 if has_mastitis_today else 0
            ]
            data.append(row)
    
    columns = [
        'Cow_ID', 'Day', 'Breed', 'Months_after_giving_birth', 'Previous_Mastits_status',
        'IUFL', 'EUFL', 'IUFR', 'EUFR', 'IURL', 'EURL', 'IURR', 'EURR',
        'Temperature', 'Hardness', 'Pain', 'Milk_visibility', 'class1'
    ]
    
    df = pd.DataFrame(data, columns=columns)
    
    # Print summary statistics
    print(f"Dataset shape: {df.shape}")
    print(f"Class distribution: {df['class1'].value_counts(normalize=True).round(3)}")
    print(f"Number of cows with mastitis: {df.groupby('Cow_ID')['class1'].max().sum()} out of {num_cows}")
    print(f"Mastitis by breed: {df.groupby('Breed')['class1'].mean().round(3)}")
    print(f"Mastitis by previous history: {df.groupby('Previous_Mastits_status')['class1'].mean().round(3)}")
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    
    return df

def analyze_dataset(df):
    """Generate basic analysis and visualizations of the dataset"""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Set up the figure
        plt.figure(figsize=(15, 10))
        
        # 1. Plot temperature distribution by class
        plt.subplot(2, 2, 1)
        sns.boxplot(x='class1', y='Temperature', data=df)
        plt.title('Temperature Distribution by Mastitis Status')
        
        # 2. Plot symptom progression
        plt.subplot(2, 2, 2)
        # Get cows that develop mastitis
        mastitis_cows = df[df['class1'] == 1]['Cow_ID'].unique()
        if len(mastitis_cows) > 0:
            sample_cow = mastitis_cows[0]
            cow_data = df[df['Cow_ID'] == sample_cow].sort_values('Day')
            
            plt.plot(cow_data['Day'], cow_data['Temperature'], 'o-', label='Temperature')
            plt.plot(cow_data['Day'], cow_data['Hardness'], 's-', label='Hardness')
            plt.plot(cow_data['Day'], cow_data['Pain'], '^-', label='Pain')
            plt.plot(cow_data['Day'], cow_data['Milk_visibility'], 'D-', label='Milk Changes')
            plt.plot(cow_data['Day'], cow_data['class1'], '*-', label='Mastitis')
            plt.title(f'Symptom Progression for {sample_cow}')
            plt.legend()
        
        # 3. Plot udder measurements
        plt.subplot(2, 2, 3)
        udder_cols = ['IUFL', 'EUFL', 'IUFR', 'EUFR', 'IURL', 'EURL', 'IURR', 'EURR']
        sns.boxplot(x='class1', y='value', data=pd.melt(df, id_vars=['class1'], value_vars=udder_cols))
        plt.title('Udder Measurements by Mastitis Status')
        
        # 4. Plot risk factors
        plt.subplot(2, 2, 4)
        risk_data = df.groupby(['Previous_Mastits_status', df['Months_after_giving_birth'] > 6])['class1'].mean().reset_index()
        risk_data.columns = ['Previous_Mastitis', 'Older_Cow', 'Mastitis_Rate']
        risk_data['Risk_Group'] = risk_data.apply(
            lambda x: f"{'Previous: Yes' if x['Previous_Mastitis']==1 else 'Previous: No'}, {'Older: Yes' if x['Older_Cow'] else 'Older: No'}", 
            axis=1
        )
        sns.barplot(x='Risk_Group', y='Mastitis_Rate', data=risk_data)
        plt.title('Mastitis Rate by Risk Factors')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('mastitis_analysis.png')
        print("Analysis plot saved as 'mastitis_analysis.png'")
        
    except ImportError:
        print("Matplotlib and/or seaborn not available. Skipping visualizations.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate synthetic cow mastitis data')
    parser.add_argument('--cows', type=int, default=50000, help='Number of cows to generate')
    parser.add_argument('--days', type=int, default=30, help='Days of observation per cow')
    parser.add_argument('--prevalence', type=float, default=0.3, help='Base mastitis prevalence (0-1)')
    parser.add_argument('--output', type=str, default='cow_mastitis_data.csv', help='Output filename')
    parser.add_argument('--analyze', action='store_true', help='Generate analysis plots')
    
    args = parser.parse_args()
    
    print(f"Generating data for {args.cows} cows over {args.days} days with {args.prevalence:.1%} base mastitis prevalence")
    df = generate_realistic_cow_data(
        num_cows=args.cows,
        days_per_cow=args.days,
        mastitis_prevalence=args.prevalence,
        output_file=args.output
    )
    
    if args.analyze:
        analyze_dataset(df)