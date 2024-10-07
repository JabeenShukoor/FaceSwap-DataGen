#importing packages 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from collections import Counter

def visualise(sampled_metadata):
    # Visualization of distributions
    plt.figure(figsize=(12, 6))

    # Gender Distribution
    plt.subplot(1, 2, 1)
    sns.countplot(data=sampled_metadata, x='gender', order=sampled_metadata['gender'].value_counts().index)
    plt.title('Gender Distribution')
    plt.xlabel('Gender')
    plt.ylabel('Count')

    # Race Distribution
    plt.subplot(1, 2, 2)
    sns.countplot(data=sampled_metadata, x='race', order=sampled_metadata['race'].value_counts().index)
    plt.title('Race Distribution')
    plt.xlabel('Race')
    plt.ylabel('Count')

    plt.tight_layout()
    plt.show()

def main(visualise_data = False):
    #Load the metadata from phase 1
    metadata = pd.read_csv("image_metadata.csv")

    #creating a combined group
    metadata['group'] = metadata[['gender', 'race', 'age', 'emotion']].astype(str).agg('_'.join, axis=1)

    # Setting target samples for each category
    target_samples_per_gender = 50  
    target_samples_per_race = 20     

    sampled_metadata = pd.DataFrame()

    # Separate sampling for each gender
    for gender in metadata['gender'].unique():
        gender_samples = metadata[metadata['gender'] == gender]
        
        for race in gender_samples['race'].unique():
            race_samples = gender_samples[gender_samples['race'] == race]
            
            # If there are enough samples for the race, sample from it
            if len(race_samples) >= target_samples_per_race:
                sampled_metadata = pd.concat([sampled_metadata, race_samples.sample(target_samples_per_race, random_state=42)])
            else:
                sampled_metadata = pd.concat([sampled_metadata, race_samples])

    #Remove duplicates after sampling
    sampled_metadata = sampled_metadata.drop_duplicates()

    #Save the sampled metadata to CSV
    sampled_metadata.to_csv("sampled_metadata.csv", index=False)

    if visualise_data:
        visualise(sampled_metadata)

    # Print the final distribution for verification
    print("Final Gender Distribution:", Counter(sampled_metadata['gender']))
    print("Final Race Distribution:", Counter(sampled_metadata['race']))

if __name__ == "__main__":
    main(True)
