#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
import spacy
import classy_classification
import pandas as pd
import os

def process_text(all_text, filename):
    #Open and Read all Training Data

    with open ('trainset/Datatype/ArtisticProcess.txt', "r") as f:
        ArtisticProcess = f.read().splitlines()

    with open ('trainset/Datatype/BestSellingBooks.txt', "r") as f:
        BestSellingBooks = f.read().splitlines()

    with open ('trainset/Datatype/BlueprintsToolkit.txt', "r") as f:
        BlueprintsToolkit = f.read().splitlines()

    with open ('trainset/Datatype/BusinessModels.txt', "r") as f:
        BusinessModels = f.read().splitlines()

    with open ('trainset/Datatype/DiscoveryInvention.txt', "r") as f:
        DiscoveryInvention = f.read().splitlines()

    with open ('trainset/Datatype/ExpertAdviceRecommentations.txt', "r") as f:
        ExpertAdviceRecommentations = f.read().splitlines()

    with open ('trainset/Datatype/GovernanceStructure.txt', "r") as f:
        GovernanceStructure = f.read().splitlines()

    with open ('trainset/Datatype/HealthRegimes.txt', "r") as f:
        HealthRegimes = f.read().splitlines()

    with open ('trainset/Datatype/IndustryStandards.txt', "r") as f:
        IndustryStandards = f.read().splitlines()

    with open ('trainset/Datatype/Lifestyle.txt', "r") as f:
        Lifestyle = f.read().splitlines()

    with open ('trainset/Datatype/OperationsManual.txt', "r") as f:
        OperationsManual = f.read().splitlines()

    with open ('trainset/Datatype/PhilosopyValues.txt', "r") as f:
        PhilosopyValues = f.read().splitlines()

    with open ('trainset/Datatype/ProductionMethod.txt', "r") as f:
        ProductionMethod = f.read().splitlines()

    with open ('trainset/Datatype/Regulations.txt', "r") as f:
        Regulations = f.read().splitlines()

    with open ('trainset/Datatype/SolutionsPlaybooks.txt', "r") as f:
        SolutionsPlaybooks = f.read().splitlines()

    with open ('trainset/Datatype/Strategies.txt', "r") as f:
        Strategies = f.read().splitlines()

        
    print("Data Type Classification Training data read!")
    
    data = {}
    data["Artistic Process"] = ArtisticProcess
    data["Best Selling Books"] = BestSellingBooks
    data["Blueprints Toolkit"] = BlueprintsToolkit
    data["Business Models"] = BusinessModels
    data["Discovery Invention"] = DiscoveryInvention
    data["Expert Advice Recommentations"] = ExpertAdviceRecommentations
    data["Governance Structure"] = GovernanceStructure
    data["Health Regimes"] = HealthRegimes
    data["Industry Standards"] = IndustryStandards
    data["Lifestyle"] = Lifestyle
    data["Operations Manual"] = OperationsManual
    data["Philosopy Values"] = PhilosopyValues
    data["Production Method"] = ProductionMethod
    data["Regulations"] = Regulations
    data["Solutions Playbooks"] = SolutionsPlaybooks
    data["Strategies"] = Strategies

    print("Applying Text Data Type Classification Model to :", filename)
    
    #Apply Classy Model
    nlp = spacy.blank("en")
    nlp.add_pipe(
        "text_categorizer", 
        config={
            "data": data, 
            "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "device": "cpu"
        }
    ) 
    
    print("Applying sentencizer!")
    sentence_model = spacy.blank("en")
    sentence_model.add_pipe("sentencizer")
    
    print("Segmentation begin:")
    segment_size = 100000  # Define the desired segment size

    num_segments = len(all_text) // segment_size + 1  # Calculate the number of segments

    final_data = []

    for i in range(num_segments):
        start_idx = i * segment_size
        end_idx = (i + 1) * segment_size
        segment = all_text[start_idx:end_idx]

        sentences = sentence_model(segment)

        for sentence in sentences.sents:
            doc = nlp(sentence.text)
            final_data.append({"sentence": doc.text, "cats": doc._.cats})

    # Extract the "sentence" and "cats" data from the final_data list, and trim spaces
    sentences = [item["sentence"].strip() for item in final_data]
    categories_data = [item["cats"] for item in final_data]

    # Create the DataFrame with the "Sentence" column
    df_cat = pd.DataFrame({"Sentence": sentences})

    print("Composing the dataframe!")
          
    # Add the category columns to the DataFrame
    category_columns = [
        "Artistic Process",
        "Best Selling Books",
        "Blueprints Toolkit",
        "Business Models",
        "Discovery Invention",
        "Expert Advice Recommentations",
        "Governance Structure",
        "Health Regimes",
        "Industry Standards",
        "Lifestyle",
        "Operations Manual",
        "Philosopy Values",
        "Production Method",
        "Regulations",
        "Solutions Playbooks",
        "Strategies"
    ]

    for category in category_columns:
        df_cat[category] = [data[category] for data in categories_data]

    # Convert the probability values to percentages and round to two decimal places
    df_cat[category_columns] = (df_cat[category_columns] * 100).round(6)

    # Add a new column with the category_column name that has the highest value
    df_cat['Category_Tag'] = df_cat[category_columns].idxmax(axis=1)

    # Add a new column for the Filename
    df_cat['Filename'] = filename

    # Generate the CSV file path
    csv_filename = os.path.join("dataset", "DataTypeClassification.csv")

    # Create the 'dataset' directory if it doesn't exist
    os.makedirs(os.path.dirname(csv_filename), exist_ok=True)

    print("Creating / Updating Data Type CSV file")
          
    # Check if the CSV file already exists
    if os.path.exists(csv_filename):
        # Load the existing CSV file into a DataFrame
        df_existing = pd.read_csv(csv_filename)

        # Append the new data to the existing DataFrame
        df_combined = pd.concat([df_existing, df_cat], ignore_index=True)
    else:
        # If the CSV file doesn't exist, just save the new DataFrame directly
        df_combined = df_cat

    # Save the combined DataFrame to the CSV file
    df_combined.to_csv(csv_filename, index=False)
    print("Data Type Classification completed!")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python extract_data_type.py <all_text> <filename>")
    else:
        all_text = sys.argv[1]
        filename = sys.argv[2]
        process_text(all_text, filename)

