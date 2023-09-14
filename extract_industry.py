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

    with open ('trainset/Industry/Accommodation Food Services.txt', "r") as f:
        AccommodationFoodService = f.read().splitlines()

    with open ('trainset/Industry/Agriculture Forestry Fishing.txt', "r") as f:
        AgricultureForestryFishing = f.read().splitlines()

    with open ('trainset/Industry/Arts Entertainment Recreation.txt', "r") as f:
        ArtsEntertainmentRecreation = f.read().splitlines()

    with open ('trainset/Industry/Economywide.txt', "r") as f:
        Economywide = f.read().splitlines()

    with open ('trainset/Industry/Educational Services.txt', "r") as f:
        EducationalServices = f.read().splitlines()

    with open ('trainset/Industry/Health Care Social Assistance.txt', "r") as f:
        HealthCareSocialAssistance = f.read().splitlines()

    with open ('trainset/Industry/Information Finance Insurance.txt', "r") as f:
        InformationFinanceInsurance = f.read().splitlines()

    with open ('trainset/Industry/Manufacturing.txt', "r") as f:
        Manufacturing = f.read().splitlines()

    with open ('trainset/Industry/Mining Quarrying Oil Gas.txt', "r") as f:
        MiningQuarryingOilGas = f.read().splitlines()

    with open ('trainset/Industry/Public Administration.txt', "r") as f:
        PublicAdministration = f.read().splitlines()

    with open ('trainset/Industry/Real Estate Rental Leasing.txt', "r") as f:
        RealEstateRentalLeasing = f.read().splitlines()

    with open ('trainset/Industry/Technical Management Services.txt', "r") as f:
        TechnicalManagementServices = f.read().splitlines()

    with open ('trainset/Industry/Transportation Warehousing.txt', "r") as f:
        TransportationWarehousing = f.read().splitlines()

    with open ('trainset/Industry/Utilities Construction.txt', "r") as f:
        UtilitiesConstruction = f.read().splitlines()

    with open ('trainset/Industry/WholeSale RetailTrade.txt', "r") as f:
        WholeSaleRetailTrade = f.read().splitlines()

        
    print("Industry Classification Training data read!")
    
    data = {}
    data["Accommodation Food Services"] = AccommodationFoodService
    data["Agriculture Forestry Fishing"] = AgricultureForestryFishing
    data["Arts Entertainment Recreation"] = ArtsEntertainmentRecreation
    data["Economywide"] = Economywide
    data["Educational Services"] = EducationalServices
    data["Health Care Social Assistance"] = HealthCareSocialAssistance
    data["Information Finance Insurance"] = InformationFinanceInsurance
    data["Manufacturing"] = Manufacturing
    data["Mining Quarrying Oil Gas"] = MiningQuarryingOilGas
    data["Public Administration"] = PublicAdministration
    data["Real Estate Rental Leasing"] = RealEstateRentalLeasing
    data["Technical Management Services"] = TechnicalManagementServices
    data["Transportation Warehousing"] = TransportationWarehousing
    data["Utilities Construction"] = UtilitiesConstruction
    data["WholeSale Retail Trade"] = WholeSaleRetailTrade

    print("Applying Industry Text Classification Model to :", filename)
    
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
        "Accommodation Food Services",
        "Agriculture Forestry Fishing",
        "Arts Entertainment Recreation",
        "Economywide",
        "Educational Services",
        "Health Care Social Assistance",
        "Information Finance Insurance",
        "Manufacturing",
        "Mining Quarrying Oil Gas",
        "Public Administration",
        "Real Estate Rental Leasing",
        "Technical Management Services",
        "Transportation Warehousing",
        "Utilities Construction",
        "WholeSale Retail Trade"
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
    csv_filename = os.path.join("dataset", "IndustryClassification.csv")

    # Create the 'dataset' directory if it doesn't exist
    os.makedirs(os.path.dirname(csv_filename), exist_ok=True)

    print("Creating / Updating Industry Classification CSV file")
          
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
    print("Industry Classification completed!")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python extract_data_type.py <all_text> <filename>")
    else:
        all_text = sys.argv[1]
        filename = sys.argv[2]
        process_text(all_text, filename)

