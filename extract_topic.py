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

    with open ('trainset/Topic/Adult.txt', "r") as f:
        Adult = f.read().splitlines()

    with open ('trainset/Topic/Arts and Entertainment.txt', "r") as f:
        ArtsEntertainment = f.read().splitlines()

    with open ('trainset/Topic/Autos and Vehicles.txt', "r") as f:
        AutosVehicles = f.read().splitlines()

    with open ('trainset/Topic/Beauty and Fitness.txt', "r") as f:
        BeautyFitness = f.read().splitlines()

    with open ('trainset/Topic/Business and Industrial.txt', "r") as f:
        BusinessIndustrial = f.read().splitlines()

    with open ('trainset/Topic/Computers and Electronics.txt', "r") as f:
        ComputersElectronics = f.read().splitlines()

    with open ('trainset/Topic/Finance.txt', "r") as f:
        Finance = f.read().splitlines()

    with open ('trainset/Topic/Food and Drink.txt', "r") as f:
        FoodDrink = f.read().splitlines()

    with open ('trainset/Topic/Games.txt', "r") as f:
        Games = f.read().splitlines()

    with open ('trainset/Topic/Health.txt', "r") as f:
        Health = f.read().splitlines()

    with open ('trainset/Topic/Hobbies and Leisure.txt', "r") as f:
        HobbiesLeisure = f.read().splitlines()

    with open ('trainset/Topic/Home and Garden.txt', "r") as f:
        HomeGarden = f.read().splitlines()

    with open ('trainset/Topic/Internet and Telecom.txt', "r") as f:
        InternetTelecom = f.read().splitlines()

    with open ('trainset/Topic/Jobs and Education.txt', "r") as f:
        JobsEducation = f.read().splitlines()

    with open ('trainset/Topic/Law and Government.txt', "r") as f:
        LawGovernment = f.read().splitlines()

    with open ('trainset/Topic/News.txt', "r") as f:
        News = f.read().splitlines()

    with open ('trainset/Topic/Online Communities.txt', "r") as f:
        OnlineCommunities = f.read().splitlines()

    with open ('trainset/Topic/People and Society.txt', "r") as f:
        PeopleSociety = f.read().splitlines()

    with open ('trainset/Topic/Pets and Animals.txt', "r") as f:
        PetsAnimals = f.read().splitlines()

    with open ('trainset/Topic/Real Estate.txt', "r") as f:
        RealEstate = f.read().splitlines()

    with open ('trainset/Topic/Reference.txt', "r") as f:
        Reference = f.read().splitlines()

    with open ('trainset/Topic/Science.txt', "r") as f:
        Science = f.read().splitlines()

    with open ('trainset/Topic/Sensitive Subjects.txt', "r") as f:
        SensitiveSubjects = f.read().splitlines()

    with open ('trainset/Topic/Shopping.txt', "r") as f:
        Shopping = f.read().splitlines()

    with open ('trainset/Topic/Sports.txt', "r") as f:
        Sports = f.read().splitlines()

    with open ('trainset/Topic/Travel.txt', "r") as f:
        Travel = f.read().splitlines()

        
    print("Topic Classification Training data read!")
    
    data = {}
    data["Adult"] = Adult
    data["Arts and Entertainment"] = ArtsEntertainment
    data["Autos and Vehicles"] = AutosVehicles
    data["Beauty and Fitness"] = BeautyFitness
    data["Business and Industrial"] = BusinessIndustrial
    data["Computers and Electronics"] = ComputersElectronics
    data["Finance"] = Finance
    data["Food and Drink"] = FoodDrink
    data["Games"] = Games
    data["Health"] = Health
    data["Hobbies and Leisure"] = HobbiesLeisure
    data["Home and Garden"] = HomeGarden
    data["Internet and Telecom"] = InternetTelecom
    data["Jobs and Education"] = JobsEducation
    data["Law and Government"] = LawGovernment
    data["News"] = News
    data["Online Communities"] = OnlineCommunities
    data["People and Society"] = PeopleSociety
    data["Pets and Animals"] = PetsAnimals
    data["Real Estate"] = RealEstate
    data["Reference"] = Reference
    data["Science"] = Science
    data["Sensitive Subjects"] = SensitiveSubjects
    data["Shopping"] = Shopping
    data["Sports"] = Sports
    data["Travel"] = Travel

    print("Applying Text Topic Classification Model to :", filename)
    
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
        "Adult",
        "Arts and Entertainment",
        "Autos and Vehicles",
        "Beauty and Fitness",
        "Business and Industrial",
        "Computers and Electronics",
        "Finance",
        "Food and Drink",
        "Games",
        "Health",
        "Hobbies and Leisure",
        "Home and Garden",
        "Internet and Telecom",
        "Jobs and Education",
        "Law and Government",
        "News",
        "Online Communities",
        "People and Society",
        "Pets and Animals",
        "Real Estate",
        "Reference",
        "Science",
        "Sensitive Subjects",
        "Shopping",
        "Sports",
        "Travel"
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
    csv_filename = os.path.join("dataset", "TopicClassification.csv")

    # Create the 'dataset' directory if it doesn't exist
    os.makedirs(os.path.dirname(csv_filename), exist_ok=True)

    print("Creating / Updating Topic CSV file")
          
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
    print("Topic Classification completed!")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python extract_data_type.py <all_text> <filename>")
    else:
        all_text = sys.argv[1]
        filename = sys.argv[2]
        process_text(all_text, filename)

