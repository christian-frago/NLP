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

    with open ('trainset/Roles/Accountants.txt', "r") as f:
        Accountants = f.read().splitlines()

    with open ('trainset/Roles/Artists.txt', "r") as f:
        Artists = f.read().splitlines()

    with open ('trainset/Roles/Books and Literature.txt', "r") as f:
        BooksLiterature = f.read().splitlines()

    with open ('trainset/Roles/Business Owners.txt', "r") as f:
        BusinessOwners = f.read().splitlines()

    with open ('trainset/Roles/Charitable Trusts and NGOs.txt', "r") as f:
        CharitableTrustsNGOs = f.read().splitlines()

    with open ('trainset/Roles/Consumer Services.txt', "r") as f:
        ConsumerServices = f.read().splitlines()

    with open ('trainset/Roles/Counselors.txt', "r") as f:
        Counselors = f.read().splitlines()

    with open ('trainset/Roles/Doctors.txt', "r") as f:
        Doctors = f.read().splitlines()

    with open ('trainset/Roles/Estate Agents.txt', "r") as f:
        EstateAgents = f.read().splitlines()

    with open ('trainset/Roles/Family Offices.txt', "r") as f:
        FamilyOffices = f.read().splitlines()

    with open ('trainset/Roles/Financial Advisors.txt', "r") as f:
        FinancialAdvisors = f.read().splitlines()

    with open ('trainset/Roles/Franchise Operations.txt', "r") as f:
        FranchiseOperations = f.read().splitlines()

    with open ('trainset/Roles/Goods Suppliers.txt', "r") as f:
        GoodsSuppliers = f.read().splitlines()

    with open ('trainset/Roles/Governments.txt', "r") as f:
        Governments = f.read().splitlines()

    with open ('trainset/Roles/Hospitals.txt', "r") as f:
        Hospitals = f.read().splitlines()

    with open ('trainset/Roles/Independent Consultants.txt', "r") as f:
        IndependentConsultants = f.read().splitlines()

    with open ('trainset/Roles/Individuals.txt', "r") as f:
        Individuals = f.read().splitlines()

    with open ('trainset/Roles/Intergovernmental and Regulators.txt', "r") as f:
        IntergovernmentalRegulators = f.read().splitlines()

    with open ('trainset/Roles/International Institutions.txt', "r") as f:
        InternationalInstitutions = f.read().splitlines()

    with open ('trainset/Roles/Lawyers.txt', "r") as f:
        Lawyers = f.read().splitlines()

    with open ('trainset/Roles/Manufacturers.txt', "r") as f:
        Manufacturers = f.read().splitlines()

    with open ('trainset/Roles/Multi Nationals.txt', "r") as f:
        MultiNationals = f.read().splitlines()

    with open ('trainset/Roles/Municipalities and Cities.txt', "r") as f:
        MunicipalitiesCities = f.read().splitlines()

    with open ('trainset/Roles/Online ReSellers.txt', "r") as f:
        OnlineReSellers = f.read().splitlines()

    with open ('trainset/Roles/Organizations.txt', "r") as f:
        Organizations = f.read().splitlines()

    with open ('trainset/Roles/Parents.txt', "r") as f:
        Parents = f.read().splitlines()

    with open ('trainset/Roles/Philanthropies and Foundations.txt', "r") as f:
        PhilanthropiesFoundations = f.read().splitlines()

    with open ('trainset/Roles/Politicians.txt', "r") as f:
        Politicians = f.read().splitlines()

    with open ('trainset/Roles/Private Corporations.txt', "r") as f:
        PrivateCorporations = f.read().splitlines()

    with open ('trainset/Roles/Professionals.txt', "r") as f:
        Professionals = f.read().splitlines()

    with open ('trainset/Roles/Public Companies.txt', "r") as f:
        PublicCompanies = f.read().splitlines()

    with open ('trainset/Roles/Regional Governments.txt', "r") as f:
        RegionalGovernments = f.read().splitlines()

    with open ('trainset/Roles/Restaurants.txt', "r") as f:
        Restaurants = f.read().splitlines()

    with open ('trainset/Roles/Retail Shops.txt', "r") as f:
        RetailShops = f.read().splitlines()

    with open ('trainset/Roles/Seniors.txt', "r") as f:
        Seniors = f.read().splitlines()

    with open ('trainset/Roles/Service Vendors.txt', "r") as f:
        ServiceVendors = f.read().splitlines()

    with open ('trainset/Roles/Singles.txt', "r") as f:
        Singles = f.read().splitlines()

    with open ('trainset/Roles/Sports Persons.txt', "r") as f:
        SportsPersons = f.read().splitlines()

    with open ('trainset/Roles/Students.txt', "r") as f:
        Students = f.read().splitlines()

    with open ('trainset/Roles/Universities.txt', "r") as f:
        Universities = f.read().splitlines()
        
    print("Role Classification Training data read!")
    
    data = {}
    data["Accountants"] = Accountants
    data["Artists"] = Artists
    data["Books and Literature"] = BooksLiterature
    data["Business Owners"] = BusinessOwners
    data["Charitable Trusts and NGOs"] = CharitableTrustsNGOs
    data["Consumer Services"] = ConsumerServices
    data["Counselors"] = Counselors
    data["Doctors"] = Doctors
    data["Estate Agents"] = EstateAgents
    data["Family Offices"] = FamilyOffices
    data["Financial Advisors"] = FinancialAdvisors
    data["Franchise Operations"] = FranchiseOperations
    data["Goods Suppliers"] = GoodsSuppliers
    data["Governments"] = Governments
    data["Hospitals"] = Hospitals
    data["Independent Consultants"] = IndependentConsultants
    data["Individuals"] = Individuals
    data["Intergovernmental and Regulators"] = IntergovernmentalRegulators
    data["International Institutions"] = InternationalInstitutions
    data["Lawyers"] = Lawyers
    data["Manufacturers"] = Manufacturers
    data["Multi Nationals"] = MultiNationals
    data["Municipalities and Cities"] = MunicipalitiesCities
    data["Online Re-Sellers"] = OnlineReSellers
    data["Organizations"] = Organizations
    data["Parents"] = Parents
    data["Philanthropies and Foundations"] = PhilanthropiesFoundations
    data["Politicians"] = Politicians
    data["Private Corporations"] = PrivateCorporations
    data["Professionals"] = Professionals
    data["Public Companies"] = PublicCompanies
    data["Regional Governments"] = RegionalGovernments
    data["Restaurants"] = Restaurants
    data["Retail Shops"] = RetailShops
    data["Seniors"] = Seniors
    data["Service Vendors"] = ServiceVendors
    data["Singles"] = Singles
    data["Sports Persons"] = SportsPersons
    data["Students"] = Students
    data["Universities"] = Universities

    print("Applying Text Role Classification Model to :", filename)
    
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
        "Accountants",
        "Artists",
        "Books and Literature",
        "Business Owners",
        "Charitable Trusts and NGOs",
        "Consumer Services",
        "Counselors",
        "Doctors",
        "Estate Agents",
        "Family Offices",
        "Financial Advisors",
        "Franchise Operations",
        "Goods Suppliers",
        "Governments",
        "Hospitals",
        "Independent Consultants",
        "Individuals",
        "Intergovernmental and Regulators",
        "International Institutions",
        "Lawyers",
        "Manufacturers",
        "Multi Nationals",
        "Municipalities and Cities",
        "Online Re-Sellers",
        "Organizations",
        "Parents",
        "Philanthropies and Foundations",
        "Politicians",
        "Private Corporations",
        "Professionals",
        "Public Companies",
        "Regional Governments",
        "Restaurants",
        "Retail Shops",
        "Seniors",
        "Service Vendors",
        "Singles",
        "Sports Persons",
        "Students",
        "Universities"
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
    csv_filename = os.path.join("dataset", "RoleClassification.csv")

    # Create the 'dataset' directory if it doesn't exist
    os.makedirs(os.path.dirname(csv_filename), exist_ok=True)

    print("Creating / Updating Role Classification CSV file")
          
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
    print("Role Classification completed!")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python extract_data_type.py <all_text> <filename>")
    else:
        all_text = sys.argv[1]
        filename = sys.argv[2]
        process_text(all_text, filename)

