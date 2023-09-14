#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import requests
import PyPDF2
import os
import subprocess

def main():
    if len(sys.argv) != 2:
        print("Usage: python pdf_text_extractor.py <PDF_URL>")
        return

    file_url = sys.argv[1]
    filename = download_pdf(file_url)
    print(filename, " downloaded")

    # Open PDF file
    pdffileObj = open(filename, 'rb')

    # Create PDF reader object
    pdfReader = PyPDF2.PdfReader(pdffileObj)

    # Get the total number of pages in the PDF
    num_pages = len(pdfReader.pages)
    print("Total number of pages:", num_pages)

    # Initialize an empty variable to store the extracted text
    all_text = ""
    
    # Loop through each page and extract text
    for page_num in range(num_pages):
        # Create a page object for the current page
        pageObj = pdfReader.pages[page_num]
        
        # Extract text from the page
        text = pageObj.extract_text()
        
        # Append the extracted text to the 'all_text' variable
        all_text += text

    pdffileObj.close()

    # Save the extracted text to a temporary file with UTF-8 encoding
    temp_text_filename = "temp_text.txt"
    with open(temp_text_filename, "w", encoding="utf-8") as temp_file:
        temp_file.write(all_text)

    # Get the absolute path to the script directory
    script_directory = os.path.dirname(os.path.abspath(__file__))

    # Construct absolute paths to the scripts
    extract_data_script = os.path.join(script_directory, "extract_data_type.py")
    extract_industry_script = os.path.join(script_directory, "extract_industry.py")
    extract_role_script = os.path.join(script_directory, "extract_role.py")
    extract_topic_script = os.path.join(script_directory, "extract_topic.py")

    # Call the scripts with absolute paths
    subprocess.run(["python", extract_data_script, temp_text_filename, filename])
    subprocess.run(["python", extract_industry_script, temp_text_filename, filename])
    subprocess.run(["python", extract_role_script, temp_text_filename, filename])
    subprocess.run(["python", extract_topic_script, temp_text_filename, filename])

    # Clean up the temporary text file
    os.remove(temp_text_filename)
    
#Downloading the pdf file
def download_pdf(url):
    
    # Get the actual filename from the URL
    filename = os.path.basename(url)

    if '.pdf' in filename:
        filename = filename[:filename.index('.pdf') + 4]
    
    #Create the pdf file locally
    r = requests.get(url, stream=True)

    #Access the pdf
    with open(filename, "wb") as pdf:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                pdf.write(chunk)

    return filename

if __name__ == "__main__":
    main()

