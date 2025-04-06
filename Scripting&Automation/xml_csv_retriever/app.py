import streamlit as st
import pandas as pd
import requests
import xml.etree.ElementTree as ET
from io import StringIO

def fetch_xml(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.content
    else:
        st.error("Failed to retrieve the XML file.")
        return None

def parse_xml(xml_data):
    # Parse the XML data
    tree = ET.ElementTree(ET.fromstring(xml_data))
    root = tree.getroot()

    # Example: Extracting data from XML
    # This part will depend on the structure of your XML file
    data = []
    for child in root:
        # Assuming each child is an entry you want to convert to a row
        entry = {elem.tag: elem.text for elem in child}
        data.append(entry)

    return pd.DataFrame(data)

def main():
    st.title("XML to CSV Converter")

    # URL input
    url = st.text_input("Enter the URL of the XML file:", "https://dblp.uni-trier.de/xml/")

    if st.button("Convert to CSV"):
        xml_data = fetch_xml(url)
        if xml_data:
            df = parse_xml(xml_data)
            if not df.empty:
                st.write("Data Preview:")
                st.dataframe(df.head())  # Display the first few rows of the DataFrame

                # Show a hint of the DataFrame structure
                st.write("Here's a hint of the extracted data:")
                st.write(df.head())  # This will show the DataFrame in a more detailed format

                # Convert DataFrame to CSV
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="output.csv",
                    mime="text/csv"
                )
            else:
                st.error("No data found in the XML file.")

if __name__ == "__main__":
    main()
