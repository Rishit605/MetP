import os
import requests
from bs4 import BeautifulSoup

# Define the URL.
url_to_scrape = "https://www.ncei.noaa.gov/data/global-forecast-system/access/grid-004-0.5-degree/forecast/202310/20231017/"

# Fetch the webpage content
response = requests.get(url_to_scrape)
soup = BeautifulSoup(response.content, "html.parser")

# Find all <a> tags with download links
download_links = soup.find_all("a", href=True)
down_path = 'C:\\Projs\COde\\Earthquake\\earthquake-prediction\\data'
count = 0 

# Iterate through the links and download the files
for link in download_links:
    if count == 1:
        break
    else:
        file_url = link["href"]


        total_url = str(url_to_scrape) + file_url
        # Customize the filename as needed
        filename = os.path.join(down_path, f"{link.text}")

        if total_url.endswith(".grb2"):

            try:
                file_content = requests.get(total_url).content
                with open(filename, "wb") as file:
                    file.write(file_content)
                    count += 1
                print(f"Downloaded: {filename}")
            except Exception as e:
                print(f"Error downloading {total_url}: {e}")

