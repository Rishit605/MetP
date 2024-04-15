### US GFS FORECASTS ###S

# url = 'https://www.ncei.noaa.gov/data/global-forecast-system/access/grid-004-0.5-degree/forecast/'

# import requests
# import json

# # Connect to the COVID19-India API
# api_url = 'https://www.ncei.noaa.gov/access/services/data/v1?dataset=daily-summaries'
# response = requests.get(api_url)

# # Check if the response is successful (status code 200)
# if response.status_code == 200:
#     data = response.text
#     parsed_data = json.loads(data)
#     print(data)
#     # Example: Extract active cases in South Andaman
# #     active_cases_south_andaman = parsed_data['Andaman and Nicobar Islands']['districtData']['South Andaman']['active']
# #     print(f"Active cases in South Andaman: {active_cases_south_andaman}")
# else:
#     print("Error fetching data from the API")

import os
import requests
from bs4 import BeautifulSoup

# Example URL to scrape (replace with your target URL)
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
            # print(total_url)
            try:
                file_content = requests.get(total_url).content
                with open(filename, "wb") as file:
                    file.write(file_content)
                    count += 1
                print(f"Downloaded: {filename}")
            except Exception as e:
                print(f"Error downloading {total_url}: {e}")

