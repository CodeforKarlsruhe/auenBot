import bs4
import requests
import json 
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import os
import time
from urllib.parse import urlparse, unquote
import re
from html import escape


demo_url = "https://commons.wikimedia.org/wiki/File:Abeille_charpentiere_1024.jpg"

opts = Options()
opts.headless = True
opts.add_argument("--disable-gpu")
opts.add_argument("--no-sandbox")
os.environ["PATH"] = "/usr/bin:" + os.environ.get("PATH", "")
driver = webdriver.Chrome(service=Service("/usr/bin/chromedriver"), options=opts)

# https://github.com/juncture-digital/iiif/blob/5d9ea0528249a6db391a99d5dc0594f492c74df3/presentation-api/src/handlers/wikimedia_commons.py
def get_license_from_structured_data(filename):
    """
    Use Wikibase/Structured Data API for reliable license extraction
    """
    api_url = "https://commons.wikimedia.org/w/api.php"
    
    print(f"Getting structured data for file: {filename}")

    # Step 1: Get the entity ID (M-ID) for the file
    params = {
        "action": "query",
        "titles": f"File:{filename}",
        "prop": "imageinfo", # "info|imageinfo",
        "iiprop": "extmetadata", # "url|size|extmetadata",
        "format": "json"
    }

    headers = {
        "User-Agent": "AuenBot/1.0 (https://auenlaend.ok-lab-karlsruhe.de; info@ok-lab-karlsruhe.de)",
        "Accept": "application/json",
        "Accept-Language": "en-US,en;q=0.9"
    }
    
    response = requests.get(api_url, params=params, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Wikimedia API request failed with status {response.status_code}")
    response = response.json()
    print(f"Wikimedia API response: {response}")
    
    # Extract page info
    pages = response.get("query", {}).get("pages", {})
    if not pages:
        return None
    page = list(pages.values())[0]

    extmetadata = page.get("imageinfo", [{}])[0].get("extmetadata", {})

    # Helper to get and clean an HTML-ish value
    def _clean_meta(field):
        v = extmetadata.get(field, {}).get("value", "")
        if not v:
            return ""
        return bs4.BeautifulSoup(v, "html.parser").get_text().strip()

    artist_text = _clean_meta("Artist") or _clean_meta("Credit") or "Unknown"
    license_url = extmetadata.get("LicenseUrl", {}).get("value", "") or ""
    # pick first available license-like field
    license_field = None
    license_name = ""
    for f in ["LicenseShortName", "Permission", "UsageTerms"]:
        val = _clean_meta(f)
        if val:
            license_field = f
            license_name = val
            break

    # alt text: prefer ImageDescription (cleaned), fallback to page title or filename
    alt_text = _clean_meta("ImageDescription") or page.get("title", "") or filename

    # Build title attribute
    title_parts = [p for p in [artist_text, license_name] if p]
    title_main = ", ".join(title_parts) if title_parts else artist_text
    if license_url:
        title_main = f"{title_main} <{license_url}>"
    title_attr = f"{title_main}, via Wikimedia Commons"
    title_attr = escape(title_attr, quote=True)

    # Build image src using Special:FilePath to get a 512px resized image
    img_src = f"https://commons.wikimedia.org/wiki/Special:FilePath/{filename}?width=512"

    file_page = f"https://commons.wikimedia.org/wiki/File:{filename}"
    alt_attr = escape(alt_text, quote=True)

    html_snippet = f'<a title="{title_attr}" href="{file_page}"><img width="512" alt="{alt_attr}" src="{img_src}"></a>'

    # Return assembled info
    return {
        "license_html": html_snippet,
        "source": "structured_data_api",
        "artist": artist_text,
        "license_name": license_name,
        "license_url": license_url,
        "license_field": license_field
    }



def process_url(url):
    print(f"Processing URL: {url}")
    driver.get(url)
    time.sleep(2)  # wait for page to load
    wait = WebDriverWait(driver, 15)

    btn = wait.until(
        EC.element_to_be_clickable(
            (By.CSS_SELECTOR, "a.stockphoto_buttonrow[title='Use this file on the web'][role='button']")
        )
    )
    btn.click()

    ta = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "textarea#stockphoto_html")))
    inner_html = ta.get_attribute("value")

    # if you want to parse the extracted HTML with BeautifulSoup:
    parsed = bs4.BeautifulSoup(inner_html, "html.parser")

    return inner_html, parsed


def getPageInfo(img):
# request like this:https://commons.wikimedia.org/w/api.php?action=query&prop=fileusage&titles=File:Abeille_charpentiere_1024.jpg&format=json
# return: {"batchcomplete":"","query":{"normalized":[{"from":"File:Abeille_charpentiere_1024.jpg","to":"File:Abeille charpentiere 1024.jpg"}],"pages":{"1353124":{"pageid":1353124,"ns":6,"title":"File:Abeille_charpentiere_1024.jpg","fileusage":[{"pageid":1273216,"ns":2,"title":"User:SeeSchloss"},{"pageid":6861583,"ns":14,"title":"Category:Xylocopa violacea"}]}}}}
# take to first normalized "to" and use it with prefix https://commons.wikimedia.org/wiki/File: to create the url
    """
    Extract the raw filename (including extension) from a Wikimedia image URL,
    stripping any size-prefix like '330px-' from thumb URLs.

    Examples:
    - .../thumb/.../Abeille_charpentiere_1024.jpg/330px-Abeille_charpentiere_1024.jpg
      -> "Abeille_charpentiere_1024.jpg"
    - .../wikipedia/commons/c/c6/Abeille_charpentiere_1024.jpg
      -> "Abeille_charpentiere_1024.jpg"
    """

    if not img:
        return None

    parsed = urlparse(img)
    path = unquote(parsed.path or "")
    segments = [s for s in path.split("/") if s]

    # For "thumb" URLs the original filename is typically the second-to-last segment.
    if "thumb" in segments and len(segments) >= 2:
        candidate = segments[-2]
    else:
        candidate = segments[-1] if segments else None

    if not candidate:
        return None

    # Remove size prefix like "330px-" if present (sometimes the filename itself appears in the last segment)
    m = re.match(r"^\d+px-(.+)$", candidate)
    if m:
        candidate = m.group(1)

    return candidate


        

def process_all():
    with open("../rawData/tiere_pflanzen_auen.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Processing {len(data)} entries")

    for entry in data:
        name = entry.get("Name", "Unknown")
        print(f"Processing entry: {name}")
        links = entry.get("Links", [])
        for link in links:
            img_url = link.get("img")
            if img_url and "wikimedia.org" in img_url:
                # check if license info already exists
                if "license_html" in link:
                    print(f"License info already exists for {name}, skipping.")
                    continue
                filename = getPageInfo(img_url)
                print(f"Extracted filename: {filename}")
                if not filename:
                    continue
                try:
                    struct_data = get_license_from_structured_data(filename)
                    print(f"Structured data license info: {struct_data}")
                    if struct_data:
                        link["license_structured_data"] = struct_data
                except Exception as e:
                    print(f"Error fetching structured data for {filename}: {e}")
                    pass
                target_url = f"https://commons.wikimedia.org/wiki/File:{filename}"
                try:
                    html,parsed = process_url(target_url)
                    print(f"Inner HTML for {name}:\n{html}\n")
                    link["license_html"] = html
                    link["license_parsed"] = str(parsed)
                except Exception as e:
                    print(f"Error processing {target_url}: {e}")
                    pass

                time.sleep(1)  # Be polite and avoid overwhelming the server

    with open("../rawData/tiere_pflanzen_auen_with_licenses.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)    


try:
    inner_html, parsed = process_url(demo_url)
    print(inner_html)
    print(parsed.prettify())
    time.sleep(2)
    print("\n\nProcessing all entries from JSON...\n\n")
    process_all()

finally:
    driver.quit()



