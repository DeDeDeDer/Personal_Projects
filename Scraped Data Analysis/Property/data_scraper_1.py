# Enter Site & Inspect Page
#import requests
from bs4 import BeautifulSoup
import lxml.html as lh
# Data Cleaning
import pandas as pd
import numpy as np
import re
# Create Sub-Folder
import pathlib2
# Read PDF
from tabula import read_pdf
import requests


"""Find Start-End Pagination"""
def start_end_page(url, page_criterion):
    # Raw HTML Source Code
    url = requests.get(url)
    htmltext = url.text
    # Convert to Dataframe format
    htmltext = htmltext.split(page_criterion)     # .split(criterion)
    htmltext = list(htmltext)
    # Filter Search only for ....
    pages_str = []
    for r in range(0, len(htmltext)):
        page_i = htmltext[r].split('"')[0]
        pages_str.append(page_i)
    pages_str = pages_str[1:]
    pages_int = [int(x) for x in pages_str]
    pages_start_end = [min(pages_int), max(pages_int)]
    return pages_start_end


"""Data Scraping Functions"""
def run_area_code_i(area_code_i):
    import requests
    # Sample    'https://www.srx.com.sg/search/sale/residential?selectedHdbTownIds=26&view=table&page=39'
    start_url = 'https://www.srx.com.sg/search/rent/'
    url_pg = start_url + 'residential?selectedHdbTownIds=' + str(area_code_i) + '&view=table'
    # Pages
    page_criterion = '&page='
    start_end_pages = start_end_page(url=url_pg, page_criterion=page_criterion)
    page_ls = list(range(start_end_pages[0], start_end_pages[1] + 1))
    print(page_ls)
    # Date-Frame
    df2 = pd.DataFrame()
    for p in range(0, len(page_ls)):
        page_i = page_ls[p]
        print(page_i)
        url = start_url + 'residential?selectedHdbTownIds=' + str(area_code_i) + '&page=' + str(page_i) + '&view=table'
        print(url)
        # Contents of the website
        page = requests.get(url)
        # Store contents
        doc = lh.fromstring(page.content)
        # Parse data that are stored between <tr>..</tr> of HTML
        tr_elements = doc.xpath('//tr')
        #Create empty list
        col = []
        i = 0
        # For each row, store each first element (header) and an empty list
        for t in tr_elements[0]:
            i += 1
            name=t.text_content()
            print ('%d:"%s"'%(i,name))
            col.append((name,[]))

        # Since out first row is the header, data is stored on the second row onwards
        for j in range(1,len(tr_elements)):
            # T is our j'th row
            T=tr_elements[j]
            # If row is not of size 10, the //tr data is not from our table
            if len(T)!=10:
                break
            # i is the index of our column
            i = 0
            # Iterate through each element of the row
            for t in T.iterchildren():
                data=t.text_content()
                # Check if row is empty
                if i > 0:
                # Convert any numerical value to integers
                    try:
                        data=int(data)
                    except:
                        pass
                # Append the data to the empty list of the i'th column
                col[i][1].append(data)
                # Increment i for the next column
                i+=1

        print([len(C) for (title,C) in col])

        Dict = {title:column for (title,column) in col}
        df = pd.DataFrame(Dict)
        df2 = pd.concat((df, df2), axis=0, sort=False, ignore_index=True)
    df2.to_csv('/Users/Derrick-Vlad-/Desktop/Output_Trail_1/Rentals/' + 'Area_Code_' + str(area_code_i) + '.csv', sep=',')
#run_area_code_i(area_code_i=28)


def run_all():
    all_area_codes = list(range(1, 27, 1))
    for a in range(0, len(all_area_codes)):
        area_code_i = all_area_codes[a]
        print('Running Area Code: ', str(area_code_i))
        run_area_code_i(area_code_i=area_code_i)



