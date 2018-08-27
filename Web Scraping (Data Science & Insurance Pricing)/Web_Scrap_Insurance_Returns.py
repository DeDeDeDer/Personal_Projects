"""
Web Scrap Kernel
"""

# Reference Site: http://www.mas.gov.sg/Statistics/Insurance-Statistics/Insurance-Company-Returns.aspx

# Enter Site & Inspect Page
import requests
from bs4 import BeautifulSoup
# Data Cleaning
import pandas as pd
import numpy as np
# Create Sub-Folder
import pathlib2
# Read PDF
from tabula import read_pdf
# Visual Settings
pd.set_option("display.max_columns", 30)

"""Stage 1:
A. Get URL Main-Links
B. Get PDF Sub-Links
C. Archive Links into CSV export"""


"""A. Get URL Main-links from main page"""
# Part 1
URL = 'http://www.mas.gov.sg/Statistics/Insurance-Statistics/Insurance-Company-Returns.aspx'
r = requests.get(URL)
soup = BeautifulSoup(r.content)
RawPage_bs = soup.find_all(["a", "p"])
RawPage_df = pd.DataFrame(columns=['URL', 'Text'])
i = 0
for n in RawPage_bs:
    A = n.get("href")
    RawPage_df.loc[i, 'URL'] = A
    B = n.text
    RawPage_df.loc[i, 'Text'] = B
    i += 1
# Part 2
LinksCriterion = '/Statistics/Insurance-Statistics/Insurance-Company-Returns/'
LinksAll = pd.DataFrame(columns=['URL',
                                 'Code', 'Name',
                                 '2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009',
                                 '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018'])
j = 0
for row in range(0, len(RawPage_df)):
    URL = RawPage_df.loc[row, 'URL']
    try:
        if str(LinksCriterion) in str(URL):
            Code = RawPage_df.loc[row, 'Text']
            CompanyName = RawPage_df.loc[row + 1, 'Text']
            LinksAll.loc[j, 'URL'] = URL
            LinksAll.loc[j, 'Code'] = Code
            LinksAll.loc[j, 'Name'] = CompanyName
            j += 1
    except ValueError:
        pass

"""B. Get PDF Sub-links from respective page"""
for link in range(0, len(LinksAll)):
    # PART 1
    URL_2 = 'http://www.mas.gov.sg' + LinksAll.loc[link, 'URL']
    r_2 = requests.get(URL_2)
    soup_2 = BeautifulSoup(r_2.content)
    RawPage_bs_2 = soup_2.find_all(["a"])
    RawPage_df_2 = pd.DataFrame(columns=['URL', 'Text'])
    i = 0
    for n in RawPage_bs_2:
        A_2 = n.get("href")
        RawPage_df_2.loc[i, 'URL'] = A_2
        B_2 = n.text
        RawPage_df_2.loc[i, 'Text'] = B_2
        i += 1
    # PART 2
    LinksCriterion_2 = '.pdf'
    k = 0
    for row in range(0, len(RawPage_df_2)):
        URL_i = RawPage_df_2.loc[row, 'URL']
        try:
            if str(LinksCriterion_2) in str(URL_i):
                Year = RawPage_df_2.loc[row, 'Text']
                LinksAll.loc[link, str(Year)] = URL_i
                k += 1
        except ValueError:
            pass

"""C. Export to CSV"""
path_0 = '/Users/Derrick-Vlad-/Desktop/Singapore Insurance Statistics/'
LinksAll.to_csv(path_0+'SG_Insurance_PDF_Links.csv', sep=',')

##########################################################################################


"""Stage 2:
A. Download & Store PDFs in respective new folder"""

"""A. Download & Store PDFs"""
Years = ['2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009',
         '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018']
path_0 = '/Users/Derrick-Vlad-/Desktop/Singapore Insurance Statistics/'
LinksAll = pd.read_csv(path_0+'SG_Insurance_PDF_Links.csv')
Link_start = 'http://www.mas.gov.sg'
LinksCriterion_2 = '.pdf'
for row in range(0, len(LinksAll)):
    Code_i = LinksAll.loc[row, 'Code']
    for yr in range(0, len(Years)):
        Year_i = Years[yr]
        PDF_Link = LinksAll.loc[row, str(Year_i)]
        if str(LinksCriterion_2) in str(PDF_Link):
            # Create Sub-Company-Folder
            pathlib2.Path(path_0 + '/' + str(Code_i)).mkdir(exist_ok=True)
            # Save PDF in Sub-Company-Folder
            r = requests.get(Link_start + PDF_Link, stream=True)
            with open(path_0 + '/' + str(Code_i) + '/' + str(Year_i) + ".pdf", "wb") as pdf:
                for chunk in r.iter_content(chunk_size=1024):
                    # writing one chunk at a time to pdf file
                    if chunk:
                        pdf.write(chunk)
            print("Completed: ", Year_i, PDF_Link)
        else:
            print("Left Out: ", Year_i, PDF_Link)

##########################################################################################


"""Stage 3:
A. Convert PDF Table into CSV & Execute preliminary Data Cleaning
B. Manual Checks"""


def clean_1_split_data(g):
    # Split Data
    g = g.applymap(str)
    g = g.dropna(how='all')
    g = g.loc[:, ~(g.astype(str) == ' ').all()]
    columns = g.columns
    for col in range(2, len(columns)):
        column_i = columns[col]
        g = pd.concat([g, g[column_i].str.split(' ', 1, expand=True)], axis=1, sort=False)
    g = g.drop(columns=columns[2:len(columns)])
    g = g.iloc[:, 1:]
    dummy_cols = ["Col_0", "Col_1", "Col_2", "Col_3", "Col_4", "Col_5", "Col_6", "Col_7", "Col_8", "Col_9", "Col_10",
                  "Col_11", "Col_12", "Col_13", "Col_14", "Col_15", "Col_16", "Col_17", "Col_18", "Col_19", "Col_20",
                  "Col_21", "Col_22", "Col_23", "Col_24", "Col_25", "Col_26", "Col_27", "Col_28", "Col_29", "Col_30",
                  "Col_31", "Col_32", "Col_33", "Col_34", "Col_35", "Col_36", "Col_37", "Col_38", "Col_39", "Col_40",
                  "Col_41", "Col_42", "Col_43", "Col_44", "Col_45", "Col_46", "Col_47", "Col_48", "Col_49", "Col_50",
                  "Col_51", "Col_52", "Col_53", "Col_54", "Col_55", "Col_56", "Col_57", "Col_58", "Col_59", "Col_60"]
    g.columns = dummy_cols[0:len(g.columns)]
    g.applymap(str)
    g = g.loc[:, ~(g.astype(str) == 'nan').all()]
    g = g.replace('nan', '')
    g.applymap(str)
    g = g.dropna(axis='columns', how='all')
    # Remove redundant columns
    g = g.replace('', np.nan)
    unnamed_cols = [c for c in g.columns if 'Unnamed' in c]
    g = g.dropna(subset=unnamed_cols, how='all')
    return g


def clean_2_squeeze_nan(y):
    original_columns = y.index.tolist()
    squeezed = y.dropna()
    squeezed.index = [original_columns[n] for n in range(squeezed.count())]
    return squeezed.reindex(original_columns, fill_value=np.nan)


def clean_3_nan_to_blanks(z):
    z = z.replace(np.nan, '')
    return z


"""A. Convert PDF Table into CSV & Clean Table alignments"""
Years = ['2012', '2013', '2014', '2015', '2016', '2017']    # Limit to reduce computational cost
path_0 = '/Users/Derrick-Vlad-/Desktop/Singapore Insurance Statistics/'
AllLinks_excel = pd.read_csv(path_0+'SG_Insurance_PDF_Links.csv', encoding="ISO-8859-1")
LinksCriterion_2 = '.pdf'
for row in range(0, len(AllLinks_excel)):
    Code_i = AllLinks_excel.loc[row, 'Code']
    for yr in range(0, len(Years)):
        Year_i = Years[yr]
        PDF_Link = AllLinks_excel.loc[row, str(Year_i)]
        if str(LinksCriterion_2) in str(PDF_Link):

            # # Part 3.1
            # Sub-Company-Folder
            CompanyFolder = path_0 + str(Code_i)
            # Create Sub-Company CSV Folder
            pathlib2.Path(CompanyFolder + '/CSV').mkdir(exist_ok=True)
            SubCompanyCSVFolder = CompanyFolder + '/CSV'
            # Read & Clean Each page
            df_0 = read_pdf(CompanyFolder + '/' + Year_i + '.pdf', pages=51)
            df_0 = clean_1_split_data(df_0).apply(clean_2_squeeze_nan, axis=1)

            df_1 = read_pdf(CompanyFolder + '/' + Year_i + '.pdf', pages=52)
            df_1 = clean_1_split_data(df_1).apply(clean_2_squeeze_nan, axis=1)

            df_2 = read_pdf(CompanyFolder + '/' + Year_i + '.pdf', pages=53)
            df_2 = clean_1_split_data(df_2).apply(clean_2_squeeze_nan, axis=1)

            df_3 = read_pdf(CompanyFolder + '/' + Year_i + '.pdf', pages=54)
            df_3 = clean_1_split_data(df_3).apply(clean_2_squeeze_nan, axis=1)

            df_4 = read_pdf(CompanyFolder + '/' + Year_i + '.pdf', pages=55)
            df_4 = clean_1_split_data(df_4).apply(clean_2_squeeze_nan, axis=1)

            df_5 = read_pdf(CompanyFolder + '/' + Year_i + '.pdf', pages=56)
            df_5 = clean_1_split_data(df_5).apply(clean_2_squeeze_nan, axis=1)

            df_6 = read_pdf(CompanyFolder + '/' + Year_i + '.pdf', pages=57)
            df_6 = clean_1_split_data(df_6).apply(clean_2_squeeze_nan, axis=1)

            df_all = [df_0, df_1, df_2, df_3, df_4, df_5, df_6]

            # # Part 3.2
            # Set Criteria
            RowCriterion_1 = 'Description'
            RowCriterion_2 = 'A. PREMIUMS'
            ColumnCriterion_1 = ['Fire', 'Motor', 'Work Injury Compensation', 'Personal Accident', 'Health',
                                 'Misc - Public Liability', 'Misc - Bonds']
            ColumnCriterion_2 = ['Description', 'Row No.', 'Property', 'Casualty and Others']
            for df in range(0, len(df_all)):
                df_i = df_all[df]
                if df_i is None:
                    continue
                column_i = df_i.columns
                if any(s in l for l in column_i for s in ColumnCriterion_1):
                    Part_A = df_i
                    Part_A_index = df
                    Part_B = df_all[Part_A_index + 1]
                    Part_B_row = Part_B.iloc[:, 0]
                    Part_B_column = Part_B.columns
                    if (RowCriterion_1 or RowCriterion_2 in Part_B_row) \
                            or (any(s in l for l in Part_B_column for s in ColumnCriterion_2)):
                        clean_3_nan_to_blanks(Part_A)
                        # Export to CSV
                        Part_A.to_csv(SubCompanyCSVFolder + '/' + Year_i + '_Original_A' + '.csv', sep=',')
                        print('A' + Year_i)
                    else:
                        # Concat PartA & B
                        Part_AB = pd.concat([Part_A, Part_B.reset_index(drop=True, inplace=True)], axis=1)
                        clean_3_nan_to_blanks(Part_AB)
                        # Export to CSV
                        Part_AB.to_csv(SubCompanyCSVFolder + '/' + Year_i + '_Original_AB' + '.csv', sep=',')
                        print('AB' + Year_i)
                    break
        else:
            x = 1  # Do nothing

##########################################################################################


"""Stage 4:
A. Collate All Data"""

"""Compile Data"""
# Part A
path_0 = '/Users/Derrick-Vlad-/Desktop/Singapore Insurance Statistics/'
Selected = pd.read_csv('/Users/Derrick-Vlad-/Desktop/'+'Selected.csv', encoding="ISO-8859-1")
CollateData = pd.DataFrame()
Years = ['2012', '2013', '2014', '2015', '2016', '2017']
for n in range(0, len(Selected)):
    code_i = Selected.loc[n, 'Code']
    CSV_Folder_i = path_0 + code_i + '/CSV'
    try:
        for yr in range(0, len(Years)):
            year_i = Years[yr]
            CSV_File_i = CSV_Folder_i + '/' + year_i + '.csv'
            CSV_i = pd.read_csv(CSV_File_i, encoding="ISO-8859-1")
            CSV_i = CSV_i.drop(columns=['Unnamed: 0', 'Row No.'], axis=1)
            CSV_i = CSV_i.transpose()
            CSV_i['Year'] = year_i
            CSV_i['Code'] = code_i
            CollateData = CollateData.append(CSV_i)
            print('Completed: ', code_i)
    except FileNotFoundError:
        print('Check code: ' + code_i)

# Part B
Original_columns = ['Description', 'A. PREMIUMS', 'Gross premiums', 'Direct business',
                    'Reinsurance business accepted -',
                    'In Singapore', 'From other ASEAN countries', 'From other countries', 'Total (2 to 4)',
                    'Reinsurance business ceded -', 'In Singapore', 'To other ASEAN countries', 'To other countries',
                    'Total (6 to 8)', 'Net premiums written (1 + 5 - 9)', 'Premium liabilities at beginning of period',
                    'Premium liabilities at end of period', 'Premiums earned during the period (10 + 11 - 12)',
                    'B. CLAIMS', 'Gross claims settled', 'Direct business',
                    'Reinsurance business accepted -', 'In Singapore', 'From other ASEAN countries',
                    'From other countries', 'Total (15 to 17)', 'Recoveries from reinsurance business ceded -',
                    'In Singapore', 'To other ASEAN countries', 'To other countries', 'Total (19 to 21)',
                    'Net claims settled (14 + 18 - 22)', 'Claims liabilities at end of period',
                    'Claims liabilities at beginning of period', 'Net claims incurred (23 + 24 - 25)',
                    'C. MANAGEMENT EXPENSES', 'Management Expenses', 'D. DISTRIBUTION EXPENSES',
                    'Commissions', 'Reinsurance commissions', 'Net commissions incurred (28 - 29)',
                    'Other distribution expenses', 'E. UNDERWRITING RESULTS',
                    'Underwriting gain / (loss) (13 - 26 - 27 - 30 - 31)', 'F. NET INVESTMENT INCOME',
                    'G. OPERATING RESULT (32 + 33)', 'Year', 'Code']
CollateData.reset_index(level=0, inplace=True)
CollateData.columns = Original_columns
CollateData = CollateData[CollateData.Description.str.contains("Description") == False].reset_index(drop=True)
CollateData.to_csv('/Users/Derrick-Vlad-/Desktop/'+'SG_Insurance_BigData.csv', sep=',')

##########################################################################################


"""CODE DUMP"""







































