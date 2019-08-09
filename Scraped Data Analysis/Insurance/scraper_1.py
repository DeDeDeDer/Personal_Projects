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
# Others
import requests
import io


"""
///////////////////////////////////////////////////////////////
GET HTML SOURCE CODE///////////////////////////////////////////
///////////////////////////////////////////////////////////////
"""
path_0 = '/Users/Derrick-Vlad-/Desktop/'

def html_source_df(url):
    # Raw HTML Source Code
    url = requests.get(url)
    htmltext = url.text
    # Convert to Dataframe format
    htmltext = htmltext.split('<')  # '\n'
    htmltext = list(htmltext)
    df = pd.DataFrame(htmltext)
    df.columns = ['FirstCol']
    return df


def compile_pdf_links(ref_link, criterion):
    # Establish Reference
    reference = pd.read_csv(ref_link)
    Link = reference['Full-Link']
    Code = reference['Code']
    # Set New df
    new_df = pd.DataFrame()
    c = []
    p = []
    for r in range(0, len(reference)):
        # Loop through each Link
        code_i = Code.iloc[r]
        link_i = Link.iloc[r]
        # Open & Convert
        df = html_source_df(url=link_i)
        # Filter Search only PDF links
        pdf_links = df[df['FirstCol'].str.contains(criterion)]
        pdf_links = list(pdf_links['FirstCol'])
        # Multiple codes for list
        count = len(pdf_links)
        codes = [code_i]*count
        # Impute Extend into List
        c.extend(codes)
        p.extend(pdf_links)

    # Compile into dataframe
    new_df['Code'] = c
    new_df['PDF_links'] = p
    new_df.columns = ['Code', 'PDF_links']
    # Output
    new_df.to_csv(path_0+'Compiled PDF Links.csv', sep=',')
#compile_pdf_links(ref_link=path_0+'Consolidated/REVAMP/revamp_ref.csv', criterion=".pdf")


"""
///////////////////////////////////////////////////////////////
Download PDFs//////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
"""
def download_pdfs(store_link, ref_link):
    # Establish Reference
    reference = pd.read_csv(ref_link)
    Link = reference['Full-PDF_links']
    Code = reference['Code']
    Year = reference['Year']
    for r in range(0, len(reference)):
        # Loop through each Link
        code_i = Code.iloc[r]
        link_i = Link.iloc[r]
        year_i = Year.iloc[r]
        # Create Sub-Company-Folder
        pathlib2.Path(store_link + '/PDFs/' + str(code_i)).mkdir(exist_ok=True)
        # Save PDF in Sub-Company-Folder
        r = requests.get(link_i, stream=True)
        with open(store_link + '/PDFs/' + str(code_i) + '/' + str(year_i) + ".pdf", "wb") as pdf:
            for chunk in r.iter_content(chunk_size=1024):
                # writing one chunk at a time to pdf file
                if chunk:
                    pdf.write(chunk)
#download_pdfs(store_link=path_0+'Consolidated/REVAMP', ref_link=path_0+'Consolidated/REVAMP/Compiled PDF lInks ORGANISED.csv')
#download_pdfs(store_link=path_0+'Consolidated/REVAMP', ref_link=path_0+'Consolidated/REVAMP/ad-Hoc_1.csv')





"""
///////////////////////////////////////////////////////////////
PDF to CSV/////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
"""
from Cleaning_Functions import *


def pdf_to_csv(store_link, ref_link):
    # Establish Reference
    reference = pd.read_csv(ref_link)
    Page = reference['Pages']
    Page2 = reference['Pages2'].astype(str)
    Overlap = reference['Overlap']
    Code = reference['Code']
    Year = reference['Year']
    endcount = len(reference)#744,745,746,747     749,750,751,752,753,754,755        769.770       792-796
    for r in range(795, len(reference)):
        # Loop through each reference
        code_i = Code.iloc[r]
        page_i = Page.iloc[r]
        page2_i = Page2.iloc[r]
        year_i = Year.iloc[r]
        overlap_i = Overlap.iloc[r]
        where_i = str(store_link) + str(code_i) + '/' + str(year_i)
        if page_i == 'ERROR':
             pass
        # For Normal Cases
        elif overlap_i == 9999:
            # Read PDF - PART 1
            df_1 = read_pdf(where_i + '.pdf', pages=page_i)
            df_1 = df_1.apply(clean_2_squeeze_nan, axis=1)
            # Convert PDF to CSV
            df_1.to_csv(where_i + '.csv', sep=',', encoding='utf-8-sig')
        # For Overlap cases
        elif overlap_i == 1:
            # Read PDF - PART 1
            df_1 = read_pdf(where_i + '.pdf', pages=page_i)
            df_1 = df_1.apply(clean_2_squeeze_nan, axis=1)
            # Convert PDF to CSV
            df_1.to_csv(where_i + '.csv', sep=',', encoding='utf-8-sig')

            # Read PDF - PART 2
            df_2 = read_pdf(where_i + '.pdf', pages=page2_i)
            df_2 = df_2.apply(clean_2_squeeze_nan, axis=1)
            # Convert PDF to CSV
            df_2.to_csv(where_i + '_Part_2.csv', sep=',', encoding='utf-8-sig')
        else:
            pass
        # Progress Report
        running = 'Running ' + str(code_i) + ' ' + str(year_i)
        ratio = str(r) + ' out of ' + str(endcount) + ' Completed'
        print(running)
        print(ratio)
    print('Completed')
#pdf_to_csv(store_link='/Users/Derrick-Vlad-/Desktop/Consolidated/REVAMP/PDFs/', ref_link=path_0+'Consolidated/REVAMP/Compiled PDF lInks ORGANISED.csv')




"""
///////////////////////////////////////////////////////////////
Clean CSV//////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
"""
def clean_csv(store_link, ref_link):
    # Establish Reference
    reference = pd.read_csv(ref_link)
    Overlap = reference['Overlap']
    Code = reference['Code']
    Year = reference['Year']
    Page = reference['Pages']
    endcount = len(reference)#744,745,746,747     749,750,751,752,753,754,755        769.770       792-796
    for r in range(150, 743):
        # Loop through each reference
        code_i = Code.iloc[r]
        year_i = Year.iloc[r]
        overlap_i = Overlap.iloc[r]
        page_i = Page.iloc[r]
        where_i = str(store_link) + str(code_i) + '/' + str(year_i)
        if page_i == 'ERROR':
             pass
        else:
            # Read CSV
            df_1 = pd.read_csv(where_i + '.csv')
            df_1 = df_1.apply(clean_2_squeeze_nan, axis=1)
            df_1 = clean_1_split_data(df_1).apply(clean_2_squeeze_nan, axis=1)
            df_1.to_csv(where_i + '.csv', sep=',', encoding='utf-8-sig')

        # Progress Report
        running = 'Running ' + str(code_i) + ' ' + str(year_i)
        ratio = str(r) + ' out of ' + str(endcount) + ' Completed'
        print(running)
        print(ratio)
    print('Completed')


# clean_csv(store_link='/Users/Derrick-Vlad-/Desktop/Consolidated/REVAMP/PDFs/',
#           ref_link=path_0+'Consolidated/REVAMP/Compiled PDF lInks ORGANISED.csv')




"""
///////////////////////////////////////////////////////////////
CSV Stack per Insurer & Neaten/////////////////////////////////
///////////////////////////////////////////////////////////////
"""

def create_dummy_cols_list(count):
    items = []
    list1 = ['col_'] * count
    list2 = list(range(0, count))
    list2 = [str(i) for i in list2]
    for i in range(0, len(list1)):
        item = list1[i] + list2[i]
        items.append(item)
    return items


def all_collate_csv_2(source):
    import glob
    import os
    # Get list of files in directory
    directory = source
    ls_all = [x[0] for x in os.walk(directory)]
    ls_folders = ls_all[1:]
    for r in range(0, len(ls_folders)):
        # Current loop folder
        where_i = ls_folders[r]
        print('Running:  ' + where_i)
        # Current Code
        code_i = ls_folders[r][-5:]
        # Establish all ~.csv~ files
        allFiles_0 = glob.glob(where_i + "/*.csv")
        allFiles = [x for x in allFiles_0 if not '_Part_2' in x]
        allFiles = [x for x in allFiles if not 'Compiled' in x]
        allFiles = [x for x in allFiles if not 'Sample' in x]
        # Establish Storage Data-frame
        stockstats_data = pd.DataFrame()
        # Loop each ~.csv~ file & Concatenate
        for f in range(0, len(allFiles)):
            file_i = allFiles[f]
            year_i = allFiles[f][-8:-4]
            df = pd.read_csv(file_i, encoding='unicode_escape')
            cnt = len(df.columns)
            df = df.iloc[:, 0:cnt]
            actuall_col = ['Description', 'Row No.', 'Marine and Aviation - Cargo',
                          'Marine and Aviation - Hull and Liability', 'Fire', 'Motor',
                          'Work Injury Compensation', 'Personal Accident', 'Health',
                          'Misc - Public Liability', 'Misc - Bonds', 'Misc - Engineering / CAR / EAR',
                          'Misc - Professional Indemnity', 'Misc - Credit / Political Risk',
                          'Misc - Others', 'Misc - Sub-Total', 'Total']
            ext_cols = create_dummy_cols_list(200)
            new_col = actuall_col + ext_cols
            df.columns = new_col[0:cnt]
            df['Code'] = [code_i] * len(df)
            df['Year'] = [year_i] * len(df)
            df = df.replace('', 'nan')
            df = df.dropna(axis='columns', how='all')
            df = df.replace('nan', '')

            stockstats_data = pd.concat((df, stockstats_data), axis=0, sort=False, ignore_index=True)
            stockstats_data.to_csv(path_0 + 'CompileBackTest/' + code_i + '_Compiled_4.csv', sep=',')
        print('Completed:  ' + where_i)
#all_collate_csv_2(source=path_0 + 'Consolidated/REVAMP/PDFs')


def single_collate_csv_2(code):
    import glob
    where_i = path_0+'Consolidated/REVAMP' + '/PDFs/' + code
    # Establish all ~.csv~ files
    allFiles_ = glob.glob(where_i + "/*.csv")
    allFiles = [x for x in allFiles_ if not '_Part_2' in x]
    # Establish Storage Data-frame
    stockstats_data = pd.DataFrame()

    # Loop each ~.csv~ file & Concatenate
    for f in range(0,len(allFiles)):
        file_i = allFiles[f]
        df = pd.read_csv(file_i,encoding = 'unicode_escape')    #.transpose()
        df = df.iloc[:, 0:17]
        df.columns = ['Description', 'Row No.', 'Marine and Aviation - Cargo',
                      'Marine and Aviation - Hull and Liability', 'Fire', 'Motor',
                      'Work Injury Compensation', 'Personal Accident', 'Health',
                      'Misc - Public Liability', 'Misc - Bonds', 'Misc - Engineering / CAR / EAR',
                      'Misc - Professional Indemnity', 'Misc - Credit / Political Risk',
                      'Misc - Others', 'Misc - Sub-Total', 'Total']

        df['Code'] = [code]*len(df)
        year_i = allFiles[f][-8:-4]
        df['Year'] = [year_i] * len(df)
        stockstats_data = pd.concat((df, stockstats_data), axis=0, sort=False, ignore_index=True)  # , ignore_index=True
        stockstats_data.to_csv(where_i + '/Compiled.csv', sep=',', encoding='utf-8-sig')
#single_collate_csv_2(code='I518C')



def part2_collate_csv(source):
    import glob
    import os
    # Get list of files in directory
    directory = source
    ls_all = [x[0] for x in os.walk(directory)]
    ls_folders = ls_all[1:]
    for r in range(0, len(ls_folders)):
        # Current loop folder
        where_i = ls_folders[r]
        print('Running:  ' + where_i)
        # Current Code
        code_i = ls_folders[r][-5:]
        # Establish all ~.csv~ files
        allFiles_0 = glob.glob(where_i + "/*.csv")
        allFiles = [x for x in allFiles_0 if '_Part_2' in x]
        # Establish Storage Data-frame
        stockstats_data = pd.DataFrame()
        # Loop each ~.csv~ file & Concatenate
        for f in range(0, len(allFiles)):
            file_i = allFiles[f]
            year_i = allFiles[f][-15:-11]
            df = pd.read_csv(file_i, encoding='unicode_escape')

            cnt = len(df.columns)
            df = df.iloc[:, 0:cnt]
            actuall_col = ['Description', 'Row No.', 'Marine and Aviation - Cargo',
                          'Marine and Aviation - Hull and Liability', 'Fire', 'Motor',
                          'Work Injury Compensation', 'Personal Accident', 'Health',
                          'Misc - Public Liability', 'Misc - Bonds', 'Misc - Engineering / CAR / EAR',
                          'Misc - Professional Indemnity', 'Misc - Credit / Political Risk',
                          'Misc - Others', 'Misc - Sub-Total', 'Total']
            ext_cols = create_dummy_cols_list(200)
            new_col = actuall_col + ext_cols
            df.columns = new_col[0:cnt]
            df['Code'] = [code_i] * len(df)
            df['Year'] = [year_i] * len(df)
            df = df.replace('', 'nan')
            df = df.dropna(axis='columns', how='all')
            df = df.replace('nan', '')

            stockstats_data = pd.concat((df, stockstats_data), axis=0, sort=False, ignore_index=True)
            stockstats_data.to_csv(path_0 + 'part_2_backtest4/' + code_i + '_part_2_Compiled_4.csv', sep=',')
        print('Completed:  ' + where_i)
#part2_collate_csv(source=path_0 + 'Consolidated/REVAMP/PDFs')




"""
///////////////////////////////////////////////////////////////
Merge All CSV/////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
"""

def trial_merger_all(source):
    import glob
    import os
    # Get list of files in directory
    directory = source
    ls_all = [x[0] for x in os.walk(directory)]
    ls_folders = ls_all[1:]
    # Establish Storage Data-frame
    stockstats_data = pd.DataFrame()
    for r in range(0, len(ls_folders)):
        # Current loop folder
        where_i = ls_folders[r]
        # Current Code
        # Establish all ~.csv~ files
        allFiles_0 = glob.glob(where_i + "/*.csv")
        allFiles = [x for x in allFiles_0 if 'Compiled' in x]

        try:
            link = str(allFiles[0])
            df = pd.read_csv(link, encoding='unicode_escape')
            stockstats_data = pd.concat((df, stockstats_data), axis=0, sort=False)  # , ignore_index=True
            #stockstats_data.to_csv('/Users/Derrick-Vlad-/Desktop/' + 'all_data_Test_1.csv', sep=',')
        except IndexError:
            pass
    stockstats_data.to_csv('/Users/Derrick-Vlad-/Desktop/' + 'all_data_Test_2.csv', sep=',')
#trial_merger_all(source=path_0 + 'Consolidated/REVAMP/PDFs')


def merger_all_2(source):
    import glob
    import os
    # Establish all ~.csv~ files
    allFiles = glob.glob(source + "/*.csv")
    print(allFiles)
    # Establish Storage Data-frame
    stockstats_data = pd.DataFrame()
    for r in range(0, len(allFiles)):
        try:
            link = str(allFiles[r])
            df = pd.read_csv(link, encoding='unicode_escape')
            stockstats_data = pd.concat((df, stockstats_data), axis=0, sort=False, ignore_index=True)
        except IndexError:
            pass
    stockstats_data.to_csv('/Users/Derrick-Vlad-/Desktop/' + 'all_data_Test_4.csv', sep=',')
#merger_all_2(source=path_0 + 'CompileBackTest/')



def part_2_merger_all_2(source):
    import glob
    import os
    # Establish all ~.csv~ files
    allFiles = glob.glob(source + "/*.csv")
    print(allFiles)
    # Establish Storage Data-frame
    stockstats_data = pd.DataFrame()
    for r in range(0, len(allFiles)):
        try:
            link = str(allFiles[r])
            df = pd.read_csv(link, encoding='unicode_escape')
            stockstats_data = pd.concat((df, stockstats_data), axis=0, sort=False, ignore_index=True)
        except IndexError:
            pass
    stockstats_data.to_csv('/Users/Derrick-Vlad-/Desktop/' + 'all_data_Test_4_part_2.csv', sep=',')
#part_2_merger_all_2(source=path_0 + 'part_2_backtest4/')



"""
///////////////////////////////////////////////////////////////
Misc AFTER MERGE ALL///////////////////////////////////////////
///////////////////////////////////////////////////////////////
"""
def misc_1(source):
    """
    Filter for Gross Claims & Gross Premiums
    :param source: URL Location for collated data
    :return: Binary (1/0) indicator
    """
    compare = ['B. CLAIMS', 'Gross claims settled', 'Direct business']
    #compare = ['A. PREMIUMS', 'Gross premiums', 'Direct business']
    #compare = ['E. UNDERWRITING RESULTS', 'Underwriting gain / (loss) (13 - 26 - 27 - 30 - 31)', 'F. NET INVESTMENT INCOME', 'G. OPERATING RESULT (32 + 33) 34']
    df = pd.read_csv(source)
    indicator = [0] * len(df)
    for r in range(3, len(df)):
        start = int(r-3)
        end = r
        tags = list(df.iloc[start:end]['Description'].astype(str))
        a = tags[0].find(compare[0])
        b = tags[1].find(compare[1])
        c = tags[2].find(compare[2])
        sum = a+b+c
        if sum==0:
            indicator[r] = 1
        else:
            #indicator.append(0)
            pass
    print(indicator)
    ind = pd.DataFrame()
    ind['indicate'] = indicator
    ind.to_csv('/Users/Derrick-Vlad-/Desktop/' + 'claim_mask_2.csv', sep=',')
#misc_1('/Users/Derrick-Vlad-/Desktop/' + 'CompileBackTest_4/All_Data_2.csv')


"""Other Functions"""
def misc_2(source):
    """
    Filter for Operating Result
    :param source: URL Location for collated data
    :return: Binary (1/0) indicator
    """
    compare = ['G. OPERATING RESULT (32', 'G. OPERATING RESULT (32 +', 'G. OPERATING RESULT (32 + 33)', 'G. OPERATING RESULT (32 + 33) 34']
    df = pd.read_csv(source)
    indicator = [0] * len(df)
    for r in range(0, len(df)):
        tags = list(df['Description'].astype(str))
        a = tags[r].find(compare[0])
        b = tags[r].find(compare[1])
        c = tags[r].find(compare[2])
        d = tags[r].find(compare[3])
        sum = a + b + c + d
        if sum!=-1:
            indicator[r] = 0
        else:
            indicator[r] = 1

    print(indicator)
    ind = pd.DataFrame()
    ind['indicate'] = indicator
    ind.to_csv('/Users/Derrick-Vlad-/Desktop/' + 'opert_result_mask_2.csv', sep=',')
    print(ind.sum())
#misc_2('/Users/Derrick-Vlad-/Desktop/' + 'CompileBackTest_4/All_Data.csv')


def _1_split_by_value(source):
    # Read Data
    df = pd.read_csv(source)
    # Columns
    actual_col = ['Description', 'Row No.', 'Marine and Aviation - Cargo',
                  'Marine and Aviation - Hull and Liability', 'Fire', 'Motor',
                  'Work Injury Compensation', 'Personal Accident', 'Health',
                  'Misc - Public Liability', 'Misc - Bonds', 'Misc - Engineering / CAR / EAR',
                  'Misc - Professional Indemnity', 'Misc - Credit / Political Risk',
                  'Misc - Others', 'Misc - Sub-Total', 'Total']
    simple_ref_col = ['Description', 'Row No.', 'Marine and Aviation - Cargo',
                      'Marine and Aviation - Hull and Liability', 'Fire', 'Motor',
                      'Work Injury Compensation', 'Personal Accident', 'Health',
                      'Misc - Sub-Total', 'Total']

    simple_col = ['Description', 'Row No.', 'Cargo', 'Hull', 'Fire', 'Motor', 'WorkInjury', 'PersonalAccident',
                  'Health', "Misc", 'Total']
    gp_col = actual_col + ['Gross Premiums']
    gc_col = actual_col + ['Gross Claims']
    op_col = actual_col + ['Operating Result']
    # Individual DataFrame
    gp = df.loc[df['Gross Premiums'] == 1][gp_col]
    gc = df.loc[df['Gross Claims'] == 1][gc_col]
    op = df.loc[df['Operating Result'] == 1][op_col]
    # Simplify column names
    gp[simple_ref_col].columns = simple_col
    gc[simple_ref_col].columns = simple_col
    op[simple_ref_col].columns = simple_col
    # Prepare Data to Merge
    gp['Value'] = ['GrossPremium'] * len(gp)
    gc['Value'] = ['GrossClaim'] * len(gc)
    op['Value'] = ['OperatingResult'] * len(op)
    return gp, gc, op
# gp, gc, op = _1_split_by_value('/Users/Derrick-Vlad-/Desktop/' + 'CompileBackTest_4/All_Data.csv')


def _2_split_by_type(df_list):
    simple_col = ['Description', 'Row No.', 'Cargo', 'Hull', 'Fire', 'Motor', 'WorkInjury', 'PersonalAccident',
                  'Health', "Misc", 'Total']
    types = ['Cargo', 'Hull', 'Fire', 'Motor', 'WorkInjury', 'PersonalAccident', 'Health', "Misc", 'Total']
    universal_df = pd.DataFrame()

    cargo_gp, cargo_gc, cargo_op = df_list[0]['Cargo'], df_list[1]['Cargo'], df_list[2]['Cargo']
    hull_gp, hull_gc, hull_op = df_list[0]['Hull'], df_list[1]['Hull'], df_list[2]['Hull']
    fire_gp, fire_gc, fire_op = df_list[0]['Fire'], df_list[1]['Fire'], df_list[2]['Fire']
    motor_gp, motor_gc, motor_op = df_list[0]['Motor'], df_list[1]['Motor'], df_list[2]['Motor']
    wi_gp, wi_gc, wi_op = df_list[0]['WorkInjury'], df_list[1]['WorkInjury'], df_list[2]['WorkInjury']
    pa_gp, pa_gc, pa_op = df_list[0]['PersonalAccident'], df_list[1]['PersonalAccident'], df_list[2]['PersonalAccident']
    health_gp, health_gc, health_op = df_list[0]['Health'], df_list[1]['Health'], df_list[2]['Health']
    misc_gp, misc_gc, misc_op = df_list[0]['Misc'], df_list[1]['Misc'], df_list[2]['Misc']
    total_gp, total_gc, total_op = df_list[0]['Total'], df_list[1]['Total'], df_list[2]['Total']

    all_1 = [
        [cargo_gp, cargo_gc, cargo_op],
        [hull_gp, hull_gc, hull_op],
        [fire_gp, fire_gc, fire_op],
        [motor_gp, motor_gc, motor_op],
        [wi_gp, wi_gc, wi_op],
        [pa_gp, pa_gc, pa_op],
        [health_gp, health_gc, health_op],
        [misc_gp, misc_gc, misc_op],
        [total_gp, total_gc, total_op]
    ]

    for t in range(0, len(types)):
        type_i = types[t]
        df_i = all_1[t]
        all_1[t][0]['Type'] = [type_i] * len(df_i)
        all_1[t][1]['Type'] = [type_i] * len(df_i)
        all_1[t][2]['Type'] = [type_i] * len(df_i)


    all_2 = [
        cargo_gp, cargo_gc, cargo_op,
        hull_gp, hull_gc, hull_op,
        fire_gp, fire_gc, fire_op,
        motor_gp, motor_gc, motor_op,
        wi_gp, wi_gc, wi_op,
        pa_gp, pa_gc, pa_op,
        health_gp, health_gc, health_op,
        misc_gp, misc_gc, misc_op,
        total_gp, total_gc, total_op,
    ]

    for d in range(0, len(all_2)):
        df_i = all_2[d][['Value', 'Type', 'Code', 'Year']]
        universal_df = pd.concat((df_i, universal_df), axis=0, sort=False, ignore_index=True)


#_2_split_by_type(df_list=[gp, gc, op])
