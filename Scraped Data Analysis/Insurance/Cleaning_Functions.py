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


"""Set 2"""


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
    new_colll = ['Description', 'Row No.', 'Marine and Aviation - Cargo', 'Marine and Aviation - Hull and Liability',
                 'Fire', 'Motor', 'Work Injury Compensation', 'Personal Accident', 'Health',
                 'Misc - Public Liability', 'Misc - Bonds', 'Misc - Engineering / CAR / EAR',
                 'Misc - Professional Indemnity', 'Misc - Credit / Political Risk', 'Misc - Others', 'Misc - Sub-Total',
                 'Total',
                 "Col_0", "Col_1", "Col_2", "Col_3", "Col_4", "Col_5", "Col_6", "Col_7", "Col_8", "Col_9", "Col_10",
                 "Col_11", "Col_12", "Col_13", "Col_14", "Col_15", "Col_16", "Col_17", "Col_18", "Col_19", "Col_20",
                 "Col_21", "Col_22", "Col_23", "Col_24", "Col_25", "Col_26", "Col_27", "Col_28", "Col_29", "Col_30",
                 "Col_31", "Col_32", "Col_33", "Col_34", "Col_35", "Col_36", "Col_37", "Col_38", "Col_39", "Col_40",
                 "Col_41", "Col_42", "Col_43", "Col_44", "Col_45", "Col_46", "Col_47", "Col_48", "Col_49", "Col_50",
                 "Col_51", "Col_52", "Col_53", "Col_54", "Col_55", "Col_56", "Col_57", "Col_58", "Col_59", "Col_60",
                 "Col_61", "Col_62", "Col_63", "Col_64", "Col_65", "Col_66", "Col_67", "Col_68", "Col_69", "Col_70",
                 "Col_71", "Col_72", "Col_73", "Col_74", "Col_75", "Col_76", "Col_77", "Col_78", "Col_79", "Col_80"

                 ]

    dummy_cols = ["Col_0", "Col_1", "Col_2", "Col_3", "Col_4", "Col_5", "Col_6", "Col_7", "Col_8", "Col_9", "Col_10",
                  "Col_11", "Col_12", "Col_13", "Col_14", "Col_15", "Col_16", "Col_17", "Col_18", "Col_19", "Col_20",
                  "Col_21", "Col_22", "Col_23", "Col_24", "Col_25", "Col_26", "Col_27", "Col_28", "Col_29", "Col_30",
                  "Col_31", "Col_32", "Col_33", "Col_34", "Col_35", "Col_36", "Col_37", "Col_38", "Col_39", "Col_40",
                  "Col_41", "Col_42", "Col_43", "Col_44", "Col_45", "Col_46", "Col_47", "Col_48", "Col_49", "Col_50",
                  "Col_51", "Col_52", "Col_53", "Col_54", "Col_55", "Col_56", "Col_57", "Col_58", "Col_59", "Col_60"]
    g.columns = new_colll[0:len(g.columns)]
    #g.columns = dummy_cols[0:len(g.columns)]
    g.applymap(str)
    g = g.loc[:, ~(g.astype(str) == 'nan').all()]
    g = g.replace('nan', '')
    g.applymap(str)
    #g = g.dropna(axis='columns', how='all')
    # Remove redundant columns
    #g = g.replace('', np.nan)
    unnamed_cols = [c for c in g.columns if 'Unnamed' in c]
    #g = g.dropna(subset=unnamed_cols, how='all')
    return g


def clean_2_squeeze_nan(y):
    original_columns = y.index.tolist()
    squeezed = y.dropna()
    squeezed.index = [original_columns[n] for n in range(squeezed.count())]
    return squeezed.reindex(original_columns, fill_value=np.nan)


def clean_3_nan_to_blanks(z):
    z = str(z)
    z = z.replace(np.nan, '')
    return z


def clean_4_shape_to_dash(z):
    if isinstance(z, np.float64) or isinstance(z, np.int64):
        z = z
    else:
        z = z.replace('â€“', '-')
    return z


def clean_5_shape_to_dash(sequence, old, new):
    return (new if x in old else x for x in sequence)


def clean_6_shape_to_dash(z):
    if isinstance(z, np.float64) or isinstance(z, np.int64):
        z = z
    else:
        z = z.split()
        for i in z:
            if z[i] in '€':
                z[i] = '-'
            else:
                continue
    return z


def clean_text(text):
    unwanted_char = '\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\x0c\r\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f\x7f\x80\x81\x82\x83\x84\x85\x86\x87\x88\x89\x8a\x8b\x8c\x8d\x8e\x8f\x90\x91\x92\x93\x94\x95\x96\x97\x98\x99\x9a\x9b\x9c\x9d\x9e\x9f\xa0\xa1\xa2\xa3\xa4\xa5\xa6\xa7\xa8\xa9\xaa\xab\xac\xad\xae\xaf\xb0\xb1\xb2\xb3\xb4\xb5\xb6\xb7\xb8\xb9\xba\xbb\xbc\xbd\xbe\xbf\xc0\xc1\xc2\xc3\xc4\xc5\xc6\xc7\xc8\xc9\xca\xcb\xcc\xcd\xce\xcf\xd0\xd1\xd2\xd3\xd4\xd5\xd6\xd7\xd8\xd9\xda\xdb\xdc\xdd\xde\xdf\xe0\xe1\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9\xea\xeb\xec\xed\xee\xef\xf0\xf1\xf2\xf3\xf4\xf5\xf6\xf7\xf8\xf9\xfa\xfb\xfc\xfd\xfe\xff'
    text = "".join([(" " if n in unwanted_char else n) for n in text if n not in unwanted_char])
    return text


def clean_7_shape_to_dash(z):
    if isinstance(z, np.float64) is True or isinstance(z, np.int64) is True:
        z = z
    else:
        z = str(z)
        z = z.replace(u"\u2022", "*").encode("utf-8").decode("ascii", errors="replace").encode('latin-1', 'replace').decode("utf-8")
        z = z.replace('???', '-')
        print(z)
    return z





