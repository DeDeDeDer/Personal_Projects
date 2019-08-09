"""Packages"""
# Structure
import pandas as pd
import numpy as np
# Visual
import matplotlib.pyplot as plt
import seaborn as sns
from geopy.geocoders import Nominatim
import ssl


"""Functions"""
def converge_all():
    all_area_codes = list(range(1, 27, 1))
    df2 = pd.DataFrame()
    for a in range(0, len(all_area_codes)):
        area_code_i = all_area_codes[a]
        print('Running Area Code: ', str(area_code_i))
        file_i = '/Users/Derrick-Vlad-/Desktop/Output_Trail_1/Raw/' + 'Area_Code_' + str(area_code_i) + '.csv'
        df = pd.read_csv(file_i, sep=',')
        df['Area_Code'] = [str(area_code_i)]*len(df)
        df2 = pd.concat((df, df2), axis=0, sort=False)  # , ignore_index=True
    df2.to_csv('/Users/Derrick-Vlad-/Desktop/' + 'Area_Code_All' + '.csv', sep=',')


def clean():
    file_i = '/Users/Derrick-Vlad-/Desktop/Output_Trail_1/Clean/' + 'All_Data.csv'
    df = pd.read_csv(file_i, sep=',', encoding='unicode_escape')
    df.columns = ['Unanmed', 'ID', 'Listing', 'ListedDate', 'Price', 'Size',
                  'PSF', 'Beds', 'Agency', 'Agent', 'Area_Code', 'Agent_Num', 'Region']
    spaced_cols = ['Listing', 'Price', 'Size', 'PSF', 'Beds', 'Agent']
    df['Listing'] = df['Listing'].str.replace("\n", "")
    df['ListedDate'] = df['ListedDate'].str.replace("\t", "")
    df.to_csv('/Users/Derrick-Vlad-/Desktop/' + 'All_Data_CLEAN' + '.csv', sep=',')


def get_coordinates(address):
    from geopy.geocoders import Nominatim
    import ssl
    API_KEY = 'Confidential'
    # Disable SSL certificate verification
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        # Legacy Python that doesn't verify HTTPS certificates by default
        pass
    else:
        # Handle target environment that doesn't support HTTPS verification
        ssl._create_default_https_context = _create_unverified_https_context

    geolocator = Nominatim(user_agent=API_KEY)
    location = geolocator.geocode(address)

    return location.latitude, location.longitude, location.address
#get_coordinates_3(address='Singapore Parc Rosewood')


def out_all_coordinates():
    new_df = pd.DataFrame()
    latitude, longitude, ADD = [], [], []
    file_i = '/Users/Derrick-Vlad-/Desktop/' + 'Unique_Addresses.csv'
    df = pd.read_csv(file_i, sep=',')
    addr = list(df['Unique_Addresses'])
    for a in range(0, len(addr)):
        addr_i = 'Singapore ' + addr[a]
        try:
            lat, lon, ad = get_coordinates(address=addr_i)
            print('loop_no#', a)
            print('lati',lat)
            print('longi',lon)
            print('add',ad)
            latitude.append(lat)
            longitude.append(lon)
            ADD.append(ad)
        except:
            print('loop_no# Not Found', a)
            latitude.append('UnableToFind')
            longitude.append('UnableToFind')
            ADD.append('UnableToFind')

    new_df['latitude'] = latitude
    new_df['longitude'] = longitude
    new_df['ADD'] = ADD
    new_df.to_csv('/Users/Derrick-Vlad-/Desktop/' + 'latlon_data.csv', sep=',')
#out_all_coordinates()


def unique_address():
    file_i = '/Users/Derrick-Vlad-/Desktop/' + 'All_Data_CLEAN.csv'
    df = pd.read_csv(file_i, sep=',', encoding='unicode_escape')
    addr = df['Listing']
    addr = [' '.join(x.split()) for x in addr]
    print(addr)
    unique = []
    for x in addr:
        if x not in unique:
            unique.append(x)
    print(unique)
    print(len(unique), ' Unique items found')
    u = pd.DataFrame(unique, columns=['Unique_Addresses'])
    u.to_csv('/Users/Derrick-Vlad-/Desktop/' + 'Unique_Addresses' + '.csv', sep=',')


def convert_quick_plot_data():
    import matplotlib.pyplot as plt
    file_i = '/Users/Derrick-Vlad-/Desktop/' + 'All_Data_CLEAN.csv'
    df = pd.read_csv(file_i, sep=',', encoding='unicode_escape')
    df['Price ($)'] = [x[2:] for x in df.Price]
    df['Price ($)'] = [x.replace(',','') for x in df['Price ($)']]
    quick = df[['Price ($)', 'Price', 'Region']]
    quick.to_csv('/Users/Derrick-Vlad-/Desktop/' + 'Quick_plot' + '.csv', sep=',')
    #df_area.boxplot(column='Price ($)', by='Region', grid=True, rot=90)


def quick_box_plot():
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    file_i = '/Users/Derrick-Vlad-/Desktop/Output_Trail_1/Clean/v1.1/' + 'Quick_Boxplot.csv'
    df = pd.read_csv(file_i, sep=',')
    df = df[['Price ($)', 'Region']]
    df.columns = ['Prices', 'Region']
    ls = list(df['Prices'])
    new = []
    for p in range(0, len(ls)):
        try:
            ls_i = ls[p]
            new.append(float(ls_i))
        except ValueError:
            new.append(np.nan)
    df['Prices'] = new
    df = df.dropna()
    # Limit Data for Outliers
    limit = 10000000
    df = df[(df['Prices'] <= limit)]
    # BoxPlot
    df.boxplot(column='Prices', by='Region', grid=True)
    plt.xticks(rotation=70)
    plt.show()

    """Customize BoxPlot"""
    # Determine the order of boxes
    order = df.groupby(by=["Region"])["Prices"].median().sort_values(ascending=True)[::-1].index

    flierprops = dict(marker='o', markerfacecolor='r', markersize=5,
                      linestyle='none', markeredgecolor='black', alpha=.1)
    meanprops = dict(color='green', )
    whiskerprops = dict(linestyle='--', color='blue')
    medianprops = dict(color='lime')
    boxprops = dict(color='b')
    # Seaborn BoxPlot
    ax = sns.boxplot(x="Region", y='Prices', data=df,
                     flierprops=flierprops,
                     meanprops=meanprops,
                     whiskerprops=whiskerprops,
                     medianprops=medianprops,
                     showmeans=False,
                     boxprops=boxprops,
                     order=order)
    plt.xticks(rotation=65)
    plt.title("Box-Plot: Singapore Housing Sale Prices By Town")
    plt.grid(True)
    # iterate over boxes
    for i, box in enumerate(ax.artists):
        box.set_edgecolor('blue')
        box.set_facecolor('white')
    plt.show()
#quick_box_plot()

