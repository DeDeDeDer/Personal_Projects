import pandas as pd
import numpy as np
import plotly.express as px
import plotly

"""Parameters"""
max_Claims = 50000000
max_Premiums = 100000000
min_all = 0
log_scaling = True

location = '/Users/Derrick-Vlad-/Desktop/Personal Projects/2019/Web Scraping/Monetary Authority Singapore (MAS)/Attempt 3/BackTests/CompileBackTest_4/Finalized_2/Plot_Data_3.csv'
typesss2 = pd.read_csv(location)
typesss2 = typesss2[(typesss2['Gross Claims'].astype(float)>min_all) & (typesss2['Gross Premiums'].astype(float)>min_all) & (typesss2['Operating Result'].astype(float)>min_all)]
typesss2 = typesss2[(typesss2['Gross Claims'].astype(float)<max_Claims) & (typesss2['Gross Premiums'].astype(float)<max_Premiums)]

continents = ['Fire', 'Cargo & Hull', 'Work Injury', 'Misc']
typesss2 = typesss2[~typesss2.Coverage.isin(continents)]

typesss2 = typesss2.replace(r'^\s*$', np.nan, regex=True)
typesss2.dropna(inplace=True)

"""Plotter"""
fig5 = px.scatter(typesss2,
                  x="Gross Claims", y="Gross Premiums",
                  animation_frame="Year", animation_group="Insurer Code",
                  size="Operating Result",
                  color="Coverage", hover_name="Insurer Code",
                  log_x=log_scaling, size_max=60, title='Evolution: 14-Years of Private General Insurance Data in SG (Logarithmic Scale)'
                  #range_x=[100, 1000000], range_y=[-50000, 250000],
                  )
plotly.offline.plot(fig5, filename='file6.html')







