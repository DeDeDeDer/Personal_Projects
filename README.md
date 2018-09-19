# Personal_Projects
This GitHub holds all my personal project's that I have worked on as a past time. 
Project's are mainly focused on Data Science, Insurance Pricing & Reserving fields.
<br>A mapping of these are laid out below.
![ScreenShot](/Pictures/MapBackground_4.png)
<br>
<br>


> # **Insurance (Pricing) & Data Science**


<br>

# **What is Predictive Modelling?**

<br>
It is simply the framework to integrate past data & statistics to predict 
future outcomes or project liabilities. There are 4 main techniques;
Bayesian, Decision Trees, Support Vector Machines & Neural Networks.
My project's utilizes mainly Bayesian & Decision Tree techniques.
Hence, focused primarily on linear regression models.
<br>


.
<br>
## [At Its Simplest, Predictive Modelling](https://medium.com/@DRicky.Ch29/at-its-simplest-predictive-modelling-b3c0c0b0716d)
<br>

![ScreenShot](/Pictures/IntroModel_1.png)

An article publication aimed at explaining concepts to:
<br>1. Generalised structure to Predictive Modelling
<br>2. Alternative interpretations to various statistical model metrics
<br>
<br>The article follows the generalized framework of:
<br>
<ol><ins>Data preparation</ins></ol>
<ol>- Preliminary data analysis, executing 4-Tier's of data cleaning. (Correct, Complete, Create, Convert)</ol>

<ol><ins>Exploratory Data Analysis</ins></ol>
<ol>- Uni- Bi- & Multi- Analysis</ol>

<ol><ins>Model Preparation</ins></ol>
<ol>- Data stratified Train/Test splits, Hyper parameter tuning, parameter evaluation metrics.</ol>
<ol>- Feature Engineering (Quantity & Quality), Feature evaluation metrics</ol>

<ol><ins>Predictive Modelling (Classification Problem)</ins></ol>
<ol>- Ensembles (Hard & Soft Voting)</ol>

<br>
<a href="https://medium.com/@DRicky.Ch29/at-its-simplest-predictive-modelling-b3c0c0b0716d"><strong>Click To View</strong></a>
<br>



<br>

# **What is Web Scraping?**

<br>
In short, it is simply the automated process of extracting data from the web. 
Subsequently, cleaning any irregularities & conducting Exploratory Data Analysis
to spot Trends & Patterns.
<br>

.
<br>
## Python Web Scraping PDF & Data Cleaning (Part 1) 
[Article](https://medium.com/@DRicky.Ch29/web-scraping-pdf-tables-data-cleaning-part-1-cb6d8d47a6de)
or
[Python Code](https://github.com/DeDeDeDer/Personal_Projects/blob/master/Web%20Scraping%20(Data%20Science%20%26%20Insurance%20Pricing)/Web_Scrap_Insurance_Returns.py)
<br>

![ScreenShot](/Pictures/WebScrapPart1.png)

A Python Kernel written to automate repetitive clicking of 1,228c URLs &
converting 1,000c PDF Tables into CSV to compile data.

<br>Contents:
<ol>1. Collate online source code URLs & sub-page URLs</ol>
<ol>2. Download online data via URLs</ol>
<ol>3. Convert & Neaten PDF Table into CSV</ol>
<ol>4. Compile all CSV Tables</ol>

<br>
<a href="https://medium.com/@DRicky.Ch29/web-scraping-pdf-tables-data-cleaning-part-1-cb6d8d47a6de"><strong>Click To View</strong></a>
<br>



.
<br>
## [Python Web Scraping Data Analysis Motor Insurance (Part 2)](https://medium.com/@DRicky.Ch29/python-web-scraping-data-analysis-motor-insurance-part-2-4cd7162ba644)
<br>

![ScreenShot](/Pictures/WebScrapPart2.png)

After extracting Annual Insurance Data Returns in the Part 1 series, we proceed to
analyze the data.

<br>Contents:
<ol><ins>Patterns</ins></ol>
<ol>1. Benchmark Range of ROC on Expense & Loss Ratios</ol>

<ol><ins>Trends</ins></ol>
<ol>2. Growing reinsurance ceded abroad beyond the ASEAN region</ol>
<ol>3. Declining averages for Earned Premiums & Claims Incurred (with falling inflation rates)</ol>
<ol>4. Average ROC, Expense & Loss Ratios</ol>


<br>
<a href="https://medium.com/@DRicky.Ch29/python-web-scraping-data-analysis-motor-insurance-part-2-4cd7162ba644"><strong>Click To View</strong></a>
<br>



<br>

# **What is Exploratory Data Analysis?**

<br>
It is simply the analyzing of data sets to summarize characteristics & patterns. 
These include Uni- Bi- & Multi- Variate Analysis. Often discovering underlying
relationships that conventional models overlook.
<br>

.
<br>
## [EDA & Feature Engineering Focused](https://www.kaggle.com/derrickchua29/feature-engineering-eda-focused/notebook)
<br>

![ScreenShot](/Pictures/EDA_article_1.png)


<br><ins>EDA Summary</ins>

<br>1. Those who have had past experience of financial distress (target variable):
<br>>Made lesser loans or exceed deadlines
<br>>Tend to have lesser dependents & debt ratio & net worth
<br>>As expected are of lower-tier income, But lower debt ratio
<br>

<br>2. Ignoring mortality and time value of money (i.e.Annuities)
<br>>Debt ratio & Net worth shows gaussian distribution against age
<br>

<br>3. Those who had acts of debt delinquency (Made loans or exceed deadlines)
<br>>Tend to be from the higher-tier income or Retired
<br>

<br>4. Others
<br>>The higher the income, the higher the debt ratio
<br>>The higher the income, the lower the dependents


<br>
<a href="https://www.kaggle.com/derrickchua29/feature-engineering-eda-focused/notebook"><strong>Click To View</strong></a>
<br>



# **What is General Linear Modelling?**

<br>
It is simply applying the fundamental straight line concept of a Y = mx + C. 
In other words, the idea that variable relationships are 1-dimensional (positive
or negative).
<br>


.<br>
## [Ensemble Models Comparison Techniques](https://www.kaggle.com/derrickchua29/ensemble-models-comparison-techniques)
<br>

![ScreenShot](/Pictures/GLM_article_1.png)


<br>A Python Kernel aimed to:
<ol>1. Get a better understanding of the simplified predictive modelling framework</ol>
<ol>2. Grasp the logic behind different coding methods & concise techniques used</ol>
<ol>3. Comparisons between different models</ol>
<br>
<ol><ins>Coding Techniques :</ins></ol>
<ol>A.List comprehensions</ol>
<ol>B.Samples to reduce computational cost</ol>
<ol>C.Concise 'def' functions that can be used repetitively</ol>
<ol>D.Pivoting using groupby</ol>
<ol>E.When & How to convert and reshape dictionary’s into lists or dataframes</ol>
<ol>F.Quickly split dataframe columns</ol>
<ol>H.Loop Sub-plots</ol>
<ol>I.Quick Lambda formulae functions</ol>
<ol>J.Quick looping print or DataFrame conversion of summative scores</ol>
<ol>K.Order plot components</ol>
<ol>L.Create & Plot Bulk Ensemble comparative results</ol>


<br>
<a href="https://www.kaggle.com/derrickchua29/ensemble-models-comparison-techniques"><strong>Click To View</strong></a>
<br>

<br>
<br>
<br>


> # **Insurance (Reserving)**


<br>

# **Claim Simulations**

<br>
In short, this projects contains a Python Kernel to automate the probabilistic  
claims simulation process for actuarial reserving calculations. 
<br>
Reserving Method Used: Inflation Adjusted Chain Ladder

.
<br>
## Claims Simulation 
[Article](https://medium.com/@DRicky.Ch29/inflation-adjusted-chain-ladder-iacl-with-only-python-pandas-module-512914d9a1d)
or
[Python Code Guide](https://www.kaggle.com/derrickchua29/simulating-claim-data-iacl-calculation)
or
[Python Code v2](https://github.com/DeDeDeDer/Personal_Projects/blob/master/Claims%20Simulation%20(Insurance%20Reserving)/Claims_Simulator.py)
<br>

![ScreenShot](/Pictures/ClaimsSimu_article_1.png)

<br>Present: Simulation supports Claim Numbers (Poisson) & Amounts (Gaussian).
<br>Ongoing: 
<br>1. Claim Numbers (Negative Binomial) & Amounts (Lognormal).
<br>2. Support Bornhuetter-Ferguson Method (BF).

<br>Contents:
<ol>0. Assumptions</ol>
<ol>1. Development-Year lags</ol>
<ol>2. Incremental & Cumulative claim amounts</ol>
<ol>3. Uplift past inflation for incremental amounts & Derive cumulative</ol>
<ol>4. Individual Loss Development Factors (LDFs)</ol>
<ol>5. Raw preliminary view of triangle</ol>
<ol>6. Establish predicted lag years data frame</ol>
<ol>7. Impute latest cumulative amounts</ol>
<ol>8. Simple Mean & Volume Weighted LDFs & 5/3 Year Averages & Select</ol>
<ol>9. Predict future cumulative amounts</ol>
<ol>10. Calculate incremental amounts</ol>
<ol>11. Project future inflation for incremental amounts</ol>
<ol>12. Reserve summation</ol>


<br>
<a href="https://medium.com/@DRicky.Ch29/web-scraping-pdf-tables-data-cleaning-part-1-cb6d8d47a6de"><strong>Click To View</strong></a>
<br>


<br>
<br>
<br>

> # **Microsoft Package**


<br>

# **Microsoft Package**

<br>
Prior to learning Python coding language, I had to refine the basics. 
Since Excel & VBA are broadly deemed essential skill-sets, I thought 
I build some personal models. Ideas are inspired whilst at my work 
placement tenure at a  consultancy company. The main objective was to 
ease manual & repetitive tasking's.
<br>

.
<br>
## Word Documentations 
[Spreadsheet](https://www.dropbox.com/s/b4cgvhjui2mj0qq/Bulk%20MailMerge%20v2.0.xlsm?dl=0)
or
[Excel VBA Code](https://www.dropbox.com/s/b4cgvhjui2mj0qq/Bulk%20MailMerge%20v2.0.xlsm?dl=0)
<br>

![ScreenShot](/Pictures/WordExcelLogo_1.png)

<br>

<br>A reproducible Excel VBA programme that automates bulk simultaneous word 
document mail merges. Data entry checks (file exists etc.) & cleaning (excess 
spaces, invalid file directory ...) are done by the coding as well. This code 
does NOT use the standard mail merge function that operates ONLY on 1-single 
document. Instead allows running on mass word documentations.

<br>Inspiration:
<br>Whilst assisting my previous employer to prepare clients for the European 
General Data Protection Regulations (GDPR) privacy documentations, I created 
this programme to streamline over 30hours of manual work.
<br>

.
<br>
## Outlook Communications 
[Spreadsheet](https://www.dropbox.com/s/o50up79cttwyfa3/Bulk%20Emailing%20v2.0.xlsm?dl=0)
or
[Excel VBA Code](https://www.dropbox.com/s/o50up79cttwyfa3/Bulk%20Emailing%20v2.0.xlsm?dl=0)
<br>

![ScreenShot](/Pictures/OutlookExcelLogo_1.png)

<br>

<br>A reproducible Excel VBA programme that automates multiple simultaneous email 
communications if recipients receive overlapping/same attachments or spreadsheet 
tables.

<br>Inspiration:
<br>A responsibility of mine at a previous company involved weekly roll-forward 
projection updates. I found this repetitive & build this model to automate the 
job. It mitigated manual human input errors & eased the job handing over
process.
<br>


<br>






