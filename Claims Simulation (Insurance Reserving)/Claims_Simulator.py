import pandas as pd
import numpy as np
import datetime
import random
import operator


class DataRange:

    def __init__(self, quantity, year_start, year_end):
        self.quantity = int(quantity)
        self.year_start = int(year_start)
        self.year_end = int(year_end)

        self.date_end = datetime.date(year_end, 12, 31)  # year, month, day
        self.date_start = datetime.date(year_start, 12, 31)  # year, month, day

    def output_poisson_claims_no(self, lambda_param):
        size = 1
        all_random_claim_no = []
        for row in range(0, self.quantity):
            random_claim_no = np.random.poisson(size, lambda_param)
            all_random_claim_no.append(random_claim_no)
        all_random_claim_no = [i[0] for i in all_random_claim_no]
        return all_random_claim_no

    def output_neg_binomial_claims_no(self, p_param):
        size = 1
        all_random_claim_no = []
        for row in range(0, self.quantity):
            random_claim_no = np.random.negative_binomial(size, p_param)
            all_random_claim_no.append(random_claim_no)
        return all_random_claim_no

    def output_dates_start(self):
        n_days = (self.date_end - self.date_start).days
        random_dates_start = []

        for row in range(0, self.quantity):
            random_days = random.randint(0, n_days - 1)
            random_dates = self.date_start + datetime.timedelta(days=1) + datetime.timedelta(days=random_days)
            random_dates_start.append(random_dates)

        random_dates_start = [d.strftime('%Y-%m-%d') for d in random_dates_start]
        return random_dates_start

    def output_dates_claimant(self, policy_start_dates, claim_nos):
        random_dates_claimant = []

        for row in range(0, self.quantity):
            reference_date = policy_start_dates[row]
            reference_claim_no = claim_nos[row]
            if reference_claim_no > 0:
                y, m, d = reference_date.split('-')
                reference_date = datetime.date(int(y), int(m), int(d))
                n_days = (self.date_end - reference_date).days
                random_days = random.randint(0, n_days - 1)
                random_dates = reference_date + datetime.timedelta(days=random_days)
                random_dates = random_dates.strftime('%Y-%m-%d')
                random_dates_claimant.append(random_dates)
            else:
                random_dates_claimant.append(reference_date)
        return random_dates_claimant

    def output_log_norm_claims_amt(self, mean, sd, claim_nos):
        all_random_claim_amt = []
        for row in range(0, self.quantity):
            reference_claim_no = claim_nos[row]
            if reference_claim_no > 0:
                random_claim_amt = np.random.lognormal(mean, sd, reference_claim_no).sum()
                all_random_claim_amt.append(random_claim_amt)
            else:
                all_random_claim_amt.append(0)
        return all_random_claim_amt

    def output_weibull_claims_amt(self, a, claim_nos):
        all_random_claim_amt = []
        for row in range(0, self.quantity):
            reference_claim_no = claim_nos[row]
            if reference_claim_no > 0:
                random_claim_amt = np.random.weibull(a, reference_claim_no).sum()
                all_random_claim_amt.append(random_claim_amt)
            else:
                all_random_claim_amt.append(0)
        return all_random_claim_amt

    @staticmethod
    def output_lag_years(policy_start_dates, claim_dates):
        lag_years = []
        y_start = [int(c.strip("'")[:4]) for c in policy_start_dates]
        y_claim = [int(c.strip("'")[:4]) for c in claim_dates]
        lag_years = list(map(operator.sub, y_claim, y_start))
        return lag_years


"""Data Range"""
PolicyNumbers = 10
Range = DataRange(PolicyNumbers, 2010, 2017)

"""Claim Numbers"""
ClaimNo = Range.output_poisson_claims_no(lambda_param=1)
# ClaimNo = Range.output_neg_binomial_claims_no(p_param=0.4)    # Alternative Distribution
print(ClaimNo)

"""Claim Dates"""
ClaimStartDates = Range.output_dates_start()
ClaimClaimDates = Range.output_dates_claimant(policy_start_dates=ClaimStartDates, claim_nos=ClaimNo)
print(ClaimStartDates)
print(ClaimClaimDates)

"""Claim Amounts"""
ClaimAmt = Range.output_log_norm_claims_amt(mean=10, sd=4, claim_nos=ClaimNo)
# ClaimAmt = Range.output_norm_claims_amt(mu=10, sigma=4, claim_nos=ClaimNo)    # Alternative Distribution
# ClaimAmt = Range.output_weibull_claims_amt(a=4, claim_nos=ClaimNo)    # Alternative Distribution
print(ClaimAmt)

"""Lag (Development) Years"""
LagYrs = Range.output_lag_years(policy_start_dates=ClaimStartDates, claim_dates=ClaimClaimDates)
print(LagYrs)


"""Compile Past Claims Data"""
# Set Data-frame
columns_1 = ['Insured_Year', 'Year_Only_Lag', 'Claims_Amount']
ClaimsData = pd.DataFrame(columns=columns_1)
ClaimsData['Claims_Amount'] = ClaimAmt
ClaimsData['Insured_Year'] = [int(c.strip("'")[:4]) for c in ClaimStartDates]
ClaimsData['Year_Only_Lag'] = LagYrs
# Incremental Claims Amount
py_data = ClaimsData['Claims_Amount'].groupby([ClaimsData['Insured_Year'], ClaimsData['Year_Only_Lag']]).sum().reset_index()
# Convert into data-frame
py_data = pd.DataFrame(py_data)
# Cumulative Claims Amount
py_data["cumsum"] = py_data["Claims_Amount"].groupby(py_data["Insured_Year"]).cumsum()


"""Past Inflation Rates"""
# Establish Inflation Index
columns_2 = ['Year', 'CumPastInflation']
Inflation_df = pd.DataFrame(columns=columns_2)
# Past Inflation Years
Inflation_df['Year'] = [2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]
# Past Inflation Index
Inflation_df['CumPastInflation'] = [1.32, 1.27, 1.28, 1.22, 1.16, 1.12, 1.09, 1.07, 1.05, 1.04, 1.00, 1.01, 1.01]


class IACL:
    def __int__(self, start_year, lag_year, amt):
        self.start_year = list(start_year)
        self.lag_year = list(lag_year)
        self.amt = list(amt)

    @staticmethod
    def uplift_past_inflation(self, start_year, lag_year, amt, year_end_cap, inflation_year, inflation_rate):
        tempo_df = pd.DataFrame(columns=['Insured_Year', 'Year_Only_Lag', 'Raw_Claims_Amount', 'Inflated_Claims_Amount'])
        tempo_df['Insured_Year'] = start_year
        tempo_df['Year_Only_Lag'] = lag_year
        tempo_df['Raw_Claims_Amount'] = amt
        tempo_df['Inflated_Claims_Amount'] = amt
        Inflation_df['Year'] = list(inflation_year)
        Inflation_df['CumPastInflation'] = list(inflation_rate)
        for row in range(0, len(tempo_df['Insured_Year'])):
            insured_year = tempo_df.loc[row, 'Insured_Year']
            lag_year = tempo_df.loc[row, 'Year_Only_Lag']
            transaction_year = insured_year + lag_year
            for year in range(0, len(Inflation_df['Year'])):
                current_year_inflation = Inflation_df.loc[year, 'Year']
                if current_year_inflation == insured_year:
                    current_year_perc = Inflation_df.loc[Inflation_df['Year'] == transaction_year, 'CumPastInflation']
                    to_year_perc = Inflation_df.loc[Inflation_df['Year'] == year_end_cap, 'CumPastInflation'].values[0]
                    uplift = to_year_perc / current_year_perc
                    tempo_df['Inflated_Claims_Amount'][row] = tempo_df['Inflated_Claims_Amount'][row] * uplift
                else:
                    tempo_df['Inflated_Claims_Amount'][row] = tempo_df['Inflated_Claims_Amount'][row]
                tempo_df['Inflated_cumsum'] = tempo_df['Inflated_Claims_Amount'].groupby(tempo_df['Insured_Year']).cumsum()
        return tempo_df

    def calc_loss_development_factor(self):

        pass
