"""Import Modules"""
# Manipulation
import pandas as pd
import numpy as np
# Calculations
import datetime
import random
import operator
# Visualizations
import matplotlib.pyplot as plt


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
Range = DataRange(PolicyNumbers, year_start=2010, year_end=2017)

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
columns_1 = ['Insured_Year', 'Year_Only_Lag', 'Raw_Claims_Amount']
ClaimsData = pd.DataFrame(columns=columns_1)
ClaimsData['Raw_Claims_Amount'] = ClaimAmt
ClaimsData['Insured_Year'] = [int(c.strip("'")[:4]) for c in ClaimStartDates]
ClaimsData['Year_Only_Lag'] = LagYrs
# Incremental Claims Amount
py_data = ClaimsData['Raw_Claims_Amount'].groupby([ClaimsData['Insured_Year'], ClaimsData['Year_Only_Lag']]).sum().reset_index()
# Convert into data-frame
py_data = pd.DataFrame(py_data)
# Cumulative Claims Amount
py_data["cumsum"] = py_data["Raw_Claims_Amount"].groupby(py_data["Insured_Year"]).cumsum()


"""Past Inflation Rates"""
# Establish Inflation Index
columns_2 = ['Year', 'CumPastInflation']
Inflation_df = pd.DataFrame(columns=columns_2)
# Past Inflation Years
Inflation_df['Year'] = [2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]
# Past Inflation Index
Inflation_df['CumPastInflation'] = [1.32, 1.27, 1.28, 1.22, 1.16, 1.12, 1.09, 1.07, 1.05, 1.04, 1.00, 1.01, 1.01]


class IACL:

    @staticmethod
    def uplift_past_inflation(start_year, lag_year, amt, year_end_cap, inflation_year, inflation_rate):
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
                elif year_end_cap == insured_year:
                    """NEW ~CONTROL~ CODE FOR BASE INFLATION YEAR FORMULA TO BE ADDED"""
                    tempo_df['Inflated_Claims_Amount'][row] = tempo_df['Inflated_Claims_Amount'][row]
                else:
                    tempo_df['Inflated_Claims_Amount'][row] = tempo_df['Inflated_Claims_Amount'][row]
                tempo_df['Inflated_cumsum'] = tempo_df['Inflated_Claims_Amount'].groupby(tempo_df['Insured_Year']).cumsum()
        return tempo_df

    @staticmethod
    def individual_loss_development_factors(data_frame, year_end_cap, start_year_col, lag_year_col, inflated_cum_amt_col):
        df = data_frame
        df['Inflated_LossDF'] = 1   # default ratio 1
        start_year_col = str(start_year_col)
        lag_year_col = str(lag_year_col)
        inflated_cum_amt_col = str(inflated_cum_amt_col)
        for row in range(0, len(df[start_year_col])):
            insured_year = df.loc[row, start_year_col]
            lag_yr = df.loc[row, lag_year_col]
            current_year = df.loc[row, start_year_col] + df.loc[row, lag_year_col]
            curr_cum_amt = df.loc[row, inflated_cum_amt_col]
            if current_year > year_end_cap or len(df.loc[(df[start_year_col] == insured_year) & (
                    df[lag_year_col] == (lag_yr + 1)), inflated_cum_amt_col]) == 0:
                next_cum_amt = 0
            else:
                next_cum_amt = df.loc[(py_data[start_year_col] == insured_year) & (
                        df[lag_year_col] == (lag_yr + 1)), inflated_cum_amt_col].values[0]
            ldf = next_cum_amt / curr_cum_amt
            df.loc[row, 'Inflated_LossDF'] = ldf
        return df

    @staticmethod
    def future_claims_data_df(data_frame, year_end_cap, start_year_col):
        df = data_frame
        year_start_cap = min(df[start_year_col])
        """Create a New Df of future Predicted Years & LagYears"""
        columns_3 = ['InsuredYear', 'PredictedYear_Only_Lag',
                     'Previous_cumsum', 'Predicted_cumsum', 'Predicted_Incremental',
                     'Previous_Inflated_cumsum', 'Predicted_Inflated_cumsum', 'Predicted_Inflated_Incremental']
        temp_df = pd.DataFrame(columns=columns_3)
        # +1 due to 31 Dec 2017 (also not a Bday) & +1 due to range exlusion of last value cap
        insured_yr = list(range(year_start_cap + 1, year_end_cap + 1, 1))
        temp_df['InsuredYear'] = insured_yr
        lags = list(range(0, year_end_cap - year_start_cap, 1))  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        temp_df['PredictedYear_Only_Lag'] = lags
        # Establish Predicted data-frame
        predicted_df = pd.DataFrame(columns=columns_3)
        """Coordinates of predicted Insured Years & Lag Years"""
        x = 1  # Do nothing
        i = 0  # For loop impute indexing
        for row in range(0, len(temp_df['InsuredYear'])):
            base_yr = temp_df.loc[row, 'InsuredYear']
            for lag in range(0, len(temp_df['PredictedYear_Only_Lag'])):
                lag_yr = temp_df.loc[lag, 'PredictedYear_Only_Lag']
                p_yr = base_yr + temp_df.loc[lag, 'PredictedYear_Only_Lag']
                if p_yr > year_end_cap:
                    predicted_df.loc[i, 'InsuredYear'] = base_yr
                    predicted_df.loc[i, 'PredictedYear_Only_Lag'] = lag_yr
                    i += 1
                else:
                    x = x
        """Impute latest cumulative amounts available"""
        # Inflated
        for row in range(0, len(predicted_df)):
            base = predicted_df.loc[row, 'InsuredYear']
            lag = predicted_df.loc[row, 'PredictedYear_Only_Lag']
            pred_yr = base + lag
            if base == year_end_cap:
                prev_inflated_cum_sum = df.loc[(df['Insured_Year'] == base), 'Inflated_cumsum'].values[0]
            else:
                if pred_yr > year_end_cap or len(df.loc[(df['Insured_Year'] == base) & (
                        df['Year_Only_Lag'] == lag - 1), 'Inflated_cumsum']) == 0:
                    max_lag = df.loc[(df['Insured_Year'] == base), 'Year_Only_Lag'].max()
                    prev_inflated_cum_sum = df.loc[(df['Insured_Year'] == base) &
                                                   (df['Year_Only_Lag'] == max_lag), 'Inflated_cumsum'].values[0]
                else:
                    prev_inflated_cum_sum = df.loc[(df['Insured_Year'] == base) &
                                                   (df['Year_Only_Lag'] == lag - 1), 'Inflated_cumsum'].values[0]
            predicted_df.loc[row, 'Previous_Inflated_cumsum'] = prev_inflated_cum_sum
        return predicted_df

    @staticmethod
    def avg_loss_development_factors(data_frame, lag_year_col, cum_amt_col, indiv_ldf_col, year_end_cap, start_year_col):
        df = data_frame
        year_start_cap = min(df[start_year_col])
        temp_df = pd.DataFrame(columns=['PredictedYear_Only_Lag'])
        lags = list(range(0, year_end_cap - year_start_cap, 1))  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        temp_df['PredictedYear_Only_Lag'] = lags
        """Establish averaged-year-to-year LDF"""
        columns_4 = ['Year_Only_Lag',
                     'Inflated_SimpleMeanLossDF', 'Inflated_VolWtdLossDF',
                     'Inflated_CumToUlt_SimpleMeanLossDF', 'Inflated_CumToUlt_VolWtdLossDF',
                     'Inflated_SimpleMeanLossDF_5year', 'Inflated_VolWtdLossDF_5year',
                     'Inflated_SimpleMeanLossDF_3year', 'Inflated_VolWtdLossDF_3year',
                     'Inflated_SelectLossDF']
        ldf_df = pd.DataFrame(columns=columns_4)
        Lags = list(range(0, year_end_cap - year_start_cap, 1))
        ldf_df['Year_Only_Lag'] = Lags
        """Inflated All Year Average LDFs"""
        # Inflated
        i = 0
        for lag in range(0, len(temp_df['PredictedYear_Only_Lag'])):
            lag_yr = temp_df.loc[lag, 'PredictedYear_Only_Lag']
            # Simple Mean
            # due to 0 input so exlude last value
            simple_mean_ldf = df.loc[df[lag_year_col] == lag_yr, indiv_ldf_col][:-1].mean()
            ldf_df.loc[i, 'Inflated_SimpleMeanLossDF'] = simple_mean_ldf
            # Volume Weighted
            denominator = df.loc[df[lag_year_col] == (lag_yr + 1), cum_amt_col].sum()
            numerator = df.loc[df[lag_year_col] == lag_yr, cum_amt_col][:-1].sum()
            vol_wtd_ldf = denominator / numerator
            ldf_df.loc[i, 'Inflated_VolWtdLossDF'] = vol_wtd_ldf
            i += 1
        # [::-1] to flip or invert the row order
        ldf_df['Inflated_CumToUlt_SimpleMeanLossDF'] = ldf_df['Inflated_SimpleMeanLossDF'][::-1].cumprod()
        ldf_df['Inflated_CumToUlt_VolWtdLossDF'] = ldf_df['Inflated_VolWtdLossDF'][::-1].cumprod()
        """Inflated 5/3 Year Average LDFs"""
        # Inflated
        i = 0
        for lag in range(0, len(temp_df['PredictedYear_Only_Lag'])):
            lag_yr = temp_df.loc[lag, 'PredictedYear_Only_Lag']
            # Simple Mean
            year_a = 5  # 5 Year
            SimpleMeanLossDF_Ayear = df.loc[df[lag_year_col] == lag_yr, indiv_ldf_col][:year_a].mean()
            ldf_df.loc[i, 'Inflated_SimpleMeanLossDF_5year'] = SimpleMeanLossDF_Ayear
            year_b = 3  # 3 Year
            SimpleMeanLossDF_Byear = df.loc[df[lag_year_col] == lag_yr, indiv_ldf_col][:year_b].mean()
            ldf_df.loc[i, 'Inflated_SimpleMeanLossDF_3year'] = SimpleMeanLossDF_Byear
            # Volume Weighted
            Deno_A = df.loc[df[lag_year_col] == (lag_yr + 1), cum_amt_col][:year_a].sum()
            Neum_A = df.loc[df[lag_year_col] == lag_yr, cum_amt_col][:year_a].sum()
            VolWtdLossDF_A = Deno_A / Neum_A
            ldf_df.loc[i, 'Inflated_VolWtdLossDF_5year'] = VolWtdLossDF_A
            Deno_B = df.loc[df[lag_year_col] == (lag_yr + 1), cum_amt_col][:year_b].sum()
            Neum_B = df.loc[df[lag_year_col] == lag_yr, cum_amt_col][:year_b].sum()
            VolWtdLossDF_B = Deno_B / Neum_B
            ldf_df.loc[i, 'Inflated_VolWtdLossDF_3year'] = VolWtdLossDF_B
            i += 1
        return ldf_df

    @staticmethod
    def select_loss_development_factors(data_frame, selected):
        df = data_frame
        """Selected"""
        df['Inflated_SelectLossDF'] = df[selected]
        return df

    @staticmethod
    def predict_claims(data_frame_pred, data_frame_check_ref, data_frame_ldf, pred_lag_year_col, lag_year_col, selected_ldf_col, inflated_cum_amt_col, start_year_col, year_end_cap, inflated_cum_amt_col_check):
        df_pred = data_frame_pred
        df_check = data_frame_check_ref
        df_ldf = data_frame_ldf
        pred_lag_year_col = str(pred_lag_year_col)
        lag_year_col = str(lag_year_col)
        start_year_col = str(start_year_col)
        selected_ldf_col = str(selected_ldf_col)
        inflated_cum_amt_col = str(inflated_cum_amt_col)
        YearEndCap = year_end_cap
        YearStartCap = min(df_check[start_year_col])
        # Inflated
        # Set Equal for easy reference
        data_frame_pred['Predicted_Inflated_cumsum'] = data_frame_pred['Previous_Inflated_cumsum']
        lagyearlimit = (YearEndCap - YearStartCap) - 1
        x = 1  # Do nothing
        for row in range(0, len(df_pred)):
            PredLagYr = df_pred.loc[row, pred_lag_year_col]
            BaseInsuredYr = df_pred.loc[row, start_year_col]
            MaxLagYr = df_check.loc[(df_check[start_year_col] == BaseInsuredYr), lag_year_col].max()
            for r in range(0, len(df_ldf)):
                if (df_ldf.loc[r, lag_year_col] == lagyearlimit):
                    x = x  # To avoid NaN
                elif (df_ldf.loc[r, lag_year_col] == MaxLagYr):
                    # LDF multiplication
                    LDF = df_ldf.loc[(df_ldf[lag_year_col] >= MaxLagYr) & (df_ldf[lag_year_col] <= (PredLagYr - 1)), selected_ldf_col].prod()
                    df_pred.loc[row, inflated_cum_amt_col] = df_pred.loc[row, inflated_cum_amt_col] * LDF
                else:
                    x = x  # Do nothing
        """Data-type adjustments"""
        # Years
        df_pred[[start_year_col, pred_lag_year_col]] = df_pred[
            [start_year_col, pred_lag_year_col]].astype(int)
        # Amounts
        df_pred[['Predicted_Inflated_cumsum', inflated_cum_amt_col]] = df_pred[
            ['Predicted_Inflated_cumsum', inflated_cum_amt_col]].astype(float)
        """Predict Incremental Amount"""
        # Inflated
        for row in range(0, len(df_pred)):
            InsurYr = df_pred.loc[row, start_year_col]
            LagYr = df_pred.loc[row, pred_lag_year_col]
            CurrCum = df_pred.loc[row, 'Predicted_Inflated_cumsum']
            # For which we can't look up in Predicted_df
            if len(df_pred.loc[(df_pred[start_year_col] == InsurYr) & (
                    df_pred[pred_lag_year_col] == LagYr - 1), 'Predicted_Inflated_cumsum']) == 0:
                PrevCum = df_check.loc[(py_data[start_year_col] == InsurYr) & (
                        df_check[lag_year_col] == LagYr - 1), inflated_cum_amt_col_check].values[0]
            # For which we can look up in Predicted_df
            else:
                PrevCum = df_pred.loc[(df_pred[start_year_col] == InsurYr) & (
                        df_pred[pred_lag_year_col] == LagYr - 1), 'Predicted_Inflated_cumsum'].values[0]

            df_pred.loc[row, 'Predicted_Inflated_Incremental'] = (CurrCum - PrevCum)

        df_pred[['Predicted_Inflated_Incremental']] = df_pred[['Predicted_Inflated_Incremental']].astype(float)

        return df_pred

    @staticmethod
    def uplift_future_inflation(data_frame, lag_year_col, start_year_col, year_end_cap, inflation_df, inflated_incr_amt_col, inflation_year, inflation_rate):
        Predicted_df = data_frame
        Inflation_df = inflation_df
        """Define proceeding inflation year from year_end_cap"""
        FutureInflation = Inflation_df.loc[(Inflation_df[inflation_year] == (year_end_cap + 1)), inflation_rate].values[0]
        for row in range(0, len(Predicted_df)):
            InsurYr = Predicted_df.loc[row, start_year_col]
            LagYr = Predicted_df.loc[row, lag_year_col]
            CurrIncremAmt = Predicted_df.loc[row, inflated_incr_amt_col]
            Predicted_df['FutureUplifted_Predicted_Inflated_Incremental'] = 1
            Predicted_df.loc[row, 'FutureUplifted_Predicted_Inflated_Incremental'] = CurrIncremAmt * (FutureInflation ** LagYr)
        return Predicted_df


"""Inflation Calculations"""
ClaimsDataInflated = IACL.uplift_past_inflation(start_year=py_data['Insured_Year'],
                                                lag_year=py_data['Year_Only_Lag'],
                                                amt=py_data['Raw_Claims_Amount'],
                                                year_end_cap=2017,
                                                inflation_year=Inflation_df['Year'],
                                                inflation_rate=Inflation_df['CumPastInflation'])

"""Individual Loss Development Factors"""
ClaimsDataInflated = IACL.individual_loss_development_factors(data_frame=ClaimsDataInflated,
                                                              year_end_cap=2017,
                                                              start_year_col='Insured_Year',
                                                              lag_year_col='Year_Only_Lag',
                                                              inflated_cum_amt_col='Inflated_cumsum')

"""Prepare Future Claims data frame"""
FutureClaimsData = IACL.future_claims_data_df(data_frame=ClaimsDataInflated,
                                              year_end_cap=2017,
                                              start_year_col='Insured_Year')

"""Loss Development Factors Calculations"""
LDF_averages = IACL.avg_loss_development_factors(data_frame=py_data,
                                                 lag_year_col='Year_Only_Lag',
                                                 cum_amt_col='Inflated_cumsum',
                                                 indiv_ldf_col='Inflated_LossDF',
                                                 year_end_cap=2017,
                                                 start_year_col='Insured_Year')

LDF_averages = IACL.select_loss_development_factors(data_frame=LDF_averages,
                                                    selected='Inflated_VolWtdLossDF')

"""Predict Future Claims; apply LDFs"""
df_pred = IACL.predict_claims(data_frame_pred=FutureClaimsData, data_frame_check_ref=ClaimsDataInflated,
                              data_frame_ldf=LDF_averages,
                              pred_lag_year_col='PredictedYear_Only_Lag', lag_year_col='Year_Only_Lag',
                              start_year_col='Insured_Year', selected_ldf_col='Inflated_SelectLossDF',
                              inflated_cum_amt_col='Previous_Inflated_cumsum',
                              inflated_cum_amt_col_check='Inflated_cumsum',
                              year_end_cap=2017)
PredictedInflatedIncrementalTriangle = pd.pivot_table(df_pred, index=["InsuredYear"],
                                                      columns=["PredictedYear_Only_Lag"],
                                                      values=["Predicted_Inflated_Incremental"])

"""Apply future inflation rates"""
IACL.uplift_future_inflation(data_frame=df_pred, lag_year_col='PredictedYear_Only_Lag', start_year_col="InsuredYear",
                             year_end_cap=2017,
                             inflation_df=Inflation_df,
                             inflated_incr_amt_col=["Predicted_Inflated_Incremental"],
                             inflation_year=Inflation_df['Year'], inflation_rate=Inflation_df['CumPastInflation'])


def SubPlotFullClaims(PastDataFrameName, PastInsuredYearColumn, PastLagYearColumn, PastValueColumn,
                      FutureDataFrameName, FutureInsuredYearColumn, FutureLagYearColumn, FutureValueColumn):
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    # https://stackoverflow.com/questions/16419670/increase-distance-between-title-and-plot-in-matplolib
    """Create New df"""
    Filtered_NewColumnNames = ["Insured_Year", "Year_Only_Lag", "ClaimAmt"]
    # Past
    Past_Filtered_df = pd.DataFrame(PastDataFrameName[[PastInsuredYearColumn, PastLagYearColumn, PastValueColumn]])
    Past_Filtered_df.columns = Filtered_NewColumnNames
    # Future
    Future_Filtered_df = pd.DataFrame(
        FutureDataFrameName[[FutureInsuredYearColumn, FutureLagYearColumn, FutureValueColumn]])
    Future_Filtered_df.columns = Filtered_NewColumnNames
    """Unique Insured Years List"""
    # Past
    Past_InsuredYr_List = list(PastDataFrameName[PastInsuredYearColumn].unique())
    # Future
    Future_InsuredYr_List = list(FutureDataFrameName[FutureInsuredYearColumn].unique())
    """Unique Lag Years List"""
    # Past
    Past_LagYr_List = list(PastDataFrameName[PastLagYearColumn].unique())
    # Future
    Future_LagYr_List = list(FutureDataFrameName[FutureLagYearColumn].unique())
    """Color List"""
    ALL_Colors = ['r', 'b', 'g', 'y', 'k', 'c', 'm', 'saddlebrown', 'pink', 'lawngreen']
    Past_Color_List = ALL_Colors[:len(Past_InsuredYr_List)]
    Future_Color_List = ALL_Colors[:len(Future_InsuredYr_List)]
    """Plotting"""
    fig = plt.figure(2, figsize=(12, 18))
    plt.xticks([])  # remove initial blank plot default ticks
    plt.yticks([])  # remove initial blank plot default ticks
    plt.title('Sub Plot Full Claims Data')
    rcParams['axes.titlepad'] = 100  # position title
    plt.box(on=None)  # Remove boundary line
    """Full Loop Plot"""
    Full_Filtered_df = pd.concat([Past_Filtered_df, Future_Filtered_df])
    i = 0
    for row_A in range(0, len(Past_InsuredYr_List)):
        ax = fig.add_subplot(5, 2, 1 + i)
        Year_i = Past_InsuredYr_List[row_A]
        Full_SubFiltered_df = Full_Filtered_df.loc[Full_Filtered_df['Insured_Year'].isin([Year_i])]
        plt.plot(Full_SubFiltered_df['Year_Only_Lag'], Full_SubFiltered_df['ClaimAmt'],
                 label=('Predicted %d' % Year_i), linestyle='--', color=Past_Color_List[row_A])
        plt.legend()
        i += 1
        plt.xticks(np.arange(0, (YearEndCap - YearStartCap), step=1))
        plt.xlabel('Developement Year')
        plt.ylabel('Claims Value')
    """Past Loop Plot"""
    i = 0
    for row_A in range(0, len(Past_InsuredYr_List)):
        ax = fig.add_subplot(5, 2, 1 + i)
        Year_i = Past_InsuredYr_List[row_A]
        Past_SubFiltered_df = Past_Filtered_df.loc[Past_Filtered_df['Insured_Year'].isin([Year_i])]
        plt.plot(Past_SubFiltered_df['Year_Only_Lag'], Past_SubFiltered_df['ClaimAmt'],
                 label=('Historical %d' % Year_i), linestyle='-', color=Past_Color_List[row_A], marker='o')
        plt.legend()
        i += 1
    """Plot Attributes"""
    fig.tight_layout()
    # plt.show()


"""Plot"""
# Output Plot
SubPlotFullClaims(PastDataFrameName=py_data, PastInsuredYearColumn="Insured_Year",
                  PastLagYearColumn="Year_Only_Lag", PastValueColumn="Inflated_cumsum",
                  FutureDataFrameName=Predicted_df, FutureInsuredYearColumn="InsuredYear",
                  FutureLagYearColumn="PredictedYear_Only_Lag", FutureValueColumn="Predicted_Inflated_cumsum")



