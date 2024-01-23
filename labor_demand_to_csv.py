import pandas as pd
import warnings
warnings.filterwarnings('ignore')

file_path = "C:/Users/katsa/Downloads/cps_00098.csv.gz"
data = pd.read_csv(file_path,compression='gzip')

sampled_df = data.sample(frac=0.10,random_state=42)

features = ["YEAR","SEX","AGE","RACE","HISPAN","PAIDHOUR","EARNWEEK","HOURWAGE","UHRSWORK1","EDUC"]
weight_variable = "EARNWT"
df = sampled_df[features + [weight_variable]]


df.info()
df.isna().sum()
df.nunique()
print(df.dtypes)



df.describe()
valid_conditions = (df['EARNWEEK'] > 0) & (df['EARNWEEK'] < 9999.99) & (df['UHRSWORK1'] > 0) & (df['UHRSWORK1'] < 997)
hourly_wage_condition = (df['PAIDHOUR'] == 2) & (df['HOURWAGE'] > 0) & (df['HOURWAGE'] < 99)


df.loc[valid_conditions & hourly_wage_condition, 'HOURWAGE'] = df['HOURWAGE']


not_paid_hour_condition = (df['PAIDHOUR'] == 1)
df.loc[valid_conditions & not_paid_hour_condition, 'HOURWAGE'] = df['EARNWEEK'] / df['UHRSWORK1']

df['HOURWAGE'] = df['HOURWAGE'].clip(lower=0, upper=997)



df['Low_Wage_Worker'] = [1 if x <= 15 else 0 for x in df['HOURWAGE']]

df.isna().sum()



df['RACE'].nunique()
#race



race_ = {100: 'white', 200: 'Black', 651: 'Asian'}
df['RACE'] = df['RACE'].apply(lambda x: race_[x] if x in race_ and x > 0 else 'other')



df['EDUC'].describe()

less_than_high_school = (df['EDUC'] <= 60)
high_school = (df['EDUC'] >= 70) & (df['EDUC'] <= 73)
some_college = (df['EDUC'] >= 80) & (df['EDUC'] <= 100)
college = (df['EDUC'] >= 110)


df.loc[less_than_high_school, 'EDUC'] = 'Less than High School'
df.loc[high_school, 'EDUC'] = 'High School'
df.loc[some_college, 'EDUC'] = 'Some College'
df.loc[college, 'EDUC'] = 'College'

df['EDUC'].describe()

df.isna().sum()

# turn to csv file

df.to_csv("hw3_labor_demand2.csv", index=False)