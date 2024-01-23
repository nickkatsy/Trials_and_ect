import pandas as pd
import warnings
warnings.filterwarnings('ignore')

file_path = "C:/Users/katsa/Downloads/cps_00089.csv.gz"
data = pd.read_csv(file_path,compression='gzip')

sampled_df = data.sample(frac=0.2, random_state=42)

features = ["YEAR", "LABFORCE","SEX","AGE","NCHLT5"]
weight_variable = "ASECWT"
df = sampled_df[features + [weight_variable]]


df.info()
df.isna().sum()
df.nunique()



df.describe()

df['LABFORCE'].describe()
df['LABFORCE'] = [1 if X == 2 else 0 for X in df['LABFORCE']]

df['NCHLT5'] = [1 if X >= 1 else 0 for X in df['NCHLT5']]



df.to_csv("hw2_sampled_down_again111.csv", index=False)

#statistical analysis











