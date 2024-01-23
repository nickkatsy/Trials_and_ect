import pandas as pd
import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv('C:/ML/python/data/hw3_sampled_down.csv',delimiter=',',nrows=10000)

df.isna().sum()

print(df.dtypes)

df.describe()
df['SEX'].nunique()
df['SEX'].value_counts()

df['PAIDHOUR'].nunique()


df['SEX'] = pd.get_dummies(df.SEX,prefix='SEX').iloc[:,0:1]

df['RACE'].value_counts()


education_mapping = {'Less than High School':0,'High School':1,'Some College':2,'College':3}
df['EDUC'] = df['EDUC'].map(education_mapping)

race_mapping = {'white':0,'Black':1,'Asian':2,'other':3}
df['RACE'] = df['RACE'].map(race_mapping)
df['RACE'].nunique()



df.isna().sum()



df['RACE'].nunique()
#race



import seaborn as sns
import matplotlib.pyplot as plt


sns.heatmap(df.corr(), annot=True)
plt.show()

def subplots(df):
    _, axs = plt.subplots(2,2,figsize=(15,8))
    sns.violinplot(x='EDUC',y='Low_Wage_Worker',ax=axs[0,0],data=df)
    axs[0,0].set_title('Low Wage Worker by Education')
    sns.barplot(x='SEX', y='Low_Wage_Worker',ax=axs[0,1],data=df)
    axs[0,1].set_title('Low Wage Worker by Sex')
    sns.countplot(x='RACE',hue='Low_Wage_Worker', ax=axs[1,0],data=df)
    axs[1,0].set_title('Count of Low Wage Worker by Race')
    sns.countplot(x='YEAR',hue='Low_Wage_Worker', ax=axs[1,1],data=df)
    axs[1,1].set_title('Count of Low Wage Worker by Year')
    plt.tight_layout()

subplots(df)




low_wage_df = df[df['Low_Wage_Worker'] == 1]

# Race
plt.figure(figsize=(12,5))
sns.violinplot(x='RACE',y='AGE',data=low_wage_df,hue='EDUC')
plt.title('Distribution of Low-Wage Workers at Each Age within Each Race Group')
plt.xlabel('Race')
plt.ylabel('Age')
plt.legend(title='Education Level')
plt.show()


#Education
plt.figure(figsize=(12,5))
sns.barplot(x='EDUC',y='AGE',data=low_wage_df,hue='EDUC')
plt.title('Distribution of Low-Wage Workers at Each Age at each Education Level')
plt.xlabel('Education Level')
plt.ylabel('Age')
plt.legend(title='Education Level')
plt.show()



#Descriptive stats for age


import statsmodels.api as sm

X_age = sm.add_constant(df.drop('AGE',axis=1))
y_age = df['AGE']



model_age = sm.OLS(exog=X_age,endog=y_age).fit()
print(model_age.summary())

#model for race/ethic differences leading to low_wage workforce

X_race = sm.add_constant(df['RACE'])
y_wage = df['Low_Wage_Worker']

model_race = sm.OLS(exog=X_race,endog=y_wage).fit()
print(model_race.summary())


### train/test/split

X = df.drop('EARNWEEK',axis=1)
y = df['EARNWEEK']

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.20,random_state=42)



from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)

from sklearn.linear_model import LinearRegression,Ridge,Lasso

lr = LinearRegression()
ridge = Ridge()
lasso = Lasso()


from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,BaggingRegressor
rfr = RandomForestRegressor()
gbr = GradientBoostingRegressor()
BR = BaggingRegressor()


from sklearn.metrics import r2_score,mean_squared_error

def evaluate_model(model,X_train_scaled,X_test_scaled,y_train,y_test):
    model = model.fit(X_train_scaled,y_train)
    pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test,pred)
    mse = mean_squared_error(y_test, pred)
    print(f'{model.__class__.__name__}, --R2-- {r2*100:.2f}%; --mse-- {mse:.2f}%')
    return pred


lr_pred = evaluate_model(lr, X_train_scaled, X_test_scaled, y_train, y_test)
ridge_pred = evaluate_model(ridge, X_train_scaled, X_test_scaled, y_train, y_test)
lasso_pred = evaluate_model(lasso, X_train_scaled, X_test_scaled, y_train, y_test)
rfr_pred = evaluate_model(rfr, X_train_scaled, X_test_scaled, y_train, y_test)
BR_pred = evaluate_model(BR, X_train_scaled, X_test_scaled, y_train, y_test)
gbr_pred = evaluate_model(gbr, X_train_scaled, X_test_scaled, y_train, y_test)