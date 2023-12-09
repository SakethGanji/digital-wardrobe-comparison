import pandas as pd

df = pd.read_csv('original_styles.csv')

print("Column names in the CSV file:", df.columns.tolist())

gender_column = 'gender'
master_category_column = 'masterCategory'

genders_to_remove = ['Women', 'Girls']
categories_to_remove = ['Accessories', 'Free Items', 'Personal Care', 'Home', 'Sporting Goods']

df_filtered = df[~df[gender_column].isin(genders_to_remove)]
df_filtered = df_filtered[~df_filtered[master_category_column].isin(categories_to_remove)]

df_filtered.to_csv('styles.csv', index=False)

print("CSV file has been filtered and saved as 'filtered_styles.csv'")
