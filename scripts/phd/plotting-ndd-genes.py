import pandas as pd

df_pubmed = pd.read_csv('/Users/cochral/Desktop/gene_pubmed_search.csv')
df_genes = pd.read_excel('/Users/cochral/Downloads/example.xlsx')

df_genes["Gene"] = df_genes["Gene"].str.upper()
df_pubmed["gene"] = df_pubmed["gene"].str.upper()

df = df_pubmed.merge(df_genes, left_on="gene", right_on="Gene", how="left")
df = df.drop(columns=["Gene"])

print(df)


