from Bio import Entrez
import pandas as pd
import time

Entrez.email = "lena.cochrane@crick.ac.uk"

## COUNT GROSS NUMBER OF PAPERS 
def count_papers(gene):
    handle = Entrez.esearch(db="pubmed", term=gene, retmode="xml")
    record = Entrez.read(handle)
    return int(record["Count"])

print(count_papers('neurexin'))


## COUNT PAPERS GIVEN A CERTAIN CONTEXT
def count_papers_by_context(gene):
    contexts = ["mouse", "drosophila", "zebrafish", "human", "cell line"]
    results = {}
    for c in contexts:
        query = query = f'"{gene}"[Title/Abstract] AND {c}[Title/Abstract]'
        handle = Entrez.esearch(db="pubmed", term=query, retmode="xml")
        record = Entrez.read(handle)
        results[c] = int(record["Count"])
    return results

print(count_papers_by_context("neurexin"))


genes = ["neurexin", "fmr1", "shank3", "cntnap2"]  # your gene panel

data = []
for gene in genes:
    total = count_papers(gene)
    contexts = count_papers_by_context(gene)
    data.append({"gene": gene, "total": total, **contexts})
    time.sleep(0.4)  # ~2.5 queries/sec is safe for NCBI

df = pd.DataFrame(data)
df.to_csv("gene_paper_counts.csv", index=False)






