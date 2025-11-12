import os, urllib.request, collections
from Bio import Entrez
import urllib.error
import pandas as pd
import time
import mygene
from goatools.obo_parser import GODag
from goatools.mapslim import mapslim
import numpy as np
import requests, time, re
from collections import Counter
from bs4 import BeautifulSoup #standard HTML navigation and extraction 
import json

""" 
    PIPELINE TO IDENTIFY NDD DISEASE GENES
    EXTRACTED EXCEL FILES FOR EACH DISEASE:  https://disgenet.com/search?view=ALL&idents=ALL&source=CURATED&tab=DISEASES
"""
# ------------------------------------------------------------------------------
# CLEAN_DISGENET_EXCEL_OUTPUT: excel disease x gene output from disgenet
# ------------------------------------------------------------------------------
def clean_disgenet_excel_output(df, disease):

    df = pd.read_excel(df)
    df['disease'] = disease

    columns_to_keep = ['disease', 'gene_symbol', 'score', 'geneDescription', 'ei', 'num_pmid_assoc']
    df = df[columns_to_keep]
    return df
# ------------------------------------------------------------------------------
# MERGE DFS: excel disease x gene output from disgenet
# ------------------------------------------------------------------------------
def merge_dfs(directory):

    csv_files = [f for f in os.listdir(directory) if f.endswith(".csv")]

    dfs = []
    for f in csv_files:
        path = os.path.join(directory, f)
        df = pd.read_csv(path)
        dfs.append(df)

    merged_df = pd.concat(dfs, ignore_index=True)
    merged_df = merged_df.sort_values('disease')

    columns_to_keep = ['disease', 'gene_symbol', 'score', 'geneDescription', 'ei', 'num_pmid_assoc']
    merged_df = merged_df[columns_to_keep]
    return merged_df

# ------------------------------------------------------------------------------
# FILTER_GDA_SCORE: filter disease genes with min gda score
# ------------------------------------------------------------------------------
def filter_gda_score(df, threshold=0.8):
    df['score'] = pd.to_numeric(df['score'], errors='coerce')
    return df[df['score'] >= threshold]

# ------------------------------------------------------------------------------
# ORTHOLOGUE: identifies mouse and fly orthologues
# ------------------------------------------------------------------------------
def orthologue(df):

    """ Fetch mouse and Drosophila orthologues for human genes via DIOPT (flyrnai.org)
    Input: list of human genes (case-insensitive) """

    WANT_COLS = [
        "fly_symbol", "FBgn",
        "rank", "weighted_score",
        "best_score", "best_score_reverse",
        "identity_percent", "similarity_percent"]
    
    # _empty: returns None for columns if no record for the gene 
    def _empty():
        return {k: None for k in WANT_COLS}
    
    # _to_float: protects against None errors 
    def _to_float(x):
        try:
            return float(str(x).replace(",", "").strip())
        except Exception:
            return None

    # SESSION: creates a persistent HTTP session, so multiple requests reuse the same connection
    SESSION = requests.Session()
    # SESSION.headers.update: updates the web browser to mimic a normal web browser 
    SESSION.headers.update({
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://www.flyrnai.org/diopt",
        "Origin":  "https://www.flyrnai.org"})
    try:
        SESSION.get("https://www.flyrnai.org/diopt", timeout=20)  # warm up the connection simple GET request to the DIOPT homepage
    except Exception:
        pass
    
    # FLY_ORTHOLOG: grabs fly symbol and associated fields 
    def fly_ortholog(symbol):
        # build the exact request 
        url = "https://www.flyrnai.org/cgi-bin/DRSC_orthologs.pl"
        form = {
            "input_species": "9606",
            "output_species": "7227", ## fly 
            "search_fields": "***",
            "gene_list": symbol,
            "gene_file": "",
            "submit": "Submit Search",
            "search_datasets": "All",
            "additional_filter": "NoLow",
            "output": "tsv",
        }
        try:
            r = SESSION.post(url, data=form, timeout=30) # web request
            if r.status_code != 200:
                return _empty()

            soup = BeautifulSoup(r.text, "html.parser") #BeautifulSoup turns the messy HTML string into a structured tree browser  
            tbl = soup.find("table", id="results-table") # identify results table
            if not tbl:
                return _empty()

            headers = [th.get_text(strip=True).replace("\xa0", " ")
                       for th in tbl.select("thead th")] # identifies the table headers
            row = tbl.select_one("tbody tr") # first row in the table
            if not row:
                return _empty()

            tds   = row.find_all("td") # gets all the <td> cells in that row
            cells = [td.get_text(strip=True) for td in tds] #extract text
            rec   = dict(zip(headers, cells)) # zips headers and their text

            out = {
                "fly_symbol":         rec.get("Fly Symbol"),
                "FBgn":               rec.get("Fly Species Gene ID"),
                "rank":               rec.get("Rank"),
                "weighted_score":     _to_float(rec.get("Weighted Score")),
                "best_score":         rec.get("Best Score"),
                "best_score_reverse": rec.get("Best Score Reverse"),
                "identity_percent":       None, #not on this page
                "similarity_percent":     None, #not on this page
            }

            try:
                aidx = headers.index("Alignment & Scores") #hidden behind a little "Alignment & Scores" form
                form_tag = tds[aidx].find("form")
                if form_tag and form_tag.get("action"):
                    action_url = requests.compat.urljoin("https://www.flyrnai.org",
                                                         form_tag.get("action"))
                    payload = {
                        inp.get("name"): inp.get("value", "")
                        for inp in form_tag.find_all("input")
                        if inp.get("name")}

                    """You send a second POST to that alignment URL with the same payload. 
                    Adding Referer: r.url makes it look like you came from the DIOPT page 
                    (some servers block requests that dont include it)"""

                    r2 = SESSION.post(action_url, data=payload, timeout=30,
                                      headers={"Referer": r.url})
                    txt = r2.text

                    m_id = (re.search(r'Identit(?:y|ies)\s*[:=].*?\(([\d.]+)\s*%\)', txt, re.I|re.S)
                            or re.search(r'Identity\s*[:=]\s*([\d.]+)\s*%', txt, re.I))
                    m_sm = (re.search(r'(?:Similarity|Positives)\s*[:=].*?\(([\d.]+)\s*%\)', txt, re.I|re.S)
                            or re.search(r'(?:Similarity|Positives)\s*[:=]\s*([\d.]+)\s*%', txt, re.I))
                    if m_id: out["identity_percent"]   = float(m_id.group(1))
                    if m_sm: out["similarity_percent"] = float(m_sm.group(1))

            except Exception:
                pass

            return out

        except Exception:
            return _empty()
    
    # MOUSE_ORTHOLOG: grabs mouse symbol 
    def mouse_ortholog(symbol):
        try:
            url = "https://www.flyrnai.org/cgi-bin/DRSC_orthologs.pl"
            form = {
                "input_species": "9606", "output_species": "10090", ## mouse 
                "search_fields": "***", "gene_list": symbol,
                "search_datasets": "All", "additional_filter": "NoLow", "output": "tsv",
            }
            r = SESSION.post(url, data=form, timeout=30)
            if r.status_code != 200:
                return None
            soup = BeautifulSoup(r.text, "html.parser")
            tbl = soup.find("table", id="results-table")
            if not tbl:
                return None
            headers = [th.get_text(strip=True).replace("\xa0", " ") for th in tbl.select("thead th")]
            row = tbl.select_one("tbody tr")
            if not row:
                return None
            cells = [td.get_text(strip=True) for td in row.find_all("td")]
            rec = dict(zip(headers, cells))
            return rec.get("Mouse Symbol") or next((rec[h] for h in headers if h.endswith("Symbol")), None)
        except Exception:
            return None

    
    rows = []
    for g in df['gene_symbol']:
        print(f'orthologue for {g}')
        human = str(g).upper()
        mouse_sym = mouse_ortholog(human)
        fly_rec   = fly_ortholog(human)
        rows.append({"gene_symbol": human, "mouse_symbol": mouse_sym, **fly_rec})

    df_ortho = pd.DataFrame(rows, columns=["gene_symbol", "mouse_symbol"] + WANT_COLS)
    merged = df.merge(df_ortho, on='gene_symbol', how='left')
    return merged

# ------------------------------------------------------------------------------
# FILTER_WEIGHTED_SCORE: filter disease genes with min weighted score
# ------------------------------------------------------------------------------
def filter_weighted_score(df, threshold=7):
    df['weighted_score'] = pd.to_numeric(df['weighted_score'], errors='coerce')
    return df[df['weighted_score'] >= threshold]

# ------------------------------------------------------------------------------
# GDA_X_WEIGHTED: filter disease genes with min weighted score
# ------------------------------------------------------------------------------
def gda_x_weighted_score(df):
    df['gda_x_weighted'] = df['weighted_score'] * df['score']
    return df

# ------------------------------------------------------------------------------
# TOP_20: top 20 disease genes per disease
# ------------------------------------------------------------------------------
def top_20(df):
    top20_df = (
        df.sort_values('score', ascending=False)
        .groupby('disease', group_keys=False)
        .head(20))
    return top20_df






# directory = '/Volumes/lab-windingm/home/users/cochral/PhD/NDD/GENES/disease-mining/disgenet-extracted-disease-genes'
# df = merge_dfs(directory)

# df = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/PhD/NDD/GENES/disease-mining/disease-genes-all.csv')
# df = filter_gda_score(df, threshold=0.8)
# df = orthologue(df)


df = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/PhD/NDD/GENES/disease-mining/safari/SFARI-Gene_genes_10-23-2025release_11-10-2025export.csv')
df = df[df['gene-score'] == 1]
df = df.rename(columns={'gene-symbol': 'gene_symbol'})
print(df['gene-score'].unique())
df = orthologue(df)

out = os.path.join('/Users/cochral/Desktop/test.csv')
df.to_csv(out, index=False)




