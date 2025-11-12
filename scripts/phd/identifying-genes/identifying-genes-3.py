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


""" PIPELINE TO NARROW DOWN GENES """

""" 
    ASD: GENES EXTRACTED FROM SFARI DATABASE "https://gene.sfari.org/database/gene-scoring/"
    - FILTERED FOR CATAGORY 1
    - FILTERED FOR EAGLE > 12

    SCZ: GENES EXTRACTED FROM "https://schema.broadinstitute.org/results"
    - TOP 10 EXOME-WIDE SIGNIFICANCE GENES
    - RARE PROTEIN TRUNCATING VARIANTS

    ID: GENES EXTRACTED FROM SysNDD "https://sysndd.dbmr.unibe.ch/"
    - DISEASE NAME = INTELECTUAL DISABILITY
    - CATEGORY = DEFINITIVE
    - AUTOSOMAL DOMINANT, LOF MECHANISM
    - SORTED BY NUMBER OF INDIVIDUALS MENTIONED (UNSURE IF OKAY)

    EPI: GENES EXTRACTED FROM "https://epi25.broadinstitute.org/downloads"
    - HITS FOR DEE OR ALL EPILEPSY GROUPED

    ADHD: GENES EXTRACTED FROM "https://pmc.ncbi.nlm.nih.gov/articles/PMC11759603/#SM1" 
    - MEDIA 3, SHEET = S.Table3 (CLEANED HEADER)
    - 6 GENES IDENTIFIED < 0.1 (POST EXLUSION OF SPARK COHORT-ASD RELATED)
    - KDM5B OVERLAP FROM "https://www.nature.com/articles/s41467-024-50247-7" 
    
"""


# ------------------------------------------------------------------------------
# SFARI_FILTER: filter genes extracted from sfari database
# ------------------------------------------------------------------------------
def sfari(df):
    df = df[df['gene-score'] == 1] # catagory 1
    df = df[df['eagle'] >= 12] # eagle score  
    df['disease'] = 'Autism Spectrum Disorders'  
    df = df.rename(columns={'gene-symbol': 'gene_symbol'})
    return df

# ------------------------------------------------------------------------------
# SCHEMA: filter scz genes extracted from schema
# ------------------------------------------------------------------------------
def scz_schema(df):
     
    df['Gene_ID'] = df['Gene']

    mg = mygene.MyGeneInfo()
    ensembl_ids = df["Gene"].dropna().unique().tolist()
    results = mg.querymany(
        ensembl_ids,
        scopes="ensembl.gene",
        fields="symbol",
        species="human")
     
    mapping = pd.DataFrame(results)[["query", "symbol"]].rename(columns={"query": "Gene"})
    df = df.merge(mapping, on="Gene", how="left")
    df["gene_symbol"] = df["symbol"].combine_first(df["Gene"])
    df = df.drop(columns=["symbol"])
    df["Rank"] = df.index + 1

    df_top10 = df.head(10)
    df_top10 = df_top10[["gene_symbol"] + [c for c in df.columns if c != "gene_symbol"]]
    df_top10['disease'] = 'Schizophrenia'
    return df_top10

# ------------------------------------------------------------------------------
# SYSNDD: filter genes extracted from SysNDD database
# ------------------------------------------------------------------------------
def id_sysndd(df):
    df = df[df['category_id'] == 1] # definitive
    # df = df[df['disease_ontology_name'] == 'intellectual disability']
    # df = df[df['disease_ontology_name'].str.contains('intellectual disability|intellectual developmental disorder', case=False, na=False)]
    df = df[df['disease_ontology_name'].str.contains(r'^(intellectual (disability|developmental disorder))',case=False,na=False)]
    df = df[df['inheritance_filter'] == 'Autosomal dominant']
    df = df[df['synopsis'].str.contains('de novo', case=False, na=False)]
    df['disease'] = 'Intellectual Disability'  
    df = df.rename(columns={'symbol': 'gene_symbol'})
    return df

# ------------------------------------------------------------------------------
# EPI25: filter genes extracted from epi25 database
# ------------------------------------------------------------------------------
def epi25(df):
    df['group'] = df['group'].str.strip().str.upper()
    df = df[df['group'].str.contains(('DEE|EPI'), case=False, na=False)]
    df = df[df['ptv_pval'] < 0.000001] # hits for DEE or all epilepsy 
    df = df.sort_values("ptv_pval", ascending=True)
     
    mg = mygene.MyGeneInfo()
    ensembl_ids = df["gene_id"].dropna().unique().tolist()
    results = mg.querymany(
        ensembl_ids,
        scopes="ensembl.gene",
        fields="symbol",
        species="human")
     
    mapping = pd.DataFrame(results)[["query", "symbol"]].rename(columns={"query": "gene_id"})
    df = df.merge(mapping, on="gene_id", how="left")
    df["gene_symbol"] = df["symbol"].combine_first(df["gene_id"])
    df = df.drop(columns=["symbol"])
    df = df[["gene_symbol"] + [c for c in df.columns if c != "gene_symbol"]]
    df['disease'] = 'Epilepsy'
    return df

# ------------------------------------------------------------------------------
# ADHD: filter genes adhd identified 
# ------------------------------------------------------------------------------
def adhd(df):
    df = df.rename(columns={'Gene': 'gene_symbol'})
    # tada_genes_post_spark_exclusion = 'KDM5B|CTNND2|MMP16|RAI1|ZMYND11|ST8SIA2' 
    tada_genes_post_spark_exclusion = ['KDM5B', 'CTNND2', 'MMP16', 'RAI1', 'ZMYND11', 'ST8SIA2']
    df = df[df['gene_symbol'].isin(tada_genes_post_spark_exclusion)]
    df['disease'] = 'Attention Deficit Hyperactivity Disorder'
    return df

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
    for g in df['gene_symbol'].unique():
        print(f'orthologue for {g}')
        human = str(g).upper()
        mouse_sym = mouse_ortholog(human)
        fly_rec   = fly_ortholog(human)
        rows.append({"gene_symbol": human, "mouse_symbol": mouse_sym, **fly_rec})

    df_ortho = pd.DataFrame(rows, columns=["gene_symbol", "mouse_symbol"] + WANT_COLS)
    merged = df.merge(df_ortho, on='gene_symbol', how='left')
    return merged

# ------------------------------------------------------------------------------
# DISEASE_ASSOCIATION: identifies assocations in target disorders >> 0.6
# ------------------------------------------------------------------------------
def disease_association(df):
    
    """
    Requires df['Gene'];
    Screen for >= 0.7 scored associated diseases;
    Identifies if hits are within target disorders;
    Return df['disgenet_assocations']

    target_diseases = [
            "Schizophrenia",
            "Autism Spectrum Disorders",
            "Attention Deficit Hyperactivity Disorder",
            "Epilepsy",
            "Intellectual Disability"]

    """

    DISGENET_TOKEN = "1a851ae4-eae3-4209-a5f1-bdb1f4582256" # personal token

    # Create a persistent session (reuses the same connection = faster for many genes)
    session = requests.Session()
    session.headers.update({
        "Authorization": f"Bearer {DISGENET_TOKEN}",
        "Accept": "application/json"
    })

    def gene_summary(symbol, original_disease):
        """
        Query DisGeNET for high-level gene–disease associations for one gene.
        Returns a list of disease names with score ≥ 0.8.
        """
        url = f"https://api.disgenet.com/api/v1/gda/summary?gene_symbol={symbol}"
        r = session.get(url)

        if r.status_code != 200:
            print(f"⚠️ Request failed for {symbol}: {r.status_code}")
            return [], None

        payload = r.json().get("payload", [])
        if not payload:
            print(f"No data found for {symbol}")
            return [], None

        diseases = []
        for record in payload:
            score = record.get("score", 0)
            name = record.get("diseaseName")
            if score >= 0.6 and name: ### edit score simply for assocations 
                diseases.append(name)
        
        # Define your target disease categories
        target_diseases = [
            "Schizophrenia",
            "Autism Spectrum Disorders",
            "Attention Deficit Hyperactivity Disorder",
            "Epilepsy",
            "Intellectual Disability",

        ]
        target_diseases_lower = [d.lower() for d in target_diseases]

        filtered = []
        for disease in diseases:
            if disease.lower() in target_diseases_lower:
                filtered.append(disease)

        score = None
        for record in payload:
            name = record.get("diseaseName", "")
            # if name.lower() == original_disease.lower():
            if original_disease.lower() in name.lower() or name.lower() in original_disease.lower():
                score = record.get("score", None)
                break

        return filtered, score
    

    df = df.reset_index(drop=True)
    df["disgenet_associations"] = None
    df["gda_score"] = None

    for i, row in enumerate(df.itertuples(index=False)):

        gene = str(row.gene_symbol).upper()
        original_disease = row.disease

        disease_associations, score = gene_summary(gene,original_disease)
        disease_associations = list(disease_associations)

        df.at[i, "disgenet_associations"] = disease_associations          # list of >=0.7 matches in your target set
        df.at[i, "gda_score"] = score 

        time.sleep(0.75)


    df["all_diseases"] = df.apply(
    lambda r: (r.disgenet_associations or []) + ([r.disease] if isinstance(r.disease, str) and r.disease else []),
    axis=1)

    return df


# ------------------------------------------------------------------------------
# GOTERM: identifies gene ontology profiles for each gene 
# ------------------------------------------------------------------------------ 
def goterm(df):

    """ Retrieve GO annotations and map them to GO-slim terms for each gene """

    # 1. MYGENE: identifies go info relating to human gene: bp (biological process), mf (molecular function) + cc (cellular component)
    mg = mygene.MyGeneInfo()
    genes_for_go = df["gene_symbol"].str.upper().unique().tolist()

    res = mg.querymany(
        genes_for_go,
        scopes="symbol,alias",
        fields=["symbol", "ensembl.gene", "go.BP", "go.MF", "go.CC"],
        species="human")

    # converts the MyGene JSON output into a DataFrame
    df_go = pd.DataFrame(res).dropna(subset=["query"]).drop_duplicates("query", keep="first")

    '''" symbol is the canonical (official) gene symbol recognized by NCBI or Ensembl;
        query is the gene name you provided to MyGene. '''

    if "symbol" not in df_go.columns: 
        df_go["symbol"] = df_go["query"] #fallback 

    # extracts enemble ID
    def extract_ensembl(obj):
        if isinstance(obj, dict):
            return obj.get("gene")
        if isinstance(obj, list) and len(obj) > 0 and isinstance(obj[0], dict):
            return obj[0].get("gene")
        return None

    df_go["ensembl_id"] = df_go["ensembl"].apply(extract_ensembl)


    df_go["gene_symbol"] = df_go["query"].str.upper() # uppercase gene name
    df_go["gene_matched"] = df_go["symbol"].str.upper() # uppercase canonical symbol returned by MyGene (for comparison)
    df_go["alias"] = df_go.apply(lambda r: r["query"] if r["gene_symbol"] != r["gene_matched"] else None, axis=1) # if query differs from canonical 
    
    if "go" not in df_go.columns:  
        df_go["go"] = None #fallback 

    
    """  MyGene returns go as a dict with keys BP, MF, CC, each a list of dicts with id and term.

    unpack_ids → pulls GO IDs per aspect (e.g., ["GO:000..."]).

    unpack_terms → pulls the human-readable term names per aspect; returns None if empty, else list of strings."""

    def unpack_ids(go_block, aspect):
        if not isinstance(go_block, dict): return []
        return [d.get("id") for d in go_block.get(aspect, []) if isinstance(d, dict) and d.get("id")]

    def unpack_terms(go_block, aspect):
        if not isinstance(go_block, dict): return None
        terms = [d.get("term") for d in go_block.get(aspect, []) if isinstance(d, dict) and d.get("term")]
        return terms if terms else None

    # unpack GO terms
    df_go["GO_Biological_Process"]  = df_go["go"].apply(lambda g: unpack_terms(g, "BP"))
    df_go["GO_Molecular_Function"]  = df_go["go"].apply(lambda g: unpack_terms(g, "MF"))
    df_go["GO_Cellular_Component"]  = df_go["go"].apply(lambda g: unpack_terms(g, "CC"))

    # GO IDs (MyGene) for SLIM ontology mapping later on
    df_go["BP_ids"] = df_go["go"].apply(lambda g: unpack_ids(g, "BP"))
    df_go["MF_ids"] = df_go["go"].apply(lambda g: unpack_ids(g, "MF"))
    df_go["CC_ids"] = df_go["go"].apply(lambda g: unpack_ids(g, "CC"))

    """Its essential to reduce the complexity of GO terms by associating them with higher-level, more general categories

    GO IDs pulled from MyGene (sourced from the Gene Ontology database). This section maps these IDs to their hierarchal,

    more general catagory using locally downloaded GO and GO-Slim files from the Gene Ontology Consortium """

    # 2. GO-SLIM version (a simplified, high-level version of the full GO)
    cache = os.path.join(os.path.expanduser("~"), ".cache", "go_slim") # sets up a directory for caching GO and GO-slim files locally
    os.makedirs(cache, exist_ok=True)
    go_obo = os.path.join(cache, "go-basic.obo")
    slim_obo = os.path.join(cache, "goslim_generic.obo")

    if not os.path.exists(go_obo):
        urllib.request.urlretrieve("http://purl.obolibrary.org/obo/go/go-basic.obo", go_obo)
    if not os.path.exists(slim_obo):
        urllib.request.urlretrieve("http://current.geneontology.org/ontology/subsets/goslim_generic.obo", slim_obo)

    GO = GODag(go_obo) # full GO terms downloaded
    SLIM = GODag(slim_obo) # high-level, simplified GO terms downloaded

    # GO term (ID) to its corresponding terms in the GO-slim (hierarchal GO terms) 
    def map_to_slim_names(id_list, wanted_ns):
        if not id_list:
            return None
        hits = []
        for gid in id_list:
            if gid in GO:
                direct, ancestors = mapslim(gid, GO, SLIM)
                chosen = direct if direct else ancestors  # official slim ancestors
                for sid in chosen:
                    if sid in SLIM and SLIM[sid].namespace == wanted_ns:
                        hits.append(SLIM[sid].name)
        if not hits:
            return None
        return sorted(set(hits))  # unique + stable order

    df_go["GO_Slim_BP"] = df_go["BP_ids"].apply(lambda ids: map_to_slim_names(ids, "biological_process"))
    df_go["GO_Slim_MF"] = df_go["MF_ids"].apply(lambda ids: map_to_slim_names(ids, "molecular_function"))
    df_go["GO_Slim_CC"] = df_go["CC_ids"].apply(lambda ids: map_to_slim_names(ids, "cellular_component"))


    # same as above but preserves duplicates to identify most frequent go term present!!
    def map_to_slim_names_with_dups(id_list):
        hits = []
        for gid in (id_list or []):
            if gid in GO:
                direct, ancestors = mapslim(gid, GO, SLIM)
                for sid in (direct if direct else ancestors):
                    if sid in SLIM and SLIM[sid].namespace == "biological_process":
                        hits.append(SLIM[sid].name)   # keep duplicates
        return hits

    df_go["GO_Slim_BP_all"] = df_go["BP_ids"].apply(map_to_slim_names_with_dups)

    # identify most frequent go term present
    def get_top_bp_terms(lst):
        if not lst:
            return None
        counts = Counter(lst)
        max_count = max(counts.values())
        # return all terms that share the top frequency
        top_terms = [term for term, count in counts.items() if count == max_count]
        return top_terms  # keeps as list

    df_go["GO_Slim_BP_Most_Frequent"] = df_go["GO_Slim_BP_all"].apply(get_top_bp_terms)
    
    # select specific GO and GO-SLIM terms
    df_go = df_go[[
        "gene_symbol", "gene_matched", 'ensembl_id',
        "GO_Biological_Process", "GO_Molecular_Function", "GO_Cellular_Component",
        "GO_Slim_BP", "GO_Slim_MF", "GO_Slim_CC", "GO_Slim_BP_Most_Frequent"]]

    out = df.merge(df_go, on="gene_symbol", how="left")
    print(out)
    return out

# ------------------------------------------------------------------------------
# WEIGHTED_SCORE: filters for weighted score >= 9.99
# ------------------------------------------------------------------------------
def weighted_score(df):
    df = df[df['weighted_score'] >= 9.9]
    return df
# ------------------------------------------------------------------------------
# EAGLE_SCORE: filters for weighted score >= 9.99
# ------------------------------------------------------------------------------
def eagle_score(df):
    df = df.sort_values('eagle', ascending=False)
    df_top5 = df.head(5)
    return df_top5
# ------------------------------------------------------------------------------
# ID_FILTER: filters ID genes
# ------------------------------------------------------------------------------
def id_filter(df):
    trip_unavailable = ['HDAC3', 'Capr', 'Atu', 'Fem-1', 'eEF2', 'CG5065', 'Arf79F', 'pns', 'Ufd4']
    for trip in trip_unavailable:
        df = df[df["fly_symbol"] != trip]
    df = df[df['gda_score'] >= 0.7]

    loss_of_function = 'truncation|truncating|nonsense|deletion'
    df = df[df['synopsis'].str.contains(loss_of_function, case=False, na=False)]

    def extract_individuals(text):
        # pattern matches numbers that come before "individual", "patients", or "affected"
        match = re.search(r'(?:>|≈|~)?\s*(\d+)\s*(?:individual|patient|affected)', text, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
        else:
            return None

    df['n_individuals'] = df['synopsis'].apply(extract_individuals)
    df = df.sort_values('n_individuals', ascending=False)

    df_top = df.head(6)
    return df_top

# ------------------------------------------------------------------------------
# MERGE: merge dataframes
# ------------------------------------------------------------------------------
def concat_dataframes(df1, df2, df3, df4, df5):
    combined = pd.concat([df1, df2, df3, df4, df5], ignore_index=True)
    cols_to_keep = [
    'gene_symbol','disease', 'gda_score', 'eagle', 'Rank',
    'mouse_symbol', 'fly_symbol', 'FBgn', 'weighted_score',  'rank', 'best_score', 'best_score_reverse', 'similarity_percent', 'identity_percent',
    'disgenet_associations', 'all_diseases']

    combined = combined[cols_to_keep]
    return combined



""" HARDCODED PIPELINE """

""" ASD """
### LOAD SFARI EXPORTED CSV AND FILTER FOR CATAGORY 1 GENES
df_sfari_export = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/PhD/NDD/GENES/disease-mining/refined/asd/SFARI-Gene_genes_10-23-2025release_11-10-2025export.csv')
df_sfari = sfari(df_sfari_export) 
### IDENTIFY GENE ORTHOLOGS AND DISEASE ASSOCATIONS >0.6
df_sfari = orthologue(df_sfari)
df_sfari = disease_association(df_sfari)
df_sfari.to_csv(os.path.join('/Volumes/lab-windingm/home/users/cochral/PhD/NDD/GENES/disease-mining/refined/asd/sfari.csv'), index=False)
## WEIGHTED SCORE >= 9.99
df_sfari = weighted_score(df_sfari)
## 5 GENES WITH HIGHEST EAGLE SCORES
df_sfari = eagle_score(df_sfari)
df_sfari.to_csv(os.path.join('/Volumes/lab-windingm/home/users/cochral/PhD/NDD/GENES/disease-mining/refined/asd/sfari_top.csv'), index=False)

""" SCZ """
### LOAD SCZ SCHEMA RESULTS AND FILTER FOR THE TOP 10 GENES IDENTIFIED
df_schema_export = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/PhD/NDD/GENES/disease-mining/refined/scz/meta_results_2025_11_10_13_35_57.csv')
df_schema = scz_schema(df_schema_export)
### IDENTIFY GENE ORTHOLOGS AND DISEASE ASSOCATIONS >0.6
df_schema = orthologue(df_schema)
df_schema = disease_association(df_schema)
df_schema.to_csv(os.path.join('/Volumes/lab-windingm/home/users/cochral/PhD/NDD/GENES/disease-mining/refined/scz/schema.csv'), index=False)
## WEIGHTED SCORE >= 9.99
df_schema = weighted_score(df_schema)
## TRiP UNAVAILABLE
df_schema = df_schema[df_schema["gene_symbol"] != "XPO7"]
df_schema.to_csv(os.path.join('/Volumes/lab-windingm/home/users/cochral/PhD/NDD/GENES/disease-mining/refined/scz/schema_top.csv'), index=False)

""" ID """
### LOAD SYSNDD RESULTS AND FILTER FOR ID AUTOSOMANL DOMINANT, DEFINITIVE GENES
df_id = pd.read_excel('/Volumes/lab-windingm/home/users/cochral/PhD/NDD/GENES/disease-mining/refined/id/sysndd_entity_table.xlsx',  sheet_name='data')
df_id = id_sysndd(df_id)
### IDENTIFY GENE ORTHOLOGS AND DISEASE ASSOCATIONS >0.6
df_id = orthologue(df_id)
df_id = disease_association(df_id)
df_id.to_csv(os.path.join('/Volumes/lab-windingm/home/users/cochral/PhD/NDD/GENES/disease-mining/refined/id/sysndd_id_filter.csv'), index=False)
## WEIGHTED SCORE >= 9.99
df_id = weighted_score(df_id)
## FURTHER FITERING
df_id = id_filter(df_id)
df_id.to_csv(os.path.join('/Volumes/lab-windingm/home/users/cochral/PhD/NDD/GENES/disease-mining/refined/id/sysndd_id_filter_top.csv'), index=False)

""" EPI """
## LOAD EPI25 RESULTS AND FILTER FOR TOP DEE AND ALL-EPILEPSY HITS
df_epi25 = pd.read_csv("/Volumes/lab-windingm/home/users/cochral/PhD/NDD/GENES/disease-mining/refined/epi/Epi25_gene_results.tsv.bgz", compression="gzip", sep="\t")
df_epi25 = epi25(df_epi25)
### IDENTIFY GENE ORTHOLOGS AND DISEASE ASSOCATIONS >0.6
df_epi25 = orthologue(df_epi25)
df_epi25 = disease_association(df_epi25)
## WEIGHTED SCORE >= 9.99
df_epi25 = weighted_score(df_epi25)
df_epi25.to_csv("/Volumes/lab-windingm/home/users/cochral/PhD/NDD/GENES/disease-mining/refined/epi/Epi25_gene_results_top.csv", index=False)

""" ADHD """
## LOAD RESULTS AND FILTER TO 6 IDENTIFIES ADHD GENES 
df_adhd = pd.read_excel('/Volumes/lab-windingm/home/users/cochral/PhD/NDD/GENES/disease-mining/refined/adhd/media-3.xlsx', sheet_name='S.Table3-cleaned')
df_adhd = adhd(df_adhd)
# ### IDENTIFY GENE ORTHOLOGS AND DISEASE ASSOCATIONS >0.6
df_adhd = orthologue(df_adhd)
df_adhd = disease_association(df_adhd)
df_adhd.to_csv(os.path.join('/Volumes/lab-windingm/home/users/cochral/PhD/NDD/GENES/disease-mining/refined/adhd/adhd.csv'), index=False)
## WEIGHTED SCORE >= 9.99
df_adhd = weighted_score(df_adhd)
df_adhd.to_csv(os.path.join('/Volumes/lab-windingm/home/users/cochral/PhD/NDD/GENES/disease-mining/refined/adhd/adhd_top.csv'), index=False)

## CONCAT DATAFRAMES
df = concat_dataframes(df_sfari, df_schema, df_id, df_epi25, df_adhd)
## GENE ONTOLOGY
df = goterm(df)
df.to_csv(os.path.join('/Volumes/lab-windingm/home/users/cochral/PhD/NDD/GENES/disease-mining/refined/target_genes.csv'), index=False)





