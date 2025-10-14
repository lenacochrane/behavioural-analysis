import os, urllib.request, collections
from Bio import Entrez
import urllib.error
import pandas as pd
import time
import mygene
from goatools.obo_parser import GODag
from goatools.mapslim import mapslim
import numpy as np

Entrez.email = "lena.cochrane@crick.ac.uk"

## SEARCH_PUBMED: SEARCH KEYWORDS ON PUBMED
def search_pubmed(term, retries=3):
    for i in range(retries):
        try:
            with Entrez.esearch(db="pubmed", term=term, retmode="xml") as handle:
                record = Entrez.read(handle)
            return int(record["Count"])
        except Exception as e:
            if i == retries - 1: 
                print(f"Error on term {term}: {e}")
                return None
            time.sleep(1.0)

## COUNT_PAPERS: RETURN TOTAL COUNT 
def count_papers(gene):
    return search_pubmed(f'"{gene}"[Title/Abstract]')

## COUNT_PAPERS_BY_CONTEXT: RETURN TOTAL COUNT GIVEN A CERTAIN CONTEXT E.G. MOUSE PAPERS
def count_papers_by_context(gene):
    contexts = {
        "mouse": '"Mice"[MeSH Terms]',
        "human": '"Humans"[MeSH Terms]',
        "cell line": '"Cell Line"[MeSH Terms]',
        "drosophila": '"Drosophila melanogaster"[MeSH Terms]'} # MeSH: New papers (<1–2 months old) might not yet have MeSH tags

    results = {}
    for label, mesh in contexts.items():
        query = f'"{gene}"[Title/Abstract] AND {mesh}'
        results[label] = search_pubmed(query)
        time.sleep(0.25)   # be gentle on NCBI
    return results

## PUBMED_COUNT: COUNT PAPERS 
def pubmed_count(genes):
    rows = []
    for g in genes:
        gene = g.lower()  # normalize gene names to lowercase
        total = count_papers(gene)
        ctx_counts = count_papers_by_context(gene)
        rows.append({"gene": gene, "total": total, **ctx_counts})
        time.sleep(0.4)

    df = pd.DataFrame(rows, columns=["gene","total","mouse","human","cell line","drosophila"])

    print(df)
    return df




# def goterm(df):

#     # --- Add GO annotations using MyGene ---
#     mg = mygene.MyGeneInfo()
#     genes_for_go = df["gene"].str.upper().unique().tolist()  # use uppercase for standard symbols

#     res = mg.querymany(
#         genes_for_go,
#         scopes="symbol,alias",
#         fields="go.BP.term,go.MF.term,go.CC.term",
#         species="human")

#     df_go = pd.DataFrame(res)

#     df_go = df_go.dropna(subset=["query"]).drop_duplicates("query", keep="first")

#     if "symbol" not in df_go.columns:
#         df_go["symbol"] = df_go["query"]

#     # make everything lowercase for comparison
#     df_go["gene"] = df_go["query"].str.lower()
#     df_go["gene_matched"] = df_go["symbol"].str.lower()

#     # create alias column: only fill when the matched gene ≠ what was queried
#     df_go["alias"] = df_go.apply(lambda r: r["query"] if r["gene"] != r["gene_matched"] else None, axis=1)

#     if "go" not in df_go.columns: df_go["go"] = None

#     # GO terms (handle missing dicts and None lists)
#     df_go["GO_Biological_Process"] = df_go["go"].apply(
#         lambda g: [d.get("term") for d in (g.get("BP", []) if isinstance(g, dict) else []) if isinstance(d, dict) and d.get("term")] or None
#     )
#     df_go["GO_Molecular_Function"] = df_go["go"].apply(
#         lambda g: [d.get("term") for d in (g.get("MF", []) if isinstance(g, dict) else []) if isinstance(d, dict) and d.get("term")] or None
#     )
#     df_go["GO_Cellular_Component"] = df_go["go"].apply(
#         lambda g: [d.get("term") for d in (g.get("CC", []) if isinstance(g, dict) else []) if isinstance(d, dict) and d.get("term")] or None
#     )

#     df_go = df_go[["gene", 'alias', 'gene_matched', "GO_Biological_Process","GO_Molecular_Function","GO_Cellular_Component"]]

#     df = df.merge(df_go, on="gene", how="left")

#     print(df)
#     return df

def goterm(df):
    import os, urllib.request
    import pandas as pd
    import mygene
    from goatools.obo_parser import GODag
    from goatools.mapslim import mapslim

    # --- 1) MyGene: get GO with IDs & terms ---
    mg = mygene.MyGeneInfo()
    genes_for_go = df["gene"].str.upper().unique().tolist()

    res = mg.querymany(
        genes_for_go,
        scopes="symbol,alias",
        # fields=["ensembl.gene", "go.BP", "go.MF", "go.CC"],  # includes IDs + terms
        fields=["symbol", "ensembl.gene", "go.BP", "go.MF", "go.CC"],

        species="human"
    )

    print("\n🔍 MyGene query results (first few):")
    for r in res[:10]:  # show first 10 hits
        print(f"  query={r.get('query')} → matched={r.get('symbol')}  (ensembl={r.get('ensembl')})")



    df_go = pd.DataFrame(res).dropna(subset=["query"]).drop_duplicates("query", keep="first")
    if "symbol" not in df_go.columns:
        df_go["symbol"] = df_go["query"]

    def extract_ensembl(obj):
        if isinstance(obj, dict):
            return obj.get("gene")
        if isinstance(obj, list) and len(obj) > 0 and isinstance(obj[0], dict):
            return obj[0].get("gene")
        return None

    df_go["ensembl_id"] = df_go["ensembl"].apply(extract_ensembl)


    # normalize/identity columns
    df_go["gene"] = df_go["query"].str.lower()
    df_go["gene_matched"] = df_go["symbol"].str.lower()
    df_go["alias"] = df_go.apply(lambda r: r["query"] if r["gene"] != r["gene_matched"] else None, axis=1)
    if "go" not in df_go.columns:  # safety
        df_go["go"] = None

    # helpers to unpack MyGene blocks
    def unpack_ids(go_block, aspect):
        if not isinstance(go_block, dict): return []
        return [d.get("id") for d in go_block.get(aspect, []) if isinstance(d, dict) and d.get("id")]

    def unpack_terms(go_block, aspect):
        if not isinstance(go_block, dict): return None
        terms = [d.get("term") for d in go_block.get(aspect, []) if isinstance(d, dict) and d.get("term")]
        return terms if terms else None

    # original detailed lists (KEEPING THESE)
    df_go["GO_Biological_Process"]  = df_go["go"].apply(lambda g: unpack_terms(g, "BP"))
    df_go["GO_Molecular_Function"]  = df_go["go"].apply(lambda g: unpack_terms(g, "MF"))
    df_go["GO_Cellular_Component"]  = df_go["go"].apply(lambda g: unpack_terms(g, "CC"))

    # IDs for slim mapping
    df_go["BP_ids"] = df_go["go"].apply(lambda g: unpack_ids(g, "BP"))
    df_go["MF_ids"] = df_go["go"].apply(lambda g: unpack_ids(g, "MF"))
    df_go["CC_ids"] = df_go["go"].apply(lambda g: unpack_ids(g, "CC"))

    # --- 2) Load full GO + Generic GO-slim (official) ---
    cache = os.path.join(os.path.expanduser("~"), ".cache", "go_slim")
    os.makedirs(cache, exist_ok=True)
    go_obo = os.path.join(cache, "go-basic.obo")
    slim_obo = os.path.join(cache, "goslim_generic.obo")

    if not os.path.exists(go_obo):
        urllib.request.urlretrieve("http://purl.obolibrary.org/obo/go/go-basic.obo", go_obo)
    if not os.path.exists(slim_obo):
        urllib.request.urlretrieve("http://current.geneontology.org/ontology/subsets/goslim_generic.obo", slim_obo)

    GO = GODag(go_obo)
    SLIM = GODag(slim_obo)

    # --- 3) Map to slim per namespace (multiple allowed) ---
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

    # --- 5) Add a single summary column: the dominant Biological Process slim term ---
    # def most_frequent_term(term_list):
    #     """Return the most frequent (dominant) BP slim term per gene, or None."""
    #     if not isinstance(term_list, list) or len(term_list) == 0:
    #         return None
    #     # Flatten and count occurrences (if multiple identical terms per gene)
    #     from collections import Counter
    #     counts = Counter(term_list)
    #     # Pick the most frequent term; tie-break alphabetically for consistency
    #     top = sorted(counts.items(), key=lambda x: (-x[1], x[0]))[0][0]
    #     return top

    # df_go["GO_Slim_BP_Most_Frequent"] = df_go["GO_Slim_BP_all"].apply(most_frequent_term)

    # (assumes you already have: GO, SLIM, and df_go["BP_ids"])

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

    from collections import Counter
    df_go["GO_Slim_BP_Most_Frequent"] = df_go["GO_Slim_BP_all"].apply(
        lambda lst: sorted(Counter(lst).items(), key=lambda x: (-x[1], x[0]))[0][0] if lst else None
)




    # CSV-friendly versions (nice in Excel)
    for col in ["GO_Slim_BP", "GO_Slim_MF", "GO_Slim_CC",
                "GO_Biological_Process", "GO_Molecular_Function", "GO_Cellular_Component"]:
        df_go[col + "_csv"] = df_go[col].apply(lambda x: ", ".join(x) if isinstance(x, list) else None)

    # --- 4) Keep EVERYTHING you want, merge back ---
    df_go = df_go[[
        "gene", "alias", "gene_matched", 'ensembl_id',
        # original MyGene detail (lists)
        "GO_Biological_Process", "GO_Molecular_Function", "GO_Cellular_Component",
        # slim per namespace (lists)
        "GO_Slim_BP", "GO_Slim_MF", "GO_Slim_CC",
        # csv helpers for all 6 columns above
        "GO_Biological_Process_csv", "GO_Molecular_Function_csv", "GO_Cellular_Component_csv",
        "GO_Slim_BP_csv", "GO_Slim_MF_csv", "GO_Slim_CC_csv", "GO_Slim_BP_Most_Frequent"
    ]]

    out = df.merge(df_go, on="gene", how="left")
    print(out)
    return out



df = goterm(df)


def disease_association(df):
    import requests
    import time

    def get_diseases(ensembl_id):
        """Query OpenTargets for disease associations by Ensembl ID."""
        if not isinstance(ensembl_id, str) or not ensembl_id.startswith("ENSG"):
            return None

        url = "https://api.platform.opentargets.org/api/v4/graphql"
        query = """
        query diseaseAssoc($ensembl: String!) {
          target(ensemblId: $ensembl) {
            associatedDiseases {
              rows {
                disease {
                  name
                }
              }
            }
          }
        }
        """
        try:
            response = requests.post(url, json={"query": query, "variables": {"ensembl": ensembl_id}})
            if response.status_code != 200:
                return None
            
            data = response.json()
            target = data.get("data", {}).get("target")
            if not target or not target.get("associatedDiseases"):
                return None

            rows = target["associatedDiseases"].get("rows", [])
            # only keep strong associations (score >= 0.5)
            strong = [
                r["disease"]["name"]
                for r in rows
                if r.get("disease") and r.get("association_score", 0) >= 0.5
            ]

            return sorted(set(strong)) if strong else None

        except Exception as e:
            print(f"Error fetching {ensembl_id}: {e}")
            return None

    disease_results = []
    for ensembl in df["ensembl_id"]:
        diseases = get_diseases(ensembl)
        disease_results.append(diseases)
        time.sleep(0.3)  # pause 0.3 seconds between queries

    df["disease_association"] = disease_results

    return df



# def disease_association(df):
#     import requests
#     import time

#     # ⬇️ You’ll need to paste your DisGeNET API token here
#     DISGENET_TOKEN = "YOUR_DISGENET_API_KEY"

#     def get_diseases(ensembl_id):
#         """Query DisGeNET for curated gene-disease associations."""
#         if not isinstance(ensembl_id, str) or not ensembl_id.startswith("ENSG"):
#             return None

#         url = f"https://www.disgenet.org/api/gda/gene/{ensembl_id}"
#         headers = {"Authorization": f"Bearer {DISGENET_TOKEN}"}

#         try:
#             response = requests.get(url, headers=headers)
#             if response.status_code != 200:
#                 print(f"Error fetching {ensembl_id}: {response.status_code}")
#                 return None

#             data = response.json()
#             # Each entry typically includes: disease_name, score, disease_type, etc.
#             diseases = [
#                 d["disease_name"]
#                 for d in data
#                 if d.get("score", 0) >= 0.3  # optional threshold for stronger associations
#             ]

#             return sorted(set(diseases)) if diseases else None

#         except Exception as e:
#             print(f"Error fetching {ensembl_id}: {e}")
#             return None

#     # same loop as before
#     disease_results = []
#     for ensembl in df["ensembl_id"]:
#         diseases = get_diseases(ensembl)
#         disease_results.append(diseases)
#         time.sleep(0.3)  # polite delay between requests

#     df["disease_association"] = disease_results
#     return df

## DisGeNET look into their scoring sysrem 


df = disease_association(df)




# def orthologue(df):

#     import requests

#     SESSION = requests.Session()
#     SESSION.headers.update({
#         "User-Agent": "Mozilla/5.0",
#         "Referer": "https://www.flyrnai.org/diopt",
#         "Origin": "https://www.flyrnai.org",
#     })
#     # warm up cookies
#     SESSION.get("https://www.flyrnai.org/diopt", timeout=20)

#     def get_ortholog(gene_symbol):
#         from io import StringIO
#         import re
#         import pandas as pd

#         url = "https://www.flyrnai.org/cgi-bin/DRSC_orthologs.pl"
#         form = {
#             "input_species": "9606",
#             "output_species": "7227",
#             "search_fields": "***",
#             "gene_list": gene_symbol,
#             "gene_file": "",
#             "submit": "Submit Search",
#             "search_datasets": "All",
#             "additional_filter": "NoLow",
#             "output": "tsv"}

#         r = SESSION.post(url, data=form, timeout=30)

#         print(f"{gene_symbol}: HTTP {r.status_code}, html_len={len(r.text)}")

#         # try to parse all HTML tables on the page
#         try:
#             tables = pd.read_html(r.text)
#             print(f"{gene_symbol}: found {len(tables)} table(s)")
#             # print the first few tables’ columns so we can see if the ortholog table is there
#             for i, t in enumerate(tables[:3]):
#                 print(f"  table[{i}] cols -> {list(t.columns)}")
#         except Exception as e:
#             print(f"{gene_symbol}: read_html failed: {e}")

#         if r.status_code != 200:
#             print(f"⚠️ Failed for {gene_symbol}")
#             return None

#         # optional: print a few lines to confirm we get TSV text
#         print(f"{gene_symbol}: response length={len(r.text)}")
#         from bs4 import BeautifulSoup

#         # 👇 parse the HTML table
#         soup = BeautifulSoup(r.text, "html.parser")
#         tbl = soup.find("table", id="results-table")
#         if not tbl:
#             print(f"{gene_symbol}: no results-table found")
#             return None

#         # headers
#         headers = [th.get_text(strip=True).replace("\xa0", " ") for th in tbl.select("thead th")]

#         # first data row
#         row = tbl.select_one("tbody tr")
#         cells = [td.get_text(strip=True) for td in row.find_all("td")]

#         record = dict(zip(headers, cells))


# def orthologue(df):

#     import requests
#     from bs4 import BeautifulSoup

#     SESSION = requests.Session()
#     SESSION.headers.update({
#         "User-Agent": "Mozilla/5.0",
#         "Referer": "https://www.flyrnai.org/diopt",
#         "Origin": "https://www.flyrnai.org",
#     })
#     # warm up cookies
#     SESSION.get("https://www.flyrnai.org/diopt", timeout=20)

#     def get_ortholog(gene_symbol):
#         import pandas as pd

#         url = "https://www.flyrnai.org/cgi-bin/DRSC_orthologs.pl"
#         form = {
#             "input_species": "9606",
#             "output_species": "7227",
#             "search_fields": "***",
#             "gene_list": gene_symbol,
#             "gene_file": "",
#             "submit": "Submit Search",
#             "search_datasets": "All",
#             "additional_filter": "NoLow",
#             "output": "tsv",
#         }

#         r = SESSION.post(url, data=form, timeout=30)
#         print(f"{gene_symbol}: HTTP {r.status_code}, html_len={len(r.text)}")

#         # try to parse all HTML tables on the page (just for visibility)
#         try:
#             tables = pd.read_html(r.text)
#             print(f"{gene_symbol}: found {len(tables)} table(s)")
#             for i, t in enumerate(tables[:1]):
#                 print(f"  table[{i}] cols -> {list(t.columns)}")
#         except Exception as e:
#             print(f"{gene_symbol}: read_html failed: {e}")

#         if r.status_code != 200:
#             print(f"⚠️ Failed for {gene_symbol}")
#             return None

#         # parse the results table directly
#         soup = BeautifulSoup(r.text, "html.parser")
#         tbl = soup.find("table", id="results-table")
#         if not tbl:
#             print(f"{gene_symbol}: no results-table found")
#             return None

#         headers = [th.get_text(strip=True).replace("\xa0", " ") for th in tbl.select("thead th")]
#         row = tbl.select_one("tbody tr")
#         if not row:
#             print(f"{gene_symbol}: no data row")
#             return None

#         cells = [td.get_text(strip=True) for td in row.find_all("td")]
#         record = dict(zip(headers, cells))

#         # ADD: keep the <td> elements so we can grab the alignment cell
#         tds = row.find_all("td")


#         # --- ADD: fetch Identity/Similarity via the "Alignment & Scores" form ---
#         diopt_identity_pct = None
#         diopt_similarity_pct = None
#         try:
#             align_idx = headers.index("Alignment & Scores")
#             align_td = tds[align_idx]
#             form_tag = align_td.find("form")
#             if form_tag and form_tag.get("action"):
#                 from requests.compat import urljoin

#                 action_url = urljoin("https://www.flyrnai.org", form_tag.get("action"))
#                 payload = {
#                     inp.get("name"): inp.get("value", "")
#                     for inp in form_tag.find_all("input")
#                     if inp.get("name")
#                 }

#                 # helpful: set a Referer to mimic the browser
#                 align_resp = SESSION.post(action_url, data=payload, timeout=30, headers={"Referer": r.url})
#                 txt = align_resp.text

#                 import re
#                 # Try BLAST-style: "Identities = 123/456 (27%)", "Positives = ... (xx%)"
#                 m_id = re.search(r'Identit(?:y|ies)\s*[:=].*?\(([\d.]+)\s*%\)', txt, re.I|re.S)
#                 m_sm = re.search(r'(?:Similarity|Positives)\s*[:=].*?\(([\d.]+)\s*%\)', txt, re.I|re.S)

#                 # Fallback: "Identity: 27%" / "Similarity: 35%"
#                 if not m_id:
#                     m_id = re.search(r'Identity\s*[:=]\s*([\d.]+)\s*%', txt, re.I)
#                 if not m_sm:
#                     m_sm = re.search(r'(?:Similarity|Positives)\s*[:=]\s*([\d.]+)\s*%', txt, re.I)

#                 if m_id:
#                     diopt_identity_pct = float(m_id.group(1))
#                 if m_sm:
#                     diopt_similarity_pct = float(m_sm.group(1))

#                 # debug prints so you can see what we got
#                 print(f"Alignment: identity={diopt_identity_pct}, similarity={diopt_similarity_pct}")
#             else:
#                 print("No alignment form found in cell.")
#         except Exception as e:
#             print(f"Alignment fetch failed: {e}")




#         # print just the fields we care about
#         wanted = [
#             "Fly Symbol",
#             "Fly Species Gene ID",
#             "Rank",
#             "Weighted Score",
#             "Best Score",
#             "Best Score Reverse",
#         ]
#         print(f"\n{gene_symbol} parsed columns:")
#         for k in wanted:
#             print(f"  {k}: {record.get(k)}")

#         # return a small dict (strings are fine for now)
#         return {
#             "diopt_fly_symbol":         record.get("Fly Symbol"),
#             "diopt_fbgn":               record.get("Fly Species Gene ID"),
#             "diopt_rank":               record.get("Rank"),
#             "diopt_weighted_score":     record.get("Weighted Score"),
#             "diopt_best_score":         record.get("Best Score"),
#             "diopt_best_score_reverse": record.get("Best Score Reverse"),
#             "diopt_identity_pct":       diopt_identity_pct,
#     "diopt_similarity_pct":     diopt_similarity_pct,
#         }

#     # ---- ONE-OFF TEST CALL (kept INSIDE the function, as you asked) ----
#     print("\n🔎 Testing DIOPT for SHANK3 …")
#     test_result = get_ortholog("SHANK3")
#     print("\nReturned dict:", test_result)

#     # do nothing else for now; keep your pipeline unchanged
#     return df


def orthologue(df):
    import requests, time, re
    import pandas as pd
    from bs4 import BeautifulSoup

    WANT_COLS = [
        "diopt_fly_symbol", "diopt_fbgn",
        "diopt_rank", "diopt_weighted_score",
        "diopt_best_score", "diopt_best_score_reverse",
        "diopt_identity_pct", "diopt_similarity_pct",
    ]

    def _empty():
        return {k: None for k in WANT_COLS}

    def _to_float(x):
        try:
            return float(str(x).replace(",", "").strip())
        except Exception:
            return None

    SESSION = requests.Session()
    SESSION.headers.update({
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://www.flyrnai.org/diopt",
        "Origin":  "https://www.flyrnai.org",
    })
    try:
        SESSION.get("https://www.flyrnai.org/diopt", timeout=20)
    except Exception:
        pass

    def get_ortholog(symbol):
        url = "https://www.flyrnai.org/cgi-bin/DRSC_orthologs.pl"
        form = {
            "input_species": "9606",
            "output_species": "7227",
            "search_fields": "***",
            "gene_list": symbol,
            "gene_file": "",
            "submit": "Submit Search",
            "search_datasets": "All",
            "additional_filter": "NoLow",
            "output": "tsv",
        }
        try:
            r = SESSION.post(url, data=form, timeout=30)
            if r.status_code != 200:
                return _empty()

            soup = BeautifulSoup(r.text, "html.parser")
            tbl = soup.find("table", id="results-table")
            if not tbl:
                return _empty()

            headers = [th.get_text(strip=True).replace("\xa0", " ")
                       for th in tbl.select("thead th")]
            row = tbl.select_one("tbody tr")
            if not row:
                return _empty()

            tds   = row.find_all("td")
            cells = [td.get_text(strip=True) for td in tds]
            rec   = dict(zip(headers, cells))

            out = {
                "diopt_fly_symbol":         rec.get("Fly Symbol"),
                "diopt_fbgn":               rec.get("Fly Species Gene ID"),
                "diopt_rank":               rec.get("Rank"),
                "diopt_weighted_score":     _to_float(rec.get("Weighted Score")),
                "diopt_best_score":         rec.get("Best Score"),
                "diopt_best_score_reverse": rec.get("Best Score Reverse"),
                "diopt_identity_pct":       None,
                "diopt_similarity_pct":     None,
            }

            # Alignment & Scores (optional)
            try:
                aidx = headers.index("Alignment & Scores")
                form_tag = tds[aidx].find("form")
                if form_tag and form_tag.get("action"):
                    action_url = requests.compat.urljoin("https://www.flyrnai.org",
                                                         form_tag.get("action"))
                    payload = {
                        inp.get("name"): inp.get("value", "")
                        for inp in form_tag.find_all("input")
                        if inp.get("name")
                    }
                    r2 = SESSION.post(action_url, data=payload, timeout=30,
                                      headers={"Referer": r.url})
                    txt = r2.text
                    m_id = (re.search(r'Identit(?:y|ies)\s*[:=].*?\(([\d.]+)\s*%\)', txt, re.I|re.S)
                            or re.search(r'Identity\s*[:=]\s*([\d.]+)\s*%', txt, re.I))
                    m_sm = (re.search(r'(?:Similarity|Positives)\s*[:=].*?\(([\d.]+)\s*%\)', txt, re.I|re.S)
                            or re.search(r'(?:Similarity|Positives)\s*[:=]\s*([\d.]+)\s*%', txt, re.I))
                    if m_id: out["diopt_identity_pct"]   = float(m_id.group(1))
                    if m_sm: out["diopt_similarity_pct"] = float(m_sm.group(1))
            except Exception:
                pass

            return out

        except Exception:
            return _empty()

    results = [get_ortholog(str(g).upper()) for g in df["gene"]]
    diopt_df = pd.DataFrame(results, columns=WANT_COLS)
    return pd.concat([df.reset_index(drop=True), diopt_df.reset_index(drop=True)], axis=1)



# genes  = [
#     "CHD8", "SCN2A", "SHANK3", "SYNGAP1", "ADNP", "DYRK1A", "ARID1B", "TBR1", "PTEN", "GRIN2B",
#     "DDX3X", "POGZ", "ASH1L", "ANK2", "ANKRD11", "SETD1A", "GRIN2A", "TRIO", "SP4", "CACNA1G",
#     "GRIA3", "HERC1", "RB1CC1", "XPO7", "CUL1", "AKAP11", "STAG1", "ZNF136", "CACNA1C", "DRD2",
#     "KDM5B", "SORCS3", "SEMA6D", "DUSP6", "PTPRF", "CTNNA2", "FBXO11", "PTPRD", "NRXN1", "FOXP2",
#     "ADGRL3", "SLC6A3", "DRD4", "GNB2L1", "SCN1A", "KCNQ2", "SCN8A", "STXBP1", "CDKL5", "PCDH19",
#     "CHD2", "DEPDC5", "KCNT1", "GABRA1", "GABRB3", "GABRG2", "SLC2A1", "TCF4", "MECP2", "EHMT1",
#     "SATB2", "KMT2D", "SMC1A", "WDR45", "CTNNB1", "FOXP1"]

genes  = [
    "CHD8", "SCN2A", "SHANK3", 'GNB2L1']

directory = '/Users/cochral/Desktop'
df = pubmed_count(genes)

df = orthologue(df)




output = os.path.join(directory, 'gene_pubmed_search.csv')
df.to_csv(output, index=False)




## MERGE TWO DF - ORIGINAL DF AND NEW DF WITH PAPERS AND GOTERMS

# df_pubmed = pd.read_csv('/Users/cochral/Desktop/gene_pubmed_search.csv')
# df_genes = pd.read_excel('/Users/cochral/Downloads/example.xlsx')

# df_genes["Gene"] = df_genes["Gene"].str.upper()
# df_pubmed["gene"] = df_pubmed["gene"].str.upper()

# df = df_pubmed.merge(df_genes, left_on="gene", right_on="Gene", how="left")
# df = df.drop(columns=["Gene"])

# print(df)








