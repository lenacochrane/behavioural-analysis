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

Entrez.email = "lena.cochrane@crick.ac.uk"

# ------------------------------------------------------------------------------
# SEARCH_PUBMED: run a PubMed search and return the number of matching papers
# ------------------------------------------------------------------------------
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

# ------------------------------------------------------------------------------
# COUNT_PAPERS: total number of papers mentioning the gene
# ------------------------------------------------------------------------------
def count_papers(gene):
    return search_pubmed(f'"{gene}"[Title/Abstract]')

# ------------------------------------------------------------------------------
# COUNT_PAPERS_BY_CONTEXT: number of papers in each context
# ------------------------------------------------------------------------------
def count_papers_by_context(gene, contexts):
    """search a gene in a given context: 
    dictionary key = column name, item = actual search term 
    "mouse": '"Mice"[MeSH Terms]'"""
    results = {}
    for label, mesh in contexts.items():
        query = f'"{gene}"[Title/Abstract] AND {mesh}'
        results[label] = search_pubmed(query)
        time.sleep(0.25)   # be gentle on NCBI
    return results

# ------------------------------------------------------------------------------
# PUBMED_COUNT: total number of papers mentioning the gene in a given context 
# ------------------------------------------------------------------------------
def pubmed_count(df, sleep_s=0.5,  include_aliases=True):
    """
    Compute PubMed counts per gene and context.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain:
          - 'gene' (UPPERCASE)
          - 'aliases' (list[str]) from add_aliases()
          - 'mouse_symbol' and 'fly_symbol' from orthologue()
    include_aliases : bool
        If True, build queries from the alias list.
        If False, build queries from the official gene name only.
        Mouse/fly symbols are always considered only within their species filters.
    """
    
    # model_mesh: search for gene in biological mechanisms 
    MODEL_MESH = (
        '"Disease Models, Animal"[MeSH Terms] OR '
        '"Models, Animal"[MeSH Terms] OR '
        '"Cell Line"[MeSH Terms] OR '
        '"Organoids"[MeSH Terms]')

    # contexts: dictionary key = column name, item = actual search term
    contexts = {
        'disease model': MODEL_MESH,
        "mouse": '"Mice"[MeSH Terms]',
        "human": '"Humans"[MeSH Terms]',
        "cell line": '"Cell Line"[MeSH Terms] OR "Cells, Cultured"[MeSH Terms]',
        "drosophila": '"Drosophila melanogaster"[MeSH Terms]'}
    
    # tiab: search in title or abstract
    def make_tiab(terms):
        # ("term1"[TIAB] OR "term2"[TIAB] ...)
        return "(" + " OR ".join(f"\"{t}\"[Title/Abstract]" for t in terms) + ")"

    # data = []

    df["aliases_used"] = None
    df["total"] = None
    df['disease model'] = None
    df["mouse"] = None
    df['cell line'] = None
    df['drosophila'] = None
    df["human"] = None
    df["specific_disease"] = None

    # for row in df.itertuples(index=False):
    for i, row in enumerate(df.itertuples(index=False)):
        
        # include_aliases: use all aliases or gene name 
        if include_aliases:
            search = list(getattr(row, "aliases", None) or [row.gene])
        else:
            search = [row.gene]  # only gene name # if aliases=False will run only the gene names - 

        ## total paper mentions
        base = make_tiab(search)
        total = search_pubmed(base)

        # context_counts: counts within a specific context
        context_counts = {}
        for label, mesh in contexts.items():
            if label == "mouse":
                # aliases OR mouse_symbol, but only within mouse-gated search
                terms = search.copy()
                msym = getattr(row, "mouse_symbol", None)

                if isinstance(msym, str):
                    s = msym.strip()
                    if len(s) > 2 and s not in terms: # append new aliases and avoid 2-letter inflators like 'da', 'ca'
                        terms.append(s)

                q = f"{make_tiab(terms)} AND ({mesh})"

            elif label == "drosophila":
                # aliases OR fly_symbol, but only within drosophila-gated search
                terms = search.copy()
                fsym = getattr(row, "fly_symbol", None)

                if isinstance(fsym, str):
                    s = fsym.strip()
                    if len(s) > 2 and s not in terms: # append new aliases and avoid 2-letter inflators like 'da', 'ca'
                        terms.append(s)

                q = f"{make_tiab(terms)} AND ({mesh})"

            else:
                # other contexts
                q = f"{base} AND ({mesh})"

            context_counts[label] = search_pubmed(q)
            time.sleep(sleep_s / 2)  # gentle delay between context queries
            

        diseases = []
        if row.disgenet_associations:
            diseases.extend(row.disgenet_associations)

        if row.Disease_disgenet_matched:
            diseases.extend(row.Disease_disgenet_matched)
        print(diseases)


        if diseases:
            disease_queries = [
                f'({make_tiab(search)}) AND ("{d}"[Title/Abstract])'
                for d in diseases
            ]
            disease_hits = sum(search_pubmed(q) or 0 for q in disease_queries)
        else:
            disease_hits = 0

        # context_counts["specific_disease"] = disease_hits
        # time.sleep(sleep_s)

        # data.append({
        #     "gene": row.gene,
        #     "aliases_used": search,
        #     "total": total,
        #     **context_counts,
        # })

            # --- write results directly into dataframe ---
        df.at[i, "aliases_used"] = search
        df.at[i, "total"] = total
        df.at[i, "disease model"] = context_counts.get("disease model")
        df.at[i, "mouse"] = context_counts.get("mouse")
        df.at[i, "human"] = context_counts.get("human")
        df.at[i, "cell line"] = context_counts.get("cell line")
        df.at[i, "drosophila"] = context_counts.get("drosophila")
        df.at[i, "specific_disease"] = disease_hits

        time.sleep(sleep_s)

    # pubmed_df = pd.DataFrame(
    #     data,
    #     columns=["gene", "aliases_used", "total", 'disease model', "mouse", "human", "cell line", "drosophila", 'specific_disease']
    # )
    # # df = df.merge(pubmed_df, on="gene", how="left", validate="one_to_one")
    # df = df.merge(pubmed_df, on="gene", how="left")
    return df



# ------------------------------------------------------------------------------
# ORTHOLOGUE: identifies mouse and fly orthologues
# ------------------------------------------------------------------------------
def orthologue(genes):

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
    for g in genes:
        human = str(g).upper()
        mouse_sym = mouse_ortholog(human)
        fly_rec   = fly_ortholog(human)
        rows.append({"gene": human, "mouse_symbol": mouse_sym, **fly_rec})

    df = pd.DataFrame(rows, columns=["gene", "mouse_symbol"] + WANT_COLS)
    return df

# ------------------------------------------------------------------------------
# GOTERM: identifies gene ontology profiles for each gene 
# ------------------------------------------------------------------------------ 
def goterm(df):

    """ Retrieve GO annotations and map them to GO-slim terms for each gene 
        Uses MyGene 

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain:
          - 'gene' : str

    """

    # 1. MYGENE: identifies go info relating to human gene: bp (biological process), mf (molecular function) + cc (cellular component)
    mg = mygene.MyGeneInfo()
    genes_for_go = df["gene"].str.upper().unique().tolist()

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


    df_go["gene"] = df_go["query"].str.upper() # uppercase gene name
    df_go["gene_matched"] = df_go["symbol"].str.upper() # uppercase canonical symbol returned by MyGene (for comparison)
    df_go["alias"] = df_go.apply(lambda r: r["query"] if r["gene"] != r["gene_matched"] else None, axis=1) # if query differs from canonical 
    
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
        "gene", "gene_matched", 'ensembl_id',
        "GO_Biological_Process", "GO_Molecular_Function", "GO_Cellular_Component",
        "GO_Slim_BP", "GO_Slim_MF", "GO_Slim_CC", "GO_Slim_BP_Most_Frequent"]]

    out = df.merge(df_go, on="gene", how="left")
    print(out)
    return out

# ------------------------------------------------------------------------------
# ADD_ALIASES: for each gene fetch official symbol and synonyms
# ------------------------------------------------------------------------------
def add_aliases(df, max_terms=10):
    
    # CLEAN_ALIASES: clean aliases before being used 
    def clean_aliases(alias_list):
        if not alias_list:
            return []

        chromosome = re.compile(r'^(?:del|dup|ins|inv)?\s*(?:\d{1,2}|[xy])[pq]\s*[\d\.\-\s]+$',re.IGNORECASE) # del 22q13.1
        hs = re.compile(r'^\s*Hs\.\s*[-.]?\s*\d+\s*$', re.IGNORECASE) # Hs.1234
        BAD_ALIASES = {"DEP.5", "DEP. 5", "DEP.-5", "FFEVF", "FFEVF1", "BRIGHT", "6A35", 'seven transmembrane helix receptor',
            'transcription factor-4', 'transcription factor4', 'CDG2T', 'E22', 'FCD2', 'EPND', 'FESD', 'MRD6', 'CC1', "ATG17", 'NET1',
            'NAT1', 'PKDYS', 'PKDYS1', 'EPEDD', 'EPEO2', 'NEDSID', 'Se t1', 'MRD5', 'RASA1', 'NBR13', 'C5DELq14. 3', 'C5DELq14.-3', 
            'C5DELq14.3', 'NEDHSIL', 'CL3' }
        

        cleaned = []
        for a in alias_list:
            s = str(a).strip()
            s_norm = re.sub(r'([pq])\s+', r'\1', s, flags=re.IGNORECASE) # normalise space after p/q so cytogenetic regexes catch them
            if chromosome.match(s_norm):  
                continue
            if hs.match(s_norm): 
                continue
            if s_norm.upper() in {b.upper() for b in BAD_ALIASES}:
                continue 
            # 1) p-prefixed constructs (with optional leading '-' or space)
            if re.match(r'^[\-\s]*p\d{2,5}$', s_norm, flags=re.IGNORECASE):
                continue
            # Drop short all-caps alphanumeric blocks split by space or hyphen
            if re.match(r'^[A-Z0-9]{2,5}[ -][A-Z0-9]{1,4}$', s_norm):
                continue
            # 2) orphan leading hyphen artifacts like '-foo' (often parsing noise)
            if s_norm.startswith('-'):
                continue
            # remove very short aliases (<=3 chars) ## if gene name is less than 3 will get picked up again ltr so idc # new one here shouldnt do that ifit includes a number so 
            # only drop if it's ALL letters and very short (≤3)
            if s_norm.isalpha() and len(s_norm) <= 3:
                continue
            # remove short alphanumeric blocks split by space/hyphen (e.g. BFI C3, BFI-C3) 
            if re.match(r'^[A-Z0-9]{2,5}[ -][A-Z0-9]{1,4}$', s_norm):
                continue
            if re.search(r'\bDEL\w*[pq]?\s*\d', s_norm, re.I):
                continue
            # 1) Cytogenetic / deletion-like patterns e.g. C5DELq14.3, DEL22q13
            if re.search(r'\bDEL[qQ]\d+', s_norm) or re.search(r'\bDEL\w*[pq]\d+', s_norm):
                continue
            # catch cytogenetic locus with suffix/prefix ops like 10q23del, 10q23 deletion, DEL10q23, etc.
            if (
                re.search(r'(?i)\b(?:del|dup|ins|inv)\s*(?:\d{1,2}|[xy])[pq]\d+(?:\.\d+)?\b', s_norm)
                or re.search(r'(?i)\b(?:\d{1,2}|[xy])[pq]\d+(?:\.\d+)?\s*(?:del|dup|ins|inv)\b', s_norm)
                or re.search(r'(?i)\b(?:\d{1,2}|[xy])[pq]\d+(?:\.\d+)?\s*(?:deletion|duplication|insertion|inversion)\b', s_norm)
            ):
                continue
            # 2) Short split alphanumerics like G T24 / G-T24 / C T31 / C-T31
            if re.match(r'^[A-Z]{1,2}[ -][A-Z]?\d{1,3}$', s_norm):
                continue
            if re.match(r'^[A-Z]{1,2}[ -][A-Z]?\d{1,3}$', s_norm):
                continue
            # drop obvious disease-style labels (DEExxx, EIEExxx, etc.)
            if re.match(r'^(DEE|EIEE)\s*-?\s*\d{1,3}$', s_norm, flags=re.IGNORECASE):
                continue
            
            cleaned.append(a)
        return list(dict.fromkeys(cleaned))  # deduplicate while keeping order
    

    # MyGene: pull all known synonoms; the response (hits) is a list of dictionaries
    mg = mygene.MyGeneInfo()
    syms = df["gene"].astype(str).str.upper().unique().tolist()
    fields = ["symbol","name","alias","other_names","prev_symbol","alias_symbol"]
    hits = mg.querymany(syms, scopes="symbol", fields=fields, species="human", as_dataframe=False)
    
    # generates alternative forms of the alias ["neurexin 1", "neurexin-1", "neurexin1"]
    def variants(s):
        s = s.strip()
        if not s: return []
        out = {s}
        m = re.match(r"^(.*?)[ -]?(\d+|[A-Za-z]\d+)$", s)  # e.g. "neurexin 1" / "neurexin-1"
        if m:
            base, tail = m.groups()
            out |= {f"{base}-{tail}", f"{base} {tail}", f"{base}{tail}"}
        return list(out)

    alias_map = {}
    for h in hits:
        q = (h.get("query") or "").upper() # query is gene symbol
        if not q: 
            continue
        terms = []
        # always include official symbol
        sym = (h.get("symbol") or q).upper()
        terms.append(sym)
        # official name + synonyms
        for fld in ("name","alias","other_names","prev_symbol","alias_symbol"):
            val = h.get(fld)
            vals = val if isinstance(val, list) else [val] if isinstance(val, str) else []
            for v in vals:
                terms += variants(v)
        # dedupe, simple cleanup
        seen, out = set(), []
        for t in terms:
            if not t or len(t) < 3: continue 
            k = t.lower()
            if k not in seen:
                seen.add(k); out.append(t)
        alias_map[q] = out[:max_terms] if len(out) > max_terms else out

    df = df.copy()
    df["aliases"] = df["gene"].apply(lambda g: alias_map.get(str(g).upper(), [str(g).upper()]))
    df["aliases"] = df["aliases"].apply(clean_aliases)

    return df

# ------------------------------------------------------------------------------
# DISEASE_ASSOCIATION: identifies assocations in target disorders
# ------------------------------------------------------------------------------
def disease_association(df):
    
    """
    Requires df['Gene'];
    Screen for >= 0.7 scored associated diseases;
    Identifies if hits are within target disorders;
    Return df['disgenet_assocations']
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
            if score >= 0.7 and name: ### edit score simply for assocations 
                diseases.append(name)
        
        # Define your target disease categories
        target_diseases = [
            "Schizophrenia",
            "Autism Spectrum Disorders",
            "Attention Deficit Hyperactivity Disorder",
            "Epilepsy",
            "Intellectual Disability"
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

        gene = str(row.gene).upper()
        original_disease = row.Disease_disgenet_matched[0]

        disease_associations, score = gene_summary(gene,original_disease)
        disease_associations = list(disease_associations)

        df.at[i, "disgenet_associations"] = disease_associations          # list of >=0.7 matches in your target set
        df.at[i, "gda_score"] = score 

        time.sleep(0.75)

    df["all_diseases"] = df.apply(lambda row: 
        (row.disgenet_associations or []) + (row.Disease_disgenet_matched or []),
        axis=1)

    return df


    #genes = df["gene"].astype(str).str.upper()
    # results = []
    # for gene in genes:
    #     results.append(gene_summary(gene))
    #     time.sleep(0.75)  # wait half a second between each request

    # df["disgenet_associations"] = results


# ------------------------------------------------------------------------------
# DISEASE_ASSOCIATION_DF: identifies gene x disease assocations 
# ------------------------------------------------------------------------------

def disease_association_df(genes):

    """
    Query DisGeNET for high-level gene disease associations for all genes listed in df['Gene'].
    Returns a combined DataFrame of strong associations (score ≥ 0.7).
    """

    DISGENET_TOKEN = "1a851ae4-eae3-4209-a5f1-bdb1f4582256" # personal token

    # Create a persistent session (reuses the same connection = faster for many genes)
    session = requests.Session()
    session.headers.update({
        "Authorization": f"Bearer {DISGENET_TOKEN}",
        "Accept": "application/json"
    })


    def gene_summary(symbol):
        """
        Query DisGeNET for high-level gene disease associations for one gene.
        Returns a tidy DataFrame with summary fields.
        """
        url = f"https://api.disgenet.com/api/v1/gda/summary?gene_symbol={symbol}"
        r = session.get(url)

        # Check response and extract payload
        if r.status_code != 200:
            print(f"⚠️ Request failed for {symbol}: {r.status_code}")
            return pd.DataFrame(columns=[
                "symbolOfGene", "diseaseName",  
                "score", "el", "ei"
            ])

        payload = r.json().get("payload", [])
        print(f"Fetched {len(payload)} records for {symbol}")

        # Convert to DataFrame directly
        if not payload:
            print(f"No data found for {symbol}")
            return pd.DataFrame(columns=[
                "symbolOfGene", "diseaseName", 
                "score", "el", "ei"
            ])

        df = pd.DataFrame(payload)[[
            "symbolOfGene", "diseaseName", 
            "score", "el", "ei", 
        ]].copy()

        # Sort descending by score and remove score assocations < 0.6 tbh for now no score
        df = df[df["score"] >= 0.0].sort_values(by="score", ascending=False).reset_index(drop=True)
        print(df)

        return df
    

    genes = [g.upper() for g in genes]

    dfs = []
    for gene in genes:
        df = gene_summary(gene)
        df['Gene'] = gene
        dfs.append(df)
        time.sleep(0.75)

    disease_association_df = pd.concat(dfs, ignore_index=True)
    disease_association_df = disease_association_df[["Gene", "symbolOfGene", "diseaseName", "score", "el", "ei"]]

    return disease_association_df

# ------------------------------------------------------------------------------
# MERGE: merges the output dataframe with the original gene panel dataframe
# ------------------------------------------------------------------------------
def merge(df1, df2):
    # df2 = df2.rename(columns={"gene": "Gene"}) # newly generated df
    df2["Gene"] = df2["gene"]
    merged = df1.merge(df2, on="Gene", how="left")
    merged = merged.sort_values(by='weighted_score')
    merged["filtered_weighted_score"] = merged["weighted_score"] >= 7

    # --- create a list version of disease names --- chatgpt called them weridly 
    def expand_disease(name):
        # extract full name and abbreviation in parentheses if present
        match = re.match(r"^(.*?)\s*\((.*?)\)\s*$", name)
        if match:
            full = match.group(1).strip()
            abbr = match.group(2).strip()
            # normalize ADHD style naming (remove hyphens etc.)
            full_clean = full.replace("-", " ").replace("/", " ")
            return [full_clean, abbr.lower()]
        else:
            # no parentheses — just return cleaned full name
            return [name]

    merged["Disease_disgenet_matched"] = merged["Disease"].apply(expand_disease)
    merged = merged.dropna(subset=["gene"])
    return merged
    
# ------------------------------------------------------------------------------
# CHECK_TWO_ALIASES: checks whether pubmed search behaving as expected 
# ------------------------------------------------------------------------------
def check_two_aliases(a1, a2):
    """e.g. a1: NXR1; a2: Neurexin 1"""
    q = lambda s: f"\"{s}\"[Title/Abstract]"
    A   = search_pubmed(q(a1))
    B   = search_pubmed(q(a2))
    AND = search_pubmed(f"{q(a1)} AND {q(a2)}")
    OR  = search_pubmed(f"({q(a1)} OR {q(a2)})")
    print({"A": A, "B": B, "AND": AND, "OR": OR,
           "A_plus_B_minus_AND": (A or 0) + (B or 0) - (AND or 0)})







## genes
genes = [
    "SETD1A", "GRIN2A", "GRIA3", "CACNA1G", "SP4", "RB1CC1", "TRIO", "XPO7", "CUL1", "HERC1",
    "C4A", "DRD2", "CACNA1C", "TCF4", "ZNF804A", "CHD8", "SCN2A", "SYNGAP1", "ADNP", "DYRK1A",
    "ARID1B", "SHANK3", "GRIN2B", "TCF4", "PTEN", "DDX3X", "POGZ", "CHD2", "NRXN1", "FOXP1",
    "KDM5B", "SORCS3", "ADGRL3", "ST3GAL3", "FOXP2", "MEF2C", "PTPRF", "SEMA6D", "CTNND2",
    "GNB2L1", "SNAP25", "SLC6A3", "DRD4", "SLC6A2", "DRD5", "SCN1A", "SCN2A", "SCN8A", "KCNQ2",
    "KCNQ3", "STXBP1", "GABRA1", "GABRG2", "DEPDC5", "CDKL5", "PCDH19", "KCNT1", "SLC2A1",
    "TSC1", "TSC2", "DDX3X", "DYRK1A", "ARID1B", "KMT2A", "NSD1", "CREBBP", "TCF4", "MECP2",
    "FMR1", "SHANK3", "STXBP1", "SCN2A", "SETD5", "PPP2R5D", "POGZ"]

## genes without duplicates
genes = [
    "SETD1A", "GRIN2A", "GRIA3", "CACNA1G", "SP4", "RB1CC1", "TRIO", "XPO7", "CUL1", "HERC1",
    "C4A", "DRD2", "CACNA1C", "TCF4", "ZNF804A", "CHD8", "SCN2A", "SYNGAP1", "ADNP", "DYRK1A",
    "ARID1B", "SHANK3", "GRIN2B", "PTEN", "DDX3X", "POGZ", "CHD2", "NRXN1", "FOXP1", "KDM5B",
    "SORCS3", "ADGRL3", "ST3GAL3", "FOXP2", "MEF2C", "PTPRF", "SEMA6D", "CTNND2", "GNB2L1",
    "SNAP25", "SLC6A3", "DRD4", "SLC6A2", "DRD5", "SCN1A", "SCN8A", "KCNQ2", "KCNQ3", "STXBP1",
    "GABRA1", "GABRG2", "DEPDC5", "CDKL5", "PCDH19", "KCNT1", "SLC2A1", "TSC1", "TSC2", "KMT2A",
    "NSD1", "CREBBP", "MECP2", "FMR1", "SETD5", "PPP2R5D"]



df2 = orthologue(genes)
df1= pd.read_csv('/Volumes/lab-windingm/home/users/cochral/PhD/NDD/GENES/refined_attempt_2/gene/drosophila_behavioral_screening_gene_panel.csv')
df = merge(df1,df2) #merge to chatgpt panel 



df = disease_association(df)
df = goterm(df)
df = add_aliases(df) # might be worth here checking if aliases make sense 

df = pubmed_count(df, include_aliases=False) #

directory = '/Users/cochral/Desktop'
output = os.path.join(directory, 'disease.csv')
df.to_csv(output, index=False)



# df = disease_association_df(genes) # whole new disease x disease dataframe

# df1= pd.read_csv('/Volumes/lab-windingm/home/users/cochral/PhD/NDD/GENES/refined_attempt_2/gene/drosophila_behavioral_screening_gene_panel.csv')
# df2= pd.read_csv('/Users/cochral/Desktop/disease_testing.csv')
# df = merge(df1,df2)

 








