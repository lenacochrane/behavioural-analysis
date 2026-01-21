import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
import mplcursors  
import re
import ast
from matplotlib import cm

# ------------------------------------------------------------------------------
# WEIGHTED_ORTHOLOG_SCORE: weighted score given to orthologs
# ------------------------------------------------------------------------------
def weighted_ortholog_score(df, output_directory):

    top = (df.sort_values('weighted_score', ascending=False)
        .drop_duplicates(subset="gene_symbol", keep="first") # genes appear multiple times - remove
        .loc[:, ["gene_symbol", 'weighted_score', "similarity_percent"]] # keep these columns
        .dropna(subset=['weighted_score'])) # drop rows with missing y values
    
    gene_order = top.sort_values('weighted_score', ascending=False)["gene_symbol"].tolist()
    print(gene_order)
    top = top.set_index("gene_symbol").loc[gene_order].reset_index() # access data using gene if gene is index : )

    norm = plt.Normalize(top['similarity_percent'].min(), top['similarity_percent'].max())
    palette = [cm.YlGnBu(norm(v)) for v in top['similarity_percent'].values]


    # Convert stringified lists to actual lists if needed
    df["all_diseases"] = df["all_diseases"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith("[") else x) 

    expanded = df.explode("all_diseases").dropna(subset=["all_diseases"]) 
    expanded = expanded.drop_duplicates(subset=["gene_symbol", "all_diseases"])



    # DISEASE X GENE MEMBERSHIP MATRIX
    mat = (expanded.assign(flag=1)
            .pivot_table(index="all_diseases", columns="gene_symbol", values="flag", aggfunc="max", fill_value=0).reindex(columns=gene_order))
    
    print(mat)

    
    # FIGURE SIZE / LAYOUT 
    diseases = mat.index.tolist()
    n_genes  = len(gene_order)
    n_dis    = len(diseases)

    fig_w = max(10, 0.4 * n_genes)
    fig_h = 6 + min(2.5, 0.22 * n_dis)

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs  = GridSpec(nrows=2, ncols=1, height_ratios=[3, 0.8], hspace=0.02) # 2 subplot- above and below
    
    # FIRST SUBPLOT: BARPLOT
    ax_top = fig.add_subplot(gs[0, 0])

    bar_arguments = {
        "data":top, "x":"gene_symbol", "y":'weighted_score',
        "edgecolor":"black", "dodge":False, "ax":ax_top, "alpha":0.8, "palette":palette}
    
    sns.barplot(**bar_arguments)

    # sm = plt.cm.ScalarMappable(cmap=cm.YlGnBu, norm=norm)
    # sm.set_array([])
    # plt.colorbar(sm).set_label('Sequence Similarity (%)')

    ax_top.set_ylabel("Weighted Score", fontsize=14, fontweight='bold')
    ax_top.set_xlabel("")
    ax_top.tick_params(axis="x", labelbottom=False)  # hide x labels on top (shown below)
    ax_top.set_title('Weighted Score', fontsize=16, fontweight='bold')

    for p in ax_top.patches: # add numeric value on bar
        h = p.get_height()
        if np.isfinite(h):
            ax_top.text(p.get_x() + p.get_width()/2, h,
                        f"{int(h)}", ha="center", va="bottom", fontsize=7)
            
    
    # SECOND SUBPLOT: DISEASE (striped, per-disease colors; aligned to gene_order)
    ax_bot = fig.add_subplot(gs[1, 0], sharex=ax_top)

    # Build binary membership matrix (rows=diseases, cols=genes), already aligned to gene_order
    M = mat.values.astype(bool)          # shape: (n_dis, n_genes)
    n_dis, n_genes = M.shape

    # Start with a white RGB image (no grey background)
    img = np.ones((n_dis, n_genes, 3), dtype=float)  # values in [0,1]

    # One distinct color per disease (row)
    row_palette = sns.color_palette("Set2", n_dis)

    # Color each disease row where membership == 1
    for i, color in enumerate(row_palette):
        img[i, M[i, :], :] = color  # fill cells (i, j) with that disease’s color

    # Draw the stripes image
    ax_bot.imshow(img, interpolation='nearest', origin='upper', aspect='auto') # aspect=0.7

    # --- tidy labels ---
    def _clean_disease_label(s):
        s = str(s)
        s = re.sub(r'\s*\([^)]*\)', '', s)  # remove parenthetical
        return s.strip()

    clean_diseases = [_clean_disease_label(d) for d in mat.index.tolist()]

    # Y: diseases on the right, one tick per row
    ax_bot.set_yticks(np.arange(n_dis))
    ax_bot.set_yticklabels(clean_diseases, fontsize=10)
    # ax_bot.yaxis.tick_right()

    # X: genes (same order as the top plot)
    ax_bot.set_xticks(np.arange(n_genes))
    ax_bot.set_xticklabels(gene_order, rotation=90, fontsize=8)

    ax_bot.set_xlabel("Gene")
    # ax_bot.set_ylabel("Disease")

    # Clean frame
    for sp in ax_bot.spines.values():
        sp.set_visible(False)

    sm = plt.cm.ScalarMappable(cmap=cm.YlGnBu, norm=norm)
    sm.set_array([])

    # colorbar that adjusts the layout of BOTH subplots equally
    cbar = fig.colorbar(sm, ax=[ax_top, ax_bot], fraction=0.03, pad=0.02)
    cbar.set_label('Sequence Similarity (%)')

    plt.tight_layout()

    save_path = f"{output_directory}/weighted_score.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.tight_layout()
    plt.close()



# ------------------------------------------------------------------------------
# SCATTER_SIMILARITY_IDENTITY_DISEASE: similarity versus identity - disease cue
# ------------------------------------------------------------------------------
def scatter_similarity_identitiy_disease(df, output_directory):

    plt.figure(figsize=(10,8))

    scatter_arguments = {
        "data":df,
        "x":"identity_percent",
        "y":"similarity_percent",
        "s":60,
        'hue': 'disease',
        "edgecolor":"grey"}
    
    # if include_hue:
    #     scatter_arguments["hue"] = "both_best_score"
    
    sns.scatterplot(**scatter_arguments)

    plt.xlabel("Identity (%)", fontsize=12, fontweight='bold')
    plt.ylabel("Similarity (%)", fontsize=12, fontweight='bold')
    plt.title("Fly Orthologues: Identity vs Similarity", fontsize=16, fontweight='bold')
    plt.legend(title="disease")
    plt.xlim(0,90)
    plt.ylim(0,90)
    # plt.axvline(x=30, color='grey', linestyle='--', linewidth=1)  # vertical at identity = 40
    # plt.axhline(y=50, color='grey', linestyle='--', linewidth=1)  # horizontal at similarity = 50

    # --- small static text labels next to each point ---
    for _, row in df.iterrows():
        plt.text(
            row["identity_percent"] + 0.4,  # small x-offset so text doesn't overlap the dot
            row["similarity_percent"],
            row["gene_symbol"],
            fontsize=7,
            alpha=0.7)

    plt.tight_layout()
    plt.legend()

    # --- save static version before interactive part ---
    save_path = f"{output_directory}/scatter_similarity_identity_disease.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.tight_layout()
    plt.close()


# ------------------------------------------------------------------------------
# DISEASE_ONTOLOGY_HEATMAP: gene ontology heatmap 
# ------------------------------------------------------------------------------
def disease_ontology_heatmap(df, output_directory):

    tmp = df.copy()
    # Ensure lists are actual Python lists
    tmp["all_diseases"] = tmp["all_diseases"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith("[") else x
    )
    tmp["GO_Slim_BP_Most_Frequent"] = tmp["GO_Slim_BP_Most_Frequent"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith("[") else x
    )

    # Expand both axes
    tmp = tmp.explode("all_diseases").explode("GO_Slim_BP_Most_Frequent").dropna(subset=["all_diseases", "GO_Slim_BP_Most_Frequent"])

    tmp["GO_BP"] = tmp["GO_Slim_BP_Most_Frequent"]

    print(tmp['GO_BP'].unique())



    mat = tmp.pivot_table(
    index="GO_BP",
    columns="all_diseases",
    values="gene_symbol",
    aggfunc="nunique",
    fill_value=0
)

    plt.figure(figsize=(12, 12))
    sns.heatmap(mat, cmap='GnBu' ,cbar=True)

    plt.xlabel("")
    plt.ylabel("")
    plt.title('Disease Gene Ontology', fontweight='bold', fontsize=16)

    plt.tight_layout()
    outfile = os.path.join(output_directory, 'disease_ontology.png')
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()



# ------------------------------------------------------------------------------
# GENES_ONTOLOGY_HEATMAP: gene ontology heatmap 
# ------------------------------------------------------------------------------
def genes_ontology_heatmap(df, output_directory):

    all_genes = sorted(df["gene_symbol"].unique())   # all 65
    tmp = df.copy()
    tmp["GO_BP"] = tmp["GO_Slim_BP_Most_Frequent"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else (x or []))
    tmp = tmp.explode("GO_BP").dropna(subset=["GO_BP"])

    mat = (
        tmp.assign(flag=1)
          .pivot_table(index="GO_BP", columns="gene_symbol", values="flag",
                       aggfunc="max", fill_value=0)
          .reindex(columns=all_genes, fill_value=0)   # <-- ensures all genes appear
    )

    plt.figure(figsize=(12, 12))
    sns.heatmap(mat, cmap="YlGnBu", cbar=False)
    plt.xlabel("")
    plt.ylabel("")
    plt.title('Gene Ontology', fontweight='bold', fontsize=16)
    plt.tight_layout()
    outfile = os.path.join(output_directory, 'gene_ontology_heatmap.png')
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()  


# df = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/PhD/NDD/GENES/target_genes.csv')
# output_directory = '/Volumes/lab-windingm/home/users/cochral/PhD/NDD/GENES/plots'




def weighted_ortholog_score_diff(df, output_directory):

    top = (df.sort_values('weighted_score', ascending=False)
        .drop_duplicates(subset="gene_symbol", keep="first") # genes appear multiple times - remove
        .loc[:, ["gene_symbol", 'weighted_score', "similarity_percent"]] # keep these columns
        .dropna(subset=['weighted_score'])) # drop rows with missing y values
    
    gene_order = top.sort_values('weighted_score', ascending=False)["gene_symbol"].tolist()
    print(gene_order)
    top = top.set_index("gene_symbol").loc[gene_order].reset_index() # access data using gene if gene is index : )


    # Convert stringified lists to actual lists if needed
    df["all_diseases"] = df["all_diseases"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith("[") else x) 

    expanded = df.explode("all_diseases").dropna(subset=["all_diseases"]) 
    expanded = expanded.drop_duplicates(subset=["gene_symbol", "all_diseases"])



    # DISEASE X GENE MEMBERSHIP MATRIX
    mat = (expanded.assign(flag=1)
            .pivot_table(index="all_diseases", columns="gene_symbol", values="flag", aggfunc="max", fill_value=0).reindex(columns=gene_order))
    
    print(mat)

    
    # FIGURE SIZE / LAYOUT 
    diseases = mat.index.tolist()
    n_genes  = len(gene_order)
    n_dis    = len(diseases)

    fig_w = max(10, 0.4 * n_genes)
    fig_h = 6 + min(2.5, 0.22 * n_dis)

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs  = GridSpec(nrows=2, ncols=1, height_ratios=[3, 0.8], hspace=0.02) # 2 subplot- above and below
    
    # FIRST SUBPLOT: BARPLOT
    ax_top = fig.add_subplot(gs[0, 0])

    bar_arguments = {
        "data":top, "x":"gene_symbol", "y":'weighted_score',
        "edgecolor":"black", "dodge":False, "ax":ax_top, "alpha":0.8, "color":'steelblue', 'linewidth':1.2, 'edgecolor':'black'}

    sns.barplot(**bar_arguments)

    # sm = plt.cm.ScalarMappable(cmap=cm.YlGnBu, norm=norm)
    # sm.set_array([])
    # plt.colorbar(sm).set_label('Sequence Similarity (%)')

    ax_top.set_ylabel("Weighted Score", fontsize=14, fontweight='bold')
    ax_top.set_xlabel("")
    ax_top.tick_params(axis="x", labelbottom=False)  # hide x labels on top (shown below)
    ax_top.set_title('Ortholog Score x Gene Panel', fontsize=16, fontweight='bold')

    for p in ax_top.patches: # add numeric value on bar
        h = p.get_height()
        if np.isfinite(h):
            ax_top.text(p.get_x() + p.get_width()/2, h,
                        f"{int(h)}", ha="center", va="bottom", fontsize=7)
            
    
    # SECOND SUBPLOT: DISEASE (striped, per-disease colors; aligned to gene_order)
    ax_bot = fig.add_subplot(gs[1, 0], sharex=ax_top)

    # Build binary membership matrix (rows=diseases, cols=genes), already aligned to gene_order
    M = mat.values.astype(bool)          # shape: (n_dis, n_genes)
    n_dis, n_genes = M.shape

    # Start with a white RGB image (no grey background)
    img = np.ones((n_dis, n_genes, 3), dtype=float)  # values in [0,1]

    # One distinct color per disease (row)
    row_palette = sns.color_palette("Set2", n_dis)

    # Color each disease row where membership == 1
    for i, color in enumerate(row_palette):
        img[i, M[i, :], :] = color  # fill cells (i, j) with that disease’s color

    # Draw the stripes image
    ax_bot.imshow(img, interpolation='nearest', origin='upper', aspect='auto') # aspect=0.7

    # --- tidy labels ---
    def _clean_disease_label(s):
        s = str(s)
        s = re.sub(r'\s*\([^)]*\)', '', s)  # remove parenthetical
        return s.strip()

    clean_diseases = [_clean_disease_label(d) for d in mat.index.tolist()]

    # Y: diseases on the right, one tick per row
    ax_bot.set_yticks(np.arange(n_dis))
    ax_bot.set_yticklabels(clean_diseases, fontsize=10)
    # ax_bot.yaxis.tick_right()

    # X: genes (same order as the top plot)
    ax_bot.set_xticks(np.arange(n_genes))
    ax_bot.set_xticklabels(gene_order, rotation=90, fontsize=8)

    ax_bot.set_xlabel("Gene")
    # ax_bot.set_ylabel("Disease")

    # Clean frame
    for sp in ax_bot.spines.values():
        sp.set_visible(False)

    plt.tight_layout()

    save_path = f"{output_directory}/weighted_score_check.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.tight_layout()
    plt.close()



# ------------------------------------------------------------------------------
# GENES_ONTOLOGY_HEATMAP: gene ontology heatmap 
# ------------------------------------------------------------------------------
def genes_ontology_heatmap(df, output_directory):

    all_genes = sorted(df["gene_symbol"].unique())   # all 65
    tmp = df.copy()
    tmp["GO_BP"] = tmp["GO_Slim_BP_Most_Frequent"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else (x or []))
    tmp = tmp.explode("GO_BP").dropna(subset=["GO_BP"])

    mat = (
        tmp.assign(flag=1)
          .pivot_table(index="GO_BP", columns="gene_symbol", values="flag",
                       aggfunc="max", fill_value=0)
          .reindex(columns=all_genes, fill_value=0)   # <-- ensures all genes appear
    )

    plt.figure(figsize=(6, 12))
    sns.heatmap(mat, cmap="Greens", cbar=False)
    plt.xlabel("")
    plt.ylabel("")
    plt.title('Gene Ontology', fontweight='bold', fontsize=16)
    plt.tight_layout()
    outfile = os.path.join(output_directory, 'gene_ontology_heatmap.png')
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()



def genes_ontology_heatmap(df, output_directory):


    # GO term -> snappy category
    go_to_cat = {
        # DNA & Chromosomes
        "DNA Integrity": "DNA & Chromosomes",
        "chromatin organization": "DNA & Chromosomes",
        "chromosome segregation": "DNA & Chromosomes",

        # RNA & Gene Control
        "RNA Regulation": "RNA & Gene Control",
        "regulation of DNA-templated transcription": "RNA & Gene Control",

        # Protein Factory
        "Ribosome Biogenesis": "Protein Factory",
        "cytoplasmic translation": "Protein Factory",
        "protein-containing complex assembly": "Protein Factory",

        # Protein Cleanup & Processing
        "Protein Housekeeping": "Protein Cleanup",
        "protein catabolic process": "Protein Cleanup",
        "autophagy": "Protein Cleanup",
        "lysosome organization": "Protein Cleanup",

        # Cell Architecture
        "Cytoskeleton Organization": "Cell Architecture",
        "membrane organization": "Cell Architecture",
        "cell junction organization": "Cell Architecture",
        "establishment or maintenance of cell polarity": "Cell Architecture",

        # Cell Stickiness / ECM
        "Cell Adhesion": "Cell Stickiness / ECM",
        "extracellular matrix organization": "Cell Stickiness / ECM",

        # Trafficking & Transport
        "Intracellular Trafficking": "Trafficking & Transport",
        "vesicle-mediated transport": "Trafficking & Transport",
        "transmembrane transport": "Trafficking & Transport",

        # Signals & Communication
        "Signaling": "Signaling",

        # Metabolism & Energy
        "Metabolic Process": "Metabolism & Energy",
        "generation of precursor metabolites and energy": "Metabolism & Energy",

        # Development & Differentiation
        "Cell Development": "Development",
        "anatomical structure development": "Development",
        "nervous system process": "Development",
        "reproductive process": "Development",

        # Immunity & Defense
        "Immunity & Defense": "Immunity & Defense",
        "defense response to other organism": "Immunity & Defense",

        # Cell Death
        "Cell Death": "Cell Death",
    }





    # order categories nicely in the heatmap
    cat_order = [
        "DNA & Chromosomes",
        "RNA Regulation",
        "Protein Factory",
        "Protein Housekeeping",
        "Cell Architecture",
        "Cell Adhesion",
        "Trafficking & Transport",
        "Signaling",
        "Metabolic Process",
        "Cell Development",
        "Immunity & Defense",
        "Cell Death",
    ]

    # go_to_cat = {}

#     go_to_cat = {
#     # DNA & Chromosomes -> chromatin organization
#     "DNA Integrity": "chromatin organization",
#     "chromatin organization": "chromatin organization",
#     "chromosome segregation": "chromatin organization",

#     # RNA & Gene Control -> regulation of DNA-templated transcription
#     "RNA Regulation": "regulation of DNA-templated transcription",
#     "regulation of DNA-templated transcription": "regulation of DNA-templated transcription",

#     # Protein Factory -> cytoplasmic translation
#     "Ribosome Biogenesis": "cytoplasmic translation",
#     "cytoplasmic translation": "cytoplasmic translation",
#     "protein-containing complex assembly": "cytoplasmic translation",

#     # Protein Cleanup -> autophagy
#     "Protein Housekeeping": "autophagy",
#     "protein catabolic process": "autophagy",
#     "autophagy": "autophagy",
#     "lysosome organization": "autophagy",

#     # Cell Architecture -> cell junction organization
#     "Cytoskeleton Organization": "cell junction organization",
#     "membrane organization": "cell junction organization",
#     "cell junction organization": "cell junction organization",
#     "establishment or maintenance of cell polarity": "cell junction organization",

#     # Cell Stickiness / ECM -> cell adhesion
#     "Cell Adhesion": "cell adhesion",
#     "extracellular matrix organization": "cell adhesion",

#     # Trafficking & Transport -> transmembrane transport
#     "Intracellular Trafficking": "transmembrane transport",
#     "vesicle-mediated transport": "transmembrane transport",
#     "transmembrane transport": "transmembrane transport",

#     # Signals & Communication -> (not in your cat_order list, so no anchor available)
#     "Signaling": 'signaling',

#     # Metabolism & Energy -> (not in your cat_order list, so no anchor available)
#     "Metabolic Process": 'protein catabolic process',
#     "generation of precursor metabolites and energy": 'protein catabolic process',

#     # Development -> anatomical structure development
#     "Cell Development": "anatomical structure development",
#     "anatomical structure development": "anatomical structure development",
#     "nervous system process": "anatomical structure development",
#     "reproductive process": "anatomical structure development",

#     # Immunity & Defense -> immune system process
#     "Immunity & Defense": "immune system process",
#     "defense response to other organism": "immune system process",

#     # Cell Death -> programmed cell death
#     "Cell Death": "programmed cell death",
# }




#     cat_order = [
#     "anatomical structure development",
#     'autophagy',
#     'cell adhesion',
#     'cell junction organization',
#     'chromatin organization',
#     'chromosome segregation',
#     'cytoplasmic translation',
#     'cytoskeleton organization',
#     'immune system process',
#     'nervous system process',
#     'programmed cell death',
#     'protein catabolic process',
#     'protein maturation',
#     'protein-containing complex assembly',
#     'regulation of DNA-templated transcription',
#     'signaling',
#     'transmembrane transport',
#     'vesicle-mediated transport',
#     ]


    all_genes = sorted(df["gene_symbol"].unique())   # all genes

    tmp = df.copy()

    # GO_Slim_BP is a list (or stringified list) → normalize to Python list
    tmp["GO_BP"] = tmp["GO_Slim_BP"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else (x or [])
    )


    # explode so each GO term in the list becomes its own row
    tmp = tmp.explode("GO_BP").dropna(subset=["GO_BP"])

    tmp["category"] = tmp["GO_BP"].map(go_to_cat)

    # build matrix: rows = GO terms, columns = genes, 1 if gene has that GO term
    mat = (
        tmp.assign(flag=1)
           .pivot_table(index="GO_BP", #category
                        columns="gene_symbol",
                        values="flag",
                        aggfunc="max",
                        fill_value=0)
           .reindex(columns=all_genes, fill_value=0)  # ensure all genes present
           .sort_index()                               # optional: sort GO terms
        #    .reindex(index=cat_order, fill_value=0)        # <<< USE cat_order HERE
      
    )

    plt.figure(figsize=(14, 8))
    # sns.heatmap(mat, cmap="GnBu", cbar=False)
    sns.heatmap(mat, cmap="Blues", cbar=False)
    plt.xlabel("")
    plt.ylabel("")
    plt.xticks(rotation=45, ha='right')
    plt.title('Gene Ontology', fontweight='bold', fontsize=16)
    plt.tight_layout()

    outfile = os.path.join(output_directory, 'gene_ontology_heatmap.png')
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()






def weighted_ortholog_disease_matrix(df, output_directory):

    # --- get gene order (still based on weighted_score) ---
    top = (
        df.sort_values('weighted_score', ascending=False)
          .drop_duplicates(subset="gene_symbol", keep="first")
          .loc[:, ["gene_symbol", "weighted_score"]]
          .dropna(subset=["weighted_score"])
    )

    gene_order = top["gene_symbol"].tolist()

    # --- parse disease lists ---
    df["all_diseases"] = df["all_diseases"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith("[") else x
    )

    expanded = (
        df.explode("all_diseases")
          .dropna(subset=["all_diseases"])
          .drop_duplicates(subset=["gene_symbol", "all_diseases"])
    )

    # --- disease × gene membership matrix ---
    mat = (
        expanded.assign(flag=1)
        .pivot_table(
            index="all_diseases",
            columns="gene_symbol",
            values="flag",
            aggfunc="max",
            fill_value=0
        )
        .reindex(columns=gene_order)
    )

    diseases = mat.index.tolist()
    n_dis, n_genes = mat.shape

    # --- figure size ---
    fig_w = max(10, 0.4 * n_genes)
    fig_h = max(3, 0.22 * n_dis)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # --- build stripe image ---
    M = mat.values.astype(bool)
    img = np.ones((n_dis, n_genes, 3), dtype=float)

    row_palette = sns.color_palette("Set2", n_dis)

    for i, color in enumerate(row_palette):
        img[i, M[i, :], :] = color

    ax.imshow(img, interpolation="nearest", origin="upper", aspect="auto")

    # --- clean disease labels ---
    def _clean_disease_label(s):
        s = str(s)
        s = re.sub(r'\s*\([^)]*\)', '', s)
        return s.strip()

    clean_diseases = [_clean_disease_label(d) for d in diseases]

    # --- ticks & labels ---
    ax.set_yticks(np.arange(n_dis))
    ax.set_yticklabels(clean_diseases, fontsize=10)

    ax.set_xticks(np.arange(n_genes))
    ax.set_xticklabels(gene_order, rotation=45, fontsize=8)

    ax.set_xlabel("Gene")

    # --- remove frame ---
    for sp in ax.spines.values():
        sp.set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{output_directory}/disease_gene_matrix.png",
                dpi=300, bbox_inches="tight")
    plt.savefig(f"{output_directory}/disease_gene_matrix.pdf",
                format='pdf', bbox_inches="tight")
    
    plt.close()







df = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/PhD/NDD/GENES/target_genes.csv')
output_directory = '/Volumes/lab-windingm/home/users/cochral/PhD/NDD/GENES/plots'
weighted_ortholog_disease_matrix(df, output_directory)

# genes_ontology_heatmap(df, output_directory)
# weighted_ortholog_score_diff(df, output_directory)
# scatter_similarity_identitiy_disease(df, output_directory)
genes_ontology_heatmap(df, output_directory)
disease_ontology_heatmap(df, output_directory)