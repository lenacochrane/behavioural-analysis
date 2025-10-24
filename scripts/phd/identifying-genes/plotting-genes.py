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
# MODIFY_DF: cleans df prior to plotting
# ------------------------------------------------------------------------------
def modify_df(df):
    df["both_best_score"] = ((df["best_score"].str.lower() == "yes") &(df["best_score_reverse"].str.lower() == "yes"))
    cols = ['Rank', "total", "disease model", "mouse", "human", "cell line", "drosophila", 'identity_percent', 'similarity_percent', 'weighted_score']
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

# ------------------------------------------------------------------------------
# FILTER: filter for boolean column
# ------------------------------------------------------------------------------
def filter(df, column_name):
    df = df[df[column_name]]
    return df

# ------------------------------------------------------------------------------
# WEIGHTED_ORTHOLOG_SCORE: weighted score given to orthologs
# ------------------------------------------------------------------------------
def weighted_ortholog_score(df, output_directory):

    top = (df.sort_values('weighted_score', ascending=False)
        .drop_duplicates(subset="Gene", keep="first") # genes appear multiple times - remove
        .loc[:, ["Gene", 'weighted_score', "similarity_percent"]] # keep these columns
        .dropna(subset=['weighted_score'])) # drop rows with missing y values
    
    gene_order = top.sort_values('weighted_score', ascending=False)["Gene"].tolist()
    top = top.set_index("Gene").loc[gene_order].reset_index() # access data using gene if gene is index : )

    norm = plt.Normalize(top['similarity_percent'].min(), top['similarity_percent'].max())
    palette = [cm.YlGnBu(norm(v)) for v in top['similarity_percent'].values]

    # DISEASE X GENE MEMBERSHIP MATRIX
    mat = (df.assign(flag=1)
            .pivot_table(index="Disease", columns="Gene", values="flag", aggfunc="max", fill_value=0).reindex(columns=gene_order))

    
    # FIGURE SIZE / LAYOUT 
    diseases = mat.index.tolist()
    n_genes  = len(gene_order)
    n_dis    = len(diseases)

    fig_w = max(14, 0.4 * n_genes)
    fig_h = 6 + min(2.5, 0.22 * n_dis)

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs  = GridSpec(nrows=2, ncols=1, height_ratios=[3, 0.8], hspace=0.02) # 2 subplot- above and below
    
    # FIRST SUBPLOT: BARPLOT
    ax_top = fig.add_subplot(gs[0, 0])

    bar_arguments = {
        "data":top, "x":"Gene", "y":'weighted_score',
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
    ax_bot.imshow(img, aspect=0.7, interpolation='nearest', origin='upper')

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
# SCATTER_SIMILARITY_IDENTITY: scatter plot of similarity versus identity
# ------------------------------------------------------------------------------
def scatter_similarity_identitiy(df, output_directory, include_hue=True):

    plt.figure(figsize=(10,8))

    scatter_args = {
        "data": df,
        "x": "identity_percent",
        "y": "similarity_percent",
        "s": 60,
        "edgecolor": "grey"}
    
    if include_hue:
        scatter_args["hue"] = "both_best_score"

    sns.scatterplot(**scatter_args)
    plt.xlabel("Identity (%)", fontsize=12, fontweight='bold')
    plt.ylabel("Similarity (%)", fontsize=12, fontweight='bold')
    plt.title("Fly Orthologues: Identity vs Similarity", fontsize=16, fontweight='bold')
    plt.legend(title="Both Best Score")
    plt.xlim(0,90)
    plt.ylim(0,90)
    # add reference threshold lines
    # plt.axvline(x=30, color='grey', linestyle='--', linewidth=1)  # vertical at identity = 40
    # plt.axhline(y=50, color='grey', linestyle='--', linewidth=1)  # horizontal at similarity = 50


    # --- small static text labels next to each point ---
    for _, row in df.iterrows():
        plt.text(
            row["identity_percent"] + 0.4,  # small x-offset so text doesn't overlap the dot
            row["similarity_percent"],
            row["Gene"],
            fontsize=7,
            alpha=0.7)

    plt.tight_layout()
    plt.legend()

    # --- save static version before interactive part ---
    save_path = f"{output_directory}/scatter_similarity_identity.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")

    cursor = mplcursors.cursor(hover=True)
    @cursor.connect("add")
    def on_add(sel):
        i = sel.index
        gene = df.iloc[i]["Gene"]
        sel.annotation.set_text(gene)
        sel.annotation.get_bbox_patch().set(alpha=0.9, color="white")

    plt.tight_layout()
    plt.show()

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
        'hue': 'Disease',
        "edgecolor":"grey"}
    
    # if include_hue:
    #     scatter_arguments["hue"] = "both_best_score"
    
    sns.scatterplot(**scatter_arguments)

    plt.xlabel("Identity (%)", fontsize=12, fontweight='bold')
    plt.ylabel("Similarity (%)", fontsize=12, fontweight='bold')
    plt.title("Fly Orthologues: Identity vs Similarity", fontsize=16, fontweight='bold')
    plt.legend(title="Both Best Score")
    plt.xlim(0,90)
    plt.ylim(0,90)
    # plt.axvline(x=30, color='grey', linestyle='--', linewidth=1)  # vertical at identity = 40
    # plt.axhline(y=50, color='grey', linestyle='--', linewidth=1)  # horizontal at similarity = 50

    # --- small static text labels next to each point ---
    for _, row in df.iterrows():
        plt.text(
            row["identity_percent"] + 0.4,  # small x-offset so text doesn't overlap the dot
            row["similarity_percent"],
            row["Gene"],
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
# SCATTER_SIMILARITY_IDENTITY_WEIGHTEDSCORE: similarity versus identity 
# ------------------------------------------------------------------------------
def scatter_similarity_identitiy_weightedscore(df, output_directory):

    plt.figure(figsize=(10,8))

    scatter_arguments = {
        "data":df,
        "x":"identity_percent",
        "y":"similarity_percent",
        "s":60,
        'hue':"weighted_score",   
        "palette":"viridis",              
        "edgecolor":"grey",  "legend": False}
    
    # if include_hue:
    #     scatter_arguments["hue"] = "both_best_score"
    
    scatter = sns.scatterplot(**scatter_arguments)

    norm = plt.Normalize(df["weighted_score"].min(), df["weighted_score"].max())
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=scatter, fraction=0.03, pad=0.02)
    cbar.set_label("Weighted Score", fontsize=10)

    plt.xlabel("Identity (%)", fontsize=12, fontweight='bold')
    plt.ylabel("Similarity (%)", fontsize=12, fontweight='bold')
    plt.title("Fly Orthologues: Identity vs Similarity", fontsize=16, fontweight='bold')
    plt.xlim(0,90)
    plt.ylim(0,90)
    # plt.axvline(x=30, color='grey', linestyle='--', linewidth=1)  # vertical at identity = 40
    # plt.axhline(y=50, color='grey', linestyle='--', linewidth=1)  # horizontal at similarity = 50

    # --- small static text labels next to each point ---
    for _, row in df.iterrows():
        plt.text(
            row["identity_percent"] + 0.4,  # small x-offset so text doesn't overlap the dot
            row["similarity_percent"],
            row["Gene"],
            fontsize=7,
            alpha=0.7)

    plt.tight_layout()

    # --- save static version before interactive part ---
    save_path = f"{output_directory}/scatter_similarity_identity_weightedscore.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

# ------------------------------------------------------------------------------
# SCATTER_SIMILARITY_IDENTITY_PER_DISEASE: similarity versus identity
# ------------------------------------------------------------------------------
def scatter_similarity_identitiy_per_disease(df, output_directory, include_hue=True):

    for disease in df["Disease"].dropna().unique():
        sub = df[df["Disease"] == disease]

        plt.figure(figsize=(10,8))
        scatter_arguments = {
            "data":sub,
            "x":"identity_percent",
            "y":"similarity_percent",
            "s":60,
            "edgecolor":"grey"}
        if include_hue:
            scatter_arguments["hue"] = "both_best_score"

        sns.scatterplot(**scatter_arguments)
        plt.xlabel("Identity (%)", fontsize=12, fontweight='bold')
        plt.ylabel("Similarity (%)", fontsize=12, fontweight='bold')
        plt.title(f"Fly Orthologues: Identity vs Similarity — {disease}", fontsize=16, fontweight='bold')
        plt.xlim(0,90)
        plt.ylim(0,90)


        # small gene labels
        for _, row in sub.iterrows():
            plt.text(
                row["identity_percent"] + 0.4,
                row["similarity_percent"],
                row["Gene"],
                fontsize=7, alpha=0.7
            )

        plt.tight_layout()

        def make_safe_name(disease):
            # replace spaces and slashes first
            safe_name = str(disease).replace("/", "_").replace(" ", "_")
            # check if abbreviation in parentheses exists, e.g. (ASD)
            match = re.search(r'\(([^)]+)\)', safe_name)
            if match:
                return match.group(1)  # use only the abbreviation inside ()
            else:
                return safe_name
            
        safe_name = make_safe_name(disease)
        # safe_name = str(disease).replace("/", "_").replace(" ", "_")
        outfile = os.path.join(output_directory, f"scatter_{safe_name}.png")
        plt.savefig(outfile, dpi=300, bbox_inches="tight")
        plt.close()

# ------------------------------------------------------------------------------
# BARPLOT_PUBMED: barplot for pubmed counts
# ------------------------------------------------------------------------------
def barplot_pubmed(df, output_directory, include_hue=True):

    y_cols = ["total", "disease model", "mouse", "human", "cell line", "drosophila"]

    # --- loop per disease ---
    for disease, dsub in df.groupby("Disease", dropna=True):
        dsub = dsub.dropna(subset=["Rank"]).copy()
        if dsub.empty:
            continue

        # order genes by Rank within this disease
        gene_order = dsub.sort_values("Rank")["Gene"].tolist()

        # --- loop per y column ---
        for y in y_cols:
            if y not in dsub.columns:
                continue
            dplot = dsub.dropna(subset=[y]).copy()
            if dplot.empty:
                continue

            plt.figure(figsize=(14, 6))

            bar_argument = {
                "data":dplot,
                "x":'Gene',
                "y":y,
                "order":gene_order,
                "edgecolor":'black'}
            
            if include_hue:
                bar_argument['hue'] = "both_best_score"

            ax = sns.barplot(**bar_argument)

            ax.set_xlabel('Gene (ordered by Rank)', fontsize=12, fontweight='bold')
            ax.set_ylabel(f'PubMed count for {y}', fontsize=12, fontweight='bold')
            ax.tick_params(axis='x', rotation=60)
            ax.set_title(f'{disease} — {y}', fontsize=16, fontweight='bold')
            ax.legend(title="Both best score")

            for p in ax.patches:
                h = p.get_height()
                if pd.notnull(h):
                    ax.text(
                        p.get_x() + p.get_width()/2, h,
                        f"{int(h)}",
                        ha='center', va='bottom', fontsize=8
                    )

            plt.tight_layout()

            def make_safe_name(disease):
                # replace spaces and slashes first
                safe_name = str(disease).replace("/", "_").replace(" ", "_")
                # check if abbreviation in parentheses exists, e.g. (ASD)
                match = re.search(r'\(([^)]+)\)', safe_name)
                if match:
                    return match.group(1)  # use only the abbreviation inside ()
                else:
                    return safe_name

            safe_disease = make_safe_name(disease)
            safe_y = y.replace(" ", "_")
            fname = f"{safe_disease}__{safe_y}.png"
            output = os.path.join(output_directory, fname)
            plt.savefig(output, dpi=300, bbox_inches="tight")
            plt.close()

# ------------------------------------------------------------------------------
# GENES_DISEASE: barplot and disease assocation
# ------------------------------------------------------------------------------
def genes_disease(df, output_directory, include_hue=True):

    plotting_y_cols = ["total", "disease model", "mouse", "human", "cell line", "drosophila"]

    for y in plotting_y_cols:
        if y not in df.columns:
            continue

        top = (df.sort_values(y, ascending=False)
            .drop_duplicates(subset="Gene", keep="first") # genes appear multiple times - remove
            .loc[:, ["Gene", y, "both_best_score"]] # keep these columns
            .dropna(subset=[y])) # drop rows with missing y values
        
        gene_order = top.sort_values(y, ascending=False)["Gene"].tolist()
        top = top.set_index("Gene").loc[gene_order].reset_index() # access data using gene if gene is index : )

        # DISEASE X GENE MEMBERSHIP MATRIX
        mat = (df.assign(flag=1)
                .pivot_table(index="Disease", columns="Gene", values="flag", aggfunc="max", fill_value=0).reindex(columns=gene_order))
        
        """Create a 0/1 matrix: row = disease, column = gene. Cell = 1 if that disease has that gene anywhere in the full dataframe."""
        
        # FIGURE SIZE / LAYOUT 
        diseases = mat.index.tolist()
        n_genes  = len(gene_order)
        n_dis    = len(diseases)

        fig_w = max(14, 0.4 * n_genes)
        fig_h = 6 + min(2.5, 0.22 * n_dis)

        fig = plt.figure(figsize=(fig_w, fig_h))
        gs  = GridSpec(nrows=2, ncols=1, height_ratios=[3, 0.8], hspace=0.02) # 2 subplot- above and below
        
        # FIRST SUBPLOT: BARPLOT
        ax_top = fig.add_subplot(gs[0, 0])

        bar_arguments = {
            "data":top, "x":"Gene", "y":y,
            "edgecolor":"black", "dodge":False, "ax":ax_top, "alpha":0.8}
        
        if include_hue:
            bar_arguments['hue'] = 'both_best_score'
            bar_arguments["palette"] = {True: "crimson", False: "steelblue"}
        
        else:
            bar_arguments["color"] = "steelblue"

        sns.barplot(**bar_arguments)

        ax_top.set_ylabel(f"{y.title()} PubMed count", fontsize=14, fontweight='bold')
        ax_top.set_xlabel("")
        ax_top.tick_params(axis="x", labelbottom=False)  # hide x labels on top (shown below)
        ax_top.legend(title="Both best score", loc="upper right")
        ax_top.set_title('Gene x Disease', fontsize=16, fontweight='bold')

        for p in ax_top.patches: # add numeric value on bar
            h = p.get_height()
            if np.isfinite(h):
                ax_top.text(p.get_x() + p.get_width()/2, h,
                            f"{int(h)}", ha="center", va="bottom", fontsize=7)
                
        
        # SECOND SUBPLOT: DISEASE
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
        ax_bot.imshow(img, aspect=0.7, interpolation='nearest', origin='upper')

        # --- tidy labels ---
        def _clean_disease_label(s):
            s = str(s)
            s = re.sub(r'\s*\([^)]*\)', '', s)  # remove parenthetical
            return s.strip()

        clean_diseases = [_clean_disease_label(d) for d in mat.index.tolist()]

        # Y: diseases on the right, one tick per row
        ax_bot.set_yticks(np.arange(n_dis))
        ax_bot.set_yticklabels(clean_diseases, fontsize=10)
        ax_bot.yaxis.tick_right()

        # X: genes (same order as the top plot)
        ax_bot.set_xticks(np.arange(n_genes))
        ax_bot.set_xticklabels(gene_order, rotation=90, fontsize=8)

        ax_bot.set_xlabel("Gene")
        ax_bot.set_ylabel("Disease")

        # Clean frame
        for sp in ax_bot.spines.values():
            sp.set_visible(False)


        plt.tight_layout()
        safe_y = y.replace(" ", "_")
        outfile = os.path.join(output_directory, f'genes_disease_{safe_y}.png')
        plt.savefig(outfile, dpi=300, bbox_inches="tight")
        plt.close()

# ------------------------------------------------------------------------------
# GENES_DISEASE_WEIGHTEDSCORE: barplot and disease assocation
# ------------------------------------------------------------------------------
def genes_disease_weightedscore(df, output_directory):

    plotting_y_cols = ["total", "disease model", "mouse", "human", "cell line", "drosophila"]

    for y in plotting_y_cols:
        if y not in df.columns:
            continue

        top = (df.sort_values(y, ascending=False)
            .drop_duplicates(subset="Gene", keep="first") # genes appear multiple times - remove
            .loc[:, ["Gene", y, "weighted_score"]] # keep these columns
            .dropna(subset=[y])) # drop rows with missing y values
        
        gene_order = top.sort_values(y, ascending=False)["Gene"].tolist()
        top = top.set_index("Gene").loc[gene_order].reset_index() # access data using gene if gene is index : )

        norm = plt.Normalize(top['weighted_score'].min(), top['weighted_score'].max())
        palette = [cm.YlGnBu(norm(v)) for v in top['weighted_score'].values]

        # DISEASE X GENE MEMBERSHIP MATRIX
        mat = (df.assign(flag=1)
                .pivot_table(index="Disease", columns="Gene", values="flag", aggfunc="max", fill_value=0).reindex(columns=gene_order))
        
        """Create a 0/1 matrix: row = disease, column = gene. Cell = 1 if that disease has that gene anywhere in the full dataframe."""
        
        # FIGURE SIZE / LAYOUT 
        diseases = mat.index.tolist()
        n_genes  = len(gene_order)
        n_dis    = len(diseases)

        fig_w = max(14, 0.4 * n_genes)
        fig_h = 6 + min(2.5, 0.22 * n_dis)

        fig = plt.figure(figsize=(fig_w, fig_h))
        gs  = GridSpec(nrows=2, ncols=1, height_ratios=[3, 0.8], hspace=0.02) # 2 subplot- above and below
        
        # FIRST SUBPLOT: BARPLOT
        ax_top = fig.add_subplot(gs[0, 0])

        bar_arguments = {
            "data":top, "x":"Gene", "y":y,
            "edgecolor":"black", "dodge":False, "ax":ax_top, "alpha":0.8, 'palette':palette}
        

        sns.barplot(**bar_arguments)

        ax_top.set_ylabel(f"{y.title()} PubMed count", fontsize=14, fontweight='bold')
        ax_top.set_xlabel("")
        ax_top.tick_params(axis="x", labelbottom=False)  # hide x labels on top (shown below)
        ax_top.set_title('Gene x Disease', fontsize=16, fontweight='bold')

        for p in ax_top.patches: # add numeric value on bar
            h = p.get_height()
            if np.isfinite(h):
                ax_top.text(p.get_x() + p.get_width()/2, h,
                            f"{int(h)}", ha="center", va="bottom", fontsize=7)
                
        
        # SECOND SUBPLOT: DISEASE
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
        ax_bot.imshow(img, aspect=0.7, interpolation='nearest', origin='upper')

        # --- tidy labels ---
        def _clean_disease_label(s):
            s = str(s)
            s = re.sub(r'\s*\([^)]*\)', '', s)  # remove parenthetical
            return s.strip()

        clean_diseases = [_clean_disease_label(d) for d in mat.index.tolist()]

        # Y: diseases on the right, one tick per row
        ax_bot.set_yticks(np.arange(n_dis))
        ax_bot.set_yticklabels(clean_diseases, fontsize=10)
        ax_bot.yaxis.tick_right()

        # X: genes (same order as the top plot)
        ax_bot.set_xticks(np.arange(n_genes))
        ax_bot.set_xticklabels(gene_order, rotation=90, fontsize=8)

        ax_bot.set_xlabel("Gene")
        ax_bot.set_ylabel("Disease")

        # Clean frame
        for sp in ax_bot.spines.values():
            sp.set_visible(False)

        sm = plt.cm.ScalarMappable(cmap=cm.YlGnBu, norm=norm)
        sm.set_array([])

        # get figure dimensions
        fig_w, fig_h = fig.get_size_inches()

        # add a small colorbar axis (half the normal height)
        # [x0, y0, width, height] are *fractions* of the figure area
        cax = fig.add_axes([0.93, 0.35, 0.02, 0.5])  # <-- adjust y0 and height to position
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label('Weighted Score', fontsize=12)
        cbar.ax.tick_params(labelsize=7, width=0.5, length=2)


        plt.tight_layout()
        safe_y = y.replace(" ", "_")
        outfile = os.path.join(output_directory, f'genes_disease_{safe_y}_weightedscore.png')
        plt.savefig(outfile, dpi=300, bbox_inches="tight")
        plt.close()

# ------------------------------------------------------------------------------
# GENES_DISEASE_HEATMAP: gene disease heatmap 
# ------------------------------------------------------------------------------
def genes_disease_heatmap(df, output_directory):

    all_genes = sorted(df["Gene"].unique())   # all 65

    mat = (
        df.assign(flag=1)
          .pivot_table(index="Disease", columns="Gene", values="flag",
                       aggfunc="max", fill_value=0)
          .reindex(columns=all_genes, fill_value=0)   # <-- ensures all genes appear
    )

    def short_disease(name):
        m = re.search(r"\(([^)]+)\)", str(name))
        if m:
            return m.group(1)
        return {
            "Epilepsy": "EPI",
        }.get(name, name)

    mat.index = mat.index.map(short_disease)            

    plt.figure(figsize=(12, 5))
    sns.heatmap(mat, cmap="PuBuGn", cbar=False)
    plt.xlabel("")
    plt.ylabel("")
    plt.title('Gene Disease Distribution', fontweight='bold', fontsize=16)
    plt.tight_layout()
    outfile = os.path.join(output_directory, 'gene_disease_heatmap.png')
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()

# ------------------------------------------------------------------------------
# GENES_ONTOLOGY_HEATMAP: gene ontology heatmap 
# ------------------------------------------------------------------------------
def genes_ontology_heatmap(df, output_directory):

    all_genes = sorted(df["Gene"].unique())   # all 65
    tmp = df.copy()
    tmp["GO_BP"] = tmp["GO_Slim_BP_Most_Frequent"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else (x or []))
    tmp = tmp.explode("GO_BP").dropna(subset=["GO_BP"])

    mat = (
        tmp.assign(flag=1)
          .pivot_table(index="GO_BP", columns="Gene", values="flag",
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

# ------------------------------------------------------------------------------
# DISEASE_ONTOLOGY_HEATMAP: gene ontology heatmap 
# ------------------------------------------------------------------------------
def disease_ontology_heatmap(df, output_directory):

    tmp = df.copy()
    tmp["GO_BP"] = tmp["GO_Slim_BP_Most_Frequent"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else (x or []))
    tmp = tmp.explode("GO_BP").dropna(subset=["GO_BP"])

    mat = tmp.pivot_table(
        index="GO_BP",   # diseases will form the rows (y-axis)
        columns="Disease",   # GO terms will form the columns (x-axis)
        values="Gene",     # we’ll count the number of unique genes per cell
        aggfunc="nunique", # how to combine multiple rows into one number
        fill_value=0       # fill any missing combos with 0
    )

    def short_disease(name):
        # use text in parentheses if present: "(ADHD)" -> "ADHD"
        m = re.search(r"\(([^)]+)\)", str(name))
        if m:
            return m.group(1)
        return {
            "Epilepsy": "EPI",
        }.get(name, name)

    mat = mat.rename(columns={c: short_disease(c) for c in mat.columns})

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
# GENES_ALIAS_RANKED_PUBMED: ranked genes crossmatched
# ------------------------------------------------------------------------------
def genes_alias_ranked_pubmed(df1, df2, output_directory):
    
    gene_order = (
        df1.sort_values('total', ascending=False)
           .drop_duplicates(subset='Gene', keep='first')['Gene']
           .tolist()
    )
    gene_order_alias = (
        df2.sort_values('total', ascending=False)
           .drop_duplicates(subset='Gene', keep='first')['Gene']
           .tolist()
    )

    genes = [g for g in gene_order if g in gene_order_alias] # check contain same genes

    left_pos = {g: i for i, g in enumerate(gene_order)} # y coord
    right_pos = {g: i for i, g in enumerate(gene_order_alias)} # y coord

    plt.figure(figsize=(6, len(genes) * 0.3))
    ax = plt.gca()

    # draw connecting lines
    for g in genes:
        plt.plot(
            [0, 1],
            [left_pos[g], right_pos[g]],
            color="grey", alpha=0.7, linewidth=1
        )

    # label genes on both sides
    for g in genes:
        plt.text(-0.02, left_pos[g], g, ha="right", va="center", fontsize=8)
        plt.text(1.02, right_pos[g], g, ha="left", va="center", fontsize=8)

    # cosmetics
    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(len(genes)-0.5, -0.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Gene Only", "Aliases"], fontsize=10, fontweight='bold')
    ax.set_yticks([])
    ax.set_title('Ranked Order of Total Papers', fontsize=12, fontweight='bold')
    plt.tight_layout()
    outfile = os.path.join(output_directory, 'gene_order.png')
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()


# ------------------------------------------------------------------------------
# GENES_RANKED_PUBMED_ALL: ranked genes crossmatched
# ------------------------------------------------------------------------------
def genes_ranked_pubmed_all(df, output_directory):

    total = (
        df.sort_values('total', ascending=False)
           .drop_duplicates(subset='Gene', keep='first')['Gene']
           .tolist())
    
    disease_model = (
        df.sort_values('disease model', ascending=False)
           .drop_duplicates(subset='Gene', keep='first')['Gene']
           .tolist())
    
    mouse = (
        df.sort_values('mouse', ascending=False)
           .drop_duplicates(subset='Gene', keep='first')['Gene']
           .tolist())
    
    drosophila = (
        df.sort_values('drosophila', ascending=False)
           .drop_duplicates(subset='Gene', keep='first')['Gene']
           .tolist())
    
    gene_lists = [total, disease_model, mouse, drosophila]
    col_labels = ["Total", "Disease Model", "Mouse", "Drosophila"]

    # Assume same set of genes in all lists; use the first list for iteration
    genes = total

    # y-positions per column
    pos_maps = [{g: i for i, g in enumerate(lst)} for lst in gene_lists]

    plt.figure(figsize=(10, len(genes) * 0.25))
    ax = plt.gca()

    # Connecting lines between adjacent columns
    for g in genes:
        for i in range(len(gene_lists) - 1):
            ax.plot([i, i+1],
                    [pos_maps[i][g], pos_maps[i+1][g]],
                    color="grey", alpha=0.7, linewidth=0.8)

    # Labels at every column
    every_n = 1
    for i, pos_map in enumerate(pos_maps):
        xtext = i - 0.05 if i == 0 else (i + 0.05 if i == len(pos_maps)-1 else i)
        ha = "right" if i == 0 else ("left" if i == len(pos_maps)-1 else "center")
        for j, g in enumerate(genes):
            if j % every_n:  # thin labels if desired
                continue
            ax.text(xtext, pos_map[g], g, ha=ha, va="center", fontsize=8)

    # Axes cosmetics
    ax.set_xlim(-0.2, len(gene_lists)-0.8)
    ax.set_ylim(len(genes)-0.5, -0.5)
    ax.set_xticks(range(len(gene_lists)))
    ax.set_xticklabels(col_labels, fontsize=10, fontweight='bold')
    ax.set_yticks([])
    ax.set_title("Gene Rank Comparison Across Categories", fontsize=12, fontweight='bold')

    plt.tight_layout()
    out = os.path.join(output_directory, "gene_rank_comparison_all_labels.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()

# ------------------------------------------------------------------------------
# GENES_RANKED_TOTAL_DROSOPH: ranked genes crossmatched
# ------------------------------------------------------------------------------
def genes_ranked_total_drosoph(df, output_directory):
    
    total = (
        df.sort_values('total', ascending=False)
           .drop_duplicates(subset='Gene', keep='first')['Gene']
           .tolist())
    
    drosophila = (
        df.sort_values('drosophila', ascending=False)
           .drop_duplicates(subset='Gene', keep='first')['Gene']
           .tolist())

    genes = [g for g in total if g in drosophila] # check contain same genes

    left_pos = {g: i for i, g in enumerate(total)} # y coord
    right_pos = {g: i for i, g in enumerate(drosophila)} # y coord

    plt.figure(figsize=(6, len(genes) * 0.3))
    ax = plt.gca()

    # draw connecting lines
    for g in genes:
        plt.plot(
            [0, 1],
            [left_pos[g], right_pos[g]],
            color="grey", alpha=0.7, linewidth=1
        )

    # label genes on both sides
    for g in genes:
        plt.text(-0.02, left_pos[g], g, ha="right", va="center", fontsize=8)
        plt.text(1.02, right_pos[g], g, ha="left", va="center", fontsize=8)

    # cosmetics
    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(len(genes)-0.5, -0.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Total", "Drosophila"], fontsize=10, fontweight='bold')
    ax.set_yticks([])
    ax.set_title('Ranked Order of Papers', fontsize=12, fontweight='bold')
    plt.tight_layout()
    outfile = os.path.join(output_directory, 'gene_order_total_drosoph.png')
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()

# ------------------------------------------------------------------------------
# GENES_RANKED_TOTAL_mouse: ranked genes crossmatched
# ------------------------------------------------------------------------------
def genes_ranked_total_mouse(df, output_directory):
    
    total = (
        df.sort_values('total', ascending=False)
           .drop_duplicates(subset='Gene', keep='first')['Gene']
           .tolist())
    
    mouse = (
        df.sort_values('mouse', ascending=False)
           .drop_duplicates(subset='Gene', keep='first')['Gene']
           .tolist())

    genes = [g for g in total if g in mouse] # check contain same genes

    left_pos = {g: i for i, g in enumerate(total)} # y coord
    right_pos = {g: i for i, g in enumerate(mouse)} # y coord

    plt.figure(figsize=(6, len(genes) * 0.3))
    ax = plt.gca()

    # draw connecting lines
    for g in genes:
        plt.plot(
            [0, 1],
            [left_pos[g], right_pos[g]],
            color="grey", alpha=0.7, linewidth=1
        )

    # label genes on both sides
    for g in genes:
        plt.text(-0.02, left_pos[g], g, ha="right", va="center", fontsize=8)
        plt.text(1.02, right_pos[g], g, ha="left", va="center", fontsize=8)

    # cosmetics
    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(len(genes)-0.5, -0.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Total", "Mouse"], fontsize=10, fontweight='bold')
    ax.set_yticks([])
    ax.set_title('Ranked Order of Papers', fontsize=12, fontweight='bold')
    plt.tight_layout()
    outfile = os.path.join(output_directory, 'gene_order_total_mouse.png')
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()

# ------------------------------------------------------------------------------
# GENES_SCORING_RANKED: ranking genes
# ------------------------------------------------------------------------------
def genes_scoring_ranked(df, output_directory):

    df_rank = df.drop_duplicates(subset='Gene', keep='first')

    for col in ["total", "disease model", "mouse", "drosophila", 'weighted_score']:
        df_rank[f"rank_{col.replace(' ', '_')}"] = df_rank[col].rank(method="min", ascending=False).astype(int)

    df_rank = df_rank[
        ["Gene", "Disease", "weighted_score", 
         "rank_total", "rank_disease_model", "rank_mouse", "rank_drosophila" ,'rank_weighted_score']]
    
    print(df_rank)
    
    df_rank["overall_rank"] = (
        df_rank[['rank_total', 'rank_disease_model', "rank_mouse",  "rank_drosophila", 'rank_weighted_score']].sum(axis=1))  # "rank_total", 'rank_disease_model
    

    gene_order = df_rank.sort_values('overall_rank', ascending=True)["Gene"].tolist()
    
        # DISEASE x GENE membership (use full df; not only top)
    mat = (
        df.assign(flag=1)
            .pivot_table(index="Disease", columns="Gene", values="flag", aggfunc="max", fill_value=0)
            .reindex(columns=gene_order)  # align columns to ranked gene order
    )

    # ---- Figure sizing / layout
    diseases = mat.index.tolist()
    n_genes  = len(gene_order)
    n_dis    = len(diseases)

    fig_w = max(14, 0.4 * n_genes)
    fig_h = 6 + min(2.5, 0.22 * n_dis)

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs  = GridSpec(nrows=2, ncols=1, height_ratios=[3, 0.8], hspace=0.02)

    # ---- TOP: barplot
    ax_top = fig.add_subplot(gs[0, 0])

    sns.barplot(
        data=df_rank, x="Gene", y='overall_rank',
        edgecolor="black", dodge=False, ax=ax_top, alpha=0.8, order=gene_order
    )

    ax_top.set_ylabel("Rank", fontsize=14, fontweight='bold')
    ax_top.set_xlabel("")
    ax_top.tick_params(axis="x", labelbottom=False)  # hide top x labels
    ax_top.set_title('Overall Rank', fontsize=16, fontweight='bold')

    # numeric labels on bars
    for p in ax_top.patches:
        h = p.get_height()
        if np.isfinite(h):
            ax_top.text(p.get_x() + p.get_width()/2, h,
                        f"{int(h)}", ha="center", va="bottom", fontsize=7)

    # ---- BOTTOM: disease stripes (one color per disease row)
    ax_bot = fig.add_subplot(gs[1, 0], sharex=ax_top)

    M = mat.values.astype(bool)  # (n_dis, n_genes)
    n_dis, n_genes = M.shape

    # Start white image
    img = np.ones((n_dis, n_genes, 3), dtype=float)

    # One distinct color per disease row
    row_palette = sns.color_palette("Set2", max(1, n_dis))

    for i, color in enumerate(row_palette[:n_dis]):
        if M.shape[1] > 0:
            img[i, M[i, :], :] = color

    ax_bot.imshow(img, aspect=0.7, interpolation='nearest', origin='upper')

    # Clean disease labels
    def _clean_disease_label(s):
        s = str(s)
        s = re.sub(r'\s*\([^)]*\)', '', s)  # remove parenthetical
        return s.strip()

    clean_diseases = [_clean_disease_label(d) for d in diseases]

    # y ticks = diseases on the right
    ax_bot.set_yticks(np.arange(n_dis))
    ax_bot.set_yticklabels(clean_diseases, fontsize=10)
    ax_bot.yaxis.tick_right()

    # x ticks = genes (same order as top)
    ax_bot.set_xticks(np.arange(n_genes))
    ax_bot.set_xticklabels(gene_order, rotation=90, fontsize=8)

    ax_bot.set_xlabel("Gene")
    ax_bot.set_ylabel("Disease")

    # remove frame spines
    for sp in ax_bot.spines.values():
        sp.set_visible(False)

    plt.tight_layout()
    outfile = os.path.join(output_directory, 'overall_rank_genes.png')
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()

    


    



df = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/PhD/NDD/GENES/refined_attempt_2/gene/merged_gene.csv')
output_directory = '/Volumes/lab-windingm/home/users/cochral/PhD/NDD/GENES/refined_attempt_2/gene/weighted_score_filtered'

df = modify_df(df)
df = filter(df, 'filtered_weighted_score')
genes_scoring_ranked(df, output_directory)






# df = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/PhD/NDD/GENES/refined_attempt_2/gene_automated_alias/merged_gene_automated_alias.csv')
# output_directory = '/Volumes/lab-windingm/home/users/cochral/PhD/NDD/GENES/refined_attempt_2/gene_automated_alias/weighted_score_filtered'

# """ No filter on genes """
# df = modify_df(df)
# df = filter(df, 'filtered_weighted_score')
# weighted_ortholog_score(df, output_directory)
# genes_disease_weightedscore(df, output_directory)
# scatter_similarity_identitiy(df, output_directory, include_hue=True)
# scatter_similarity_identitiy_disease (df, output_directory)
# scatter_similarity_identitiy_per_disease(df, output_directory, include_hue=True)
# scatter_similarity_identitiy_weightedscore(df, output_directory)
# barplot_pubmed(df, output_directory, include_hue=True)
# genes_disease(df, output_directory, include_hue=True)
# genes_disease_heatmap(df, output_directory)
# genes_ontology_heatmap(df, output_directory)
# disease_ontology_heatmap(df, output_directory)
# genes_ranked_pubmed_all(df, output_directory)
# genes_ranked_total_mouse(df, output_directory)
# genes_ranked_total_drosoph(df, output_directory)







# output_directory = '/Volumes/lab-windingm/home/users/cochral/PhD/NDD/GENES/refined_attempt/alias_used/high_rank_only'

# """ Filter the genes for best score = True aka highly ranked genes"""
# df = modify_df(df)
# df = filter(df, 'both_best_score')
# weighted_ortholog_score(df, output_directory)
# genes_disease_weightedscore(df, output_directory)
# scatter_similarity_identitiy(df, output_directory, include_hue=False)
# scatter_similarity_identitiy_disease (df, output_directory)
# scatter_similarity_identitiy_weightedscore(df, output_directory)
# scatter_similarity_identitiy_per_disease(df, output_directory, include_hue=False)
# barplot_pubmed(df, output_directory, include_hue=False)
# genes_disease(df, output_directory, include_hue=False)
# genes_disease_heatmap(df, output_directory)
# genes_ontology_heatmap(df, output_directory)
# disease_ontology_heatmap(df, output_directory)
# genes_ranked_pubmed_all(df, output_directory)
# genes_ranked_total_mouse(df, output_directory)
# genes_ranked_total_drosoph(df, output_directory)



"""Ranked Gene List Comparison"""
# df1 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/PhD/NDD/GENES/refined_attempt_2/gene/merged_gene.csv')
# df2 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/PhD/NDD/GENES/refined_attempt_2/gene_automated_alias/merged_gene_automated_alias.csv')
# output_directory = '/Volumes/lab-windingm/home/users/cochral/PhD/NDD/GENES/refined_attempt_2/gene_automated_alias'
# df1 = modify_df(df1)
# df1 = filter(df1, 'filtered_weighted_score')
# df2 = modify_df(df2)
# df2 = filter(df2, 'filtered_weighted_score')
# genes_alias_ranked_pubmed(df1, df2, output_directory)