import sys
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import pyarrow.feather as feather
import cv2
import numpy as np
from shapely.geometry import Point, Polygon
from shapely import wkt
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt




    #### METHOD PARTNER_MORPHOLOGY:
    def relative_partner_metrics(self):

        """ relative_step_length = how far the partner moved relative to the anchor
            forward_progress = how much of that movement was toward the anchor
            sideways_wiggle = how much of the partners movement went sideways, instead of toward or away from the anchor
            forward_change = how much the forward_progress changed from the last frame
            auc_pre = total forward_progress before frame 0
            auc_post = total forward_progress after frame 0
            auc_net = overall bias (before + after)
            auc_net_magnitude = overall size of the behavior, ignoring direction
        """

        df = self.df.copy()
        cluster_name = self.cluster_name
        eps = 1e-9 # dont divide by 0 ltr on 

        # relative partner to anchor xy coordinates 
        anchor_position = np.array([df['anchor x_body'], 


        df['relative_partner_x'] = df['partner x_body'] - df['anchor x_body']
        df['relative_partner_y'] = df['partner y_body'] - df['anchor y_body']

        # partner distance from anchor
        df['distance_from_anchor'] = np.sqrt(df['relative_partner_x']**2 + df['relative_partner_y']**2)

        df = df.sort_values(['interaction_id', 'Normalized Frame'])

        # Previous-frame values
        df['prev_relative_partner_x'] = df.groupby('interaction_id')['relative_partner_x'].shift(1)
        df['prev_relative_partner_y'] = df.groupby('interaction_id')['relative_partner_y'].shift(1)
        df['prev_distance_from_anchor'] = df.groupby('interaction_id')['distance_from_anchor'].shift(1)

        # movement from the previous frame to the current frame
        df['relative_step_x'] = df['relative_partner_x'] - df['prev_relative_partner_x']
        df['relative_step_y'] = df['relative_partner_y'] - df['prev_relative_partner_y']
        df['relative_step_length'] = np.sqrt(df['relative_step_x']**2 + df['relative_step_y']**2)

        # Unit direction (anchor -> partner) at previous frame
        df['prev_direction_x'] = df['prev_relative_partner_x'] / (df['prev_distance_from_anchor'] + eps)
        df['prev_direction_y'] = df['prev_relative_partner_y'] / (df['prev_distance_from_anchor'] + eps)

        # Signed progress toward anchor: positive = toward, negative = away
        df['forward_progress'] = -(
            df['relative_step_x'] * df['prev_direction_x'] +
            df['relative_step_y'] * df['prev_direction_y'])

        # Sideways wiggle: how much movement occurred off-axis (lateral deviation)
        df['sideways_wiggle'] = np.sqrt(
            np.maximum(0.0, df['relative_step_length']**2 - df['forward_progress']**2)
        )

        df['prev_forward_progress'] = df.groupby('interaction_id')['forward_progress'].shift(1)

        # acceleration: change in forward_progress frame-to-frame
        df['forward_change'] = df['forward_progress'] - df['prev_forward_progress']
 

        out_cols = [
            'interaction_id',
            'Normalized Frame',
            'forward_progress',          # per-frame score
            'relative_step_length',      
            'distance_from_anchor',      
            'sideways_wiggle',       # lateral deviation magnitude
            'forward_change',         # rate of change of that speed (acceleration)
            cluster_name,
            'condition'
        ]
        
        traj_progress = df[out_cols].copy()

        outdir = os.path.join(self.directory, "relative_partner_metrics")
        os.makedirs(outdir, exist_ok=True)
        outpath = os.path.join(outdir, "partner_morphology.csv")
        traj_progress.to_csv(outpath, index=False)

        grp = df.groupby('interaction_id', sort=False)

        pre_auc = grp.apply(
            lambda g: g.loc[g['Normalized Frame'] < 0, 'forward_progress'].sum()).rename('auc_pre')

        post_auc = grp.apply(
            lambda g: g.loc[g['Normalized Frame'] > 0, 'forward_progress'].sum()).rename('auc_post')

        auc = (
            pre_auc.reset_index()
            .merge(post_auc.reset_index(), on='interaction_id', how='outer'))
        
        auc['auc_net'] = auc['auc_pre'].fillna(0) + auc['auc_post'].fillna(0)
        auc['auc_net_magnitiude'] = auc['auc_pre'].fillna(0) + (-auc['auc_post'].fillna(0))

        # Attach labels (one row per interaction)
        lookup = (
            df[['interaction_id', self.cluster_name, 'condition']]
            .drop_duplicates('interaction_id')
        )
        auc = auc.merge(lookup, on='interaction_id', how='left')

        auc_path = os.path.join(outdir, "partner_morphology_auc.csv")
        auc.to_csv(auc_path, index=False)

