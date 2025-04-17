import numpy as np
from typing import Tuple
import numpy.typing as npt
import pandas as pd
from shapely.geometry import Point
import yaml

from autonomous_separation.traffic import cre_conflict
from autonomous_separation.conf_detect.state_based import StateBasedDetection
from autonomous_separation.conf_reso.algorithms.MVP import MVPResolution
from autonomous_separation.conf_reso.algorithms.VO import VOResolution

class ConflictResolutionSimulation:
    def __init__(self, case_title_selected, source_of_uncertainty):
        with open("uncertainty_quantification/uq_sim_config.yaml", "r") as f:
            self.config = yaml.safe_load(f)

        # Ownship states
        self.case_title_selected = case_title_selected # speed/heading/pos
        self.source_of_uncertainty = source_of_uncertainty # ownship/intruder

        self.x_own = self.config['ownship']['x']
        self.y_own = self.config['ownship']['y']
        self.hdg_own = self.config['ownship']['heading']
        self.gs_own = self.config['ownship']['gs']

        # Loop parameters
        self.gs_int = self.config['intruder']['gs']

        self.dcpa_start = self.config['dcpa']['dcpa_start']
        self.dcpa_end = self.config['dcpa']['dcpa_end']
        self.dcpa_delta = self.config['dcpa']['dcpa_delta']
        self.dpsi_start = self.config['heading_diff']['dpsi_start']
        self.dpsi_end = self.config['heading_diff']['dpsi_end']
        self.dpsi_delta = self.config['heading_diff']['dpsi_delta']

        # CDR params
        self.tlosh = 15
        self.rpz = 50

        # For final plot axis limits
        self.vy_init = self.gs_own * np.sin(np.radians(self.hdg_own))
        self.vx_init = self.gs_own * np.cos(np.radians(self.hdg_own))

        self.nb_samples = self.config['nb_samples']
        self.alpha_uncertainty = 0.4

        # Uncertainty switches (defaults to False)
        self.pos_uncertainty_on = False
        self.hdg_uncertainty_on = False
        self.spd_uncertainty_on = False
        self.src_ownship_on = False
        self.src_intruder_on = False

        # Noise parameters
        self.sigma = 0
        self.hdg_sigma_ownship = 0
        self.hdg_sigma_intruder = 0
        self.gs_sigma_ownship = 0
        self.gs_sigma_intruder = 0

        # For user selections
        self.case_title_selected = case_title_selected
        self.source_of_uncertainty = source_of_uncertainty

        self.reso_algo = None

        if('s' in self.case_title_selected):
            self.spd_uncertainty_on = True
        if('h' in self.case_title_selected):
            self.hdg_uncertainty_on = True
        if('p' in self.case_title_selected):
            self.pos_uncertainty_on = True

        if('o' in self.source_of_uncertainty):
            self.src_ownship_on = True
        if('i' in self.source_of_uncertainty):
            self.src_intruder_on = True

        self.set_noise_parameters()

    def set_noise_parameters(self):
        """
        Adjust noise parameters (standard deviations) based on which
        uncertainties and sources are active.
        """
        # Position noise
        if self.pos_uncertainty_on:
            self.sigma = self.config['uncertainties']['pos_sigma']  # Example standard deviation for position

        # Heading noise
        if self.hdg_uncertainty_on:
            if self.src_ownship_on:
                self.hdg_sigma_ownship = self.config['uncertainties']['hdg_sigma']
            if self.src_intruder_on:
                self.hdg_sigma_intruder = self.config['uncertainties']['hdg_sigma']

        # Speed noise
        if self.spd_uncertainty_on:
            if self.src_ownship_on:
                self.gs_sigma_ownship = self.config['uncertainties']['spd_sigma']
            if self.src_intruder_on:
                self.gs_sigma_intruder = self.config['uncertainties']['spd_sigma']

    def create_pos_noise_samples(
        self,
        x_ground_truth: float,
        y_ground_truth: float,
        sigma: float,
        nb_samples: int = 10000
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        
        std_dev = sigma / 2.448 # this one ensures 95% of the data is within 2 sigma for a 2D gaussian

        cov = np.array([[std_dev**2, 0], 
                        [0, std_dev**2]])
        
        # Generate random samples from multivariate normal distribution
        x, y = np.random.multivariate_normal((0, 0), cov, nb_samples).T

        x_noise = x_ground_truth + x
        y_noise = y_ground_truth + y

        return x_noise, y_noise

    def run_simulation(self, dpsi_val, dcpa_val):
        # instatiate the CD and CR
        cd = StateBasedDetection()
        vo = VOResolution()
        mvp = MVPResolution()

        self.dpsi_val = dpsi_val
        self.dcpa_val = dcpa_val
        
        # --- 1. Intruder scenario creation ---
        self.x_int, self.y_int, self.hdg_int, gs_int = cre_conflict(
            self.x_own, self.y_own, self.hdg_own, self.gs_own,
            dpsi_val, dcpa_val, self.tlosh, self.gs_int, self.rpz
        )

        # --- 2. Create noisy positions ---
        x_o, y_o = self.create_pos_noise_samples(
            self.x_own, self.y_own, self.sigma, self.nb_samples
        )
        x_i, y_i = self.create_pos_noise_samples(
            self.x_int, self.y_int, self.sigma, self.nb_samples
        )

        # --- 3. Create noisy heading & speed for ownship/intruder ---
        hdg_ownship = np.random.normal(
            self.hdg_own, self.hdg_sigma_ownship, self.nb_samples
        )
        gs_ownship = np.random.normal(
            self.gs_own, self.gs_sigma_ownship, self.nb_samples
        )
        hdg_intruder = np.random.normal(
            self.hdg_int, self.hdg_sigma_intruder, self.nb_samples
        )
        gs_intruder = np.random.normal(
            gs_int, self.gs_sigma_intruder, self.nb_samples
        )

        # --- 4. Build DataFrame of samples ---
        df = pd.DataFrame({
            'x_own_true': self.x_own,
            'y_own_true': self.y_own,
            'x_own_noise': x_o,
            'y_own_noise': y_o,
            'hdg_own_true': self.hdg_own,
            'gs_own_true': self.gs_own,
            'hdg_own_noise': hdg_ownship,
            'gs_own_noise': gs_ownship,
            'x_int_true': self.x_int,
            'y_int_true': self.y_int,
            'x_int_noise': x_i,
            'y_int_noise': y_i,
            'hdg_int_true': self.hdg_int,
            'gs_int_true': gs_int,
            'hdg_int_noise': hdg_intruder,
            'gs_int_noise': gs_intruder
        })

        df['pos_ownship'] = [
            Point(a, b) for a, b in zip(df['x_own_noise'], df['y_own_noise'])
        ]
        df['pos_intruder'] = [
            Point(a, b) for a, b in zip(df['x_int_noise'], df['y_int_noise'])
        ]

        # --- 5. Detect conflicts, apply conflict resolution ---
        df[['tin', 'tout', 'dcpa', 'is_conflict']] = df.apply(
            lambda row: pd.Series(cd.conf_detect_hor(
                ownship_position=row['pos_ownship'],
                ownship_gs=row['gs_own_noise'],
                ownship_trk=row['hdg_own_noise'],
                intruder_position=row['pos_intruder'],
                intruder_gs=row['gs_int_noise'],
                intruder_trk=row['hdg_int_noise'],
                rpz=self.rpz,
                tlookahead=self.tlosh
            )),
            axis=1
        )
        
        df[['vx_vo', 'vy_vo']] = df.apply(
            lambda row: pd.Series(vo.resolve(
                ownship_position=row['pos_ownship'],
                ownship_gs=row['gs_own_noise'],
                ownship_trk=row['hdg_own_noise'],
                intruder_position=row['pos_intruder'],
                intruder_gs=row['gs_int_noise'],
                intruder_trk=row['hdg_int_noise'],
                rpz=self.rpz,
                tlookahead=self.tlosh,
                method=0
            )),
            axis=1
        )

        df[['vx_mvp', 'vy_mvp']] = df.apply(
            lambda row: pd.Series(mvp.resolve(
                ownship_position=row['pos_ownship'],
                ownship_gs=row['gs_own_noise'],
                ownship_trk=row['hdg_own_noise'],
                intruder_position=row['pos_intruder'],
                intruder_gs=row['gs_int_noise'],
                intruder_trk=row['hdg_int_noise'],
                rpz=self.rpz,
                tlookahead=self.tlosh,
            )),
            axis=1
        )

        return df