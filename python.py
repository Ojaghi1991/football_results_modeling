"""# Best one + Home effect"""

import pandas as pd
import numpy as np
import pymc as pm
import pytensor.tensor as pt
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class BayesianFootballModel:
    """
    Bayesian hierarchical model for football match prediction
    Based on Baio & Blangiardo (2010) paper
    """

    def __init__(self, data_file):
        """Initialize the model with data"""
        print(f"Initializing model with data file: {data_file}")

        # Initialize model attributes first (but not data attributes)
        self.basic_model = None
        self.mixture_model = None
        self.basic_trace = None
        self.mixture_trace = None

        try:
            self.data = self.load_and_prepare_data(data_file)
            print("✓ Model initialization completed successfully")
            print(f"✓ Final check - n_games: {self.n_games}, n_teams: {self.n_teams}")
        except Exception as e:
            print(f"✗ Error during model initialization: {e}")
            print("Please check that the data file exists and has the correct format")
            raise

    def load_and_prepare_data(self, data_file):
        """Load and prepare the football data"""
        # Read the Excel file
        df = pd.read_excel('/content/finaldataset2007-08.xlsx')

        # Clean column names
        df.columns = df.columns.str.strip()

        print(f"Original data shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")

        # Create team mappings
        all_teams = pd.concat([
            df['hometeam_name'],
            df['awayteam_name']
        ]).unique()

        team_to_id = {team: i for i, team in enumerate(sorted(all_teams))}
        id_to_team = {i: team for team, i in team_to_id.items()}

        # Map team names to consecutive IDs (0-based)
        df['home_team_idx'] = df['hometeam_name'].map(team_to_id)
        df['away_team_idx'] = df['awayteam_name'].map(team_to_id)

        # Check for any mapping issues
        if df['home_team_idx'].isna().any() or df['away_team_idx'].isna().any():
            print("Warning: Some teams could not be mapped!")
            print("Home team mapping issues:", df[df['home_team_idx'].isna()]['hometeam_name'].unique())
            print("Away team mapping issues:", df[df['away_team_idx'].isna()]['awayteam_name'].unique())

        # Store team information
        self.teams = sorted(all_teams)
        self.n_teams = len(self.teams)
        self.n_games = len(df)

        print(f"Data loaded: {self.n_games} games, {self.n_teams} teams")
        print(f"Teams: {self.teams}")

        # Verify no None values
        print(f"n_games type: {type(self.n_games)}, value: {self.n_games}")
        print(f"n_teams type: {type(self.n_teams)}, value: {self.n_teams}")

        return df

    def build_basic_model(self):
        """Build the basic hierarchical model from Section 2 of the paper"""

        # Check if data is properly loaded
        if self.n_games is None or self.n_teams is None:
            raise ValueError("Data not properly loaded. Please check the data file and team mappings.")

        print(f"Building model with {self.n_games} games and {self.n_teams} teams")

        # Prepare data arrays
        home_team_idx = self.data['home_team_idx'].values
        away_team_idx = self.data['away_team_idx'].values
        y1_data = self.data['y1'].values
        y2_data = self.data['y2'].values

        # Verify data integrity
        print(f"Home team indices range: {home_team_idx.min()} to {home_team_idx.max()}")
        print(f"Away team indices range: {away_team_idx.min()} to {away_team_idx.max()}")
        print(f"Goals range - Home: {y1_data.min()} to {y1_data.max()}, Away: {y2_data.min()} to {y2_data.max()}")

        with pm.Model() as model:
            # Home advantage parameter
            home_advantage = pm.Normal("home_advantage", mu=0, sigma=10)

            # Hyperparameters for attack and defense effects
            mu_att = pm.Normal("mu_att", mu=0, sigma=10)
            mu_def = pm.Normal("mu_def", mu=0, sigma=10)
            tau_att = pm.Gamma("tau_att", alpha=0.1, beta=0.1)
            tau_def = pm.Gamma("tau_def", alpha=0.1, beta=0.1)

            # Team-specific attack and defense effects (before centering)
            att_star = pm.Normal("att_star", mu=mu_att, sigma=1/pt.sqrt(tau_att), shape=self.n_teams)
            def_star = pm.Normal("def_star", mu=mu_def, sigma=1/pt.sqrt(tau_def), shape=self.n_teams)

            # Sum-to-zero constraint (centering)
            att = pm.Deterministic("att", att_star - pt.mean(att_star))
            def_ = pm.Deterministic("def", def_star - pt.mean(def_star))

            # Create indicator matrices for team assignments
            print("Creating indicator matrices...")
            home_team_matrix = np.zeros((int(self.n_games), int(self.n_teams)))
            away_team_matrix = np.zeros((int(self.n_games), int(self.n_teams)))

            for i in range(int(self.n_games)):
                home_team_matrix[i, int(home_team_idx[i])] = 1
                away_team_matrix[i, int(away_team_idx[i])] = 1

            print("Converting to PyTensor constants...")
            # Convert to PyTensor constants
            home_team_matrix = pt.constant(home_team_matrix)
            away_team_matrix = pt.constant(away_team_matrix)

            # Calculate team-specific effects for each game using matrix multiplication
            home_att_effects = pt.dot(home_team_matrix, att)
            away_att_effects = pt.dot(away_team_matrix, att)
            home_def_effects = pt.dot(home_team_matrix, def_)
            away_def_effects = pt.dot(away_team_matrix, def_)

            # Scoring intensities
            log_theta1 = home_advantage + home_att_effects + away_def_effects
            log_theta2 = away_att_effects + home_def_effects

            theta1 = pm.Deterministic("theta1", pt.exp(log_theta1))
            theta2 = pm.Deterministic("theta2", pt.exp(log_theta2))

            # Likelihood
            y1 = pm.Poisson("y1", mu=theta1, observed=y1_data)
            y2 = pm.Poisson("y2", mu=theta2, observed=y2_data)

        print("Model built successfully!")
        self.basic_model = model
        return model

    def build_mixture_model(self):
        """Build the mixture model from Section 4 of the paper"""

        # Prepare data arrays
        home_team_idx = self.data['home_team_idx'].values
        away_team_idx = self.data['away_team_idx'].values
        y1_data = self.data['y1'].values
        y2_data = self.data['y2'].values

        with pm.Model() as model:
            # Home advantage parameter
            home_advantage = pm.Normal("home_advantage", mu=0, sigma=10)

            # Mixture parameters for each team
            # Prior probabilities for group membership (3 groups: bottom, mid, top)
            alpha_att = np.ones(3)  # Uniform prior over groups
            alpha_def = np.ones(3)

            p_att = pm.Dirichlet("p_att", a=alpha_att, shape=(self.n_teams, 3))
            p_def = pm.Dirichlet("p_def", a=alpha_def, shape=(self.n_teams, 3))

            # Group assignment for each team
            grp_att = pm.Categorical("grp_att", p=p_att, shape=self.n_teams)
            grp_def = pm.Categorical("grp_def", p=p_def, shape=self.n_teams)

            # Group-specific parameters
            # Group 1: Bottom teams (poor attack, poor defense)
            mu_att_1 = pm.TruncatedNormal("mu_att_1", mu=0, sigma=10, lower=-3, upper=0)
            mu_def_1 = pm.TruncatedNormal("mu_def_1", mu=0, sigma=10, lower=0, upper=3)
            tau_att_1 = pm.Gamma("tau_att_1", alpha=0.01, beta=0.01)
            tau_def_1 = pm.Gamma("tau_def_1", alpha=0.01, beta=0.01)

            # Group 2: Mid-table teams (average)
            mu_att_2 = pt.constant(0.0)
            mu_def_2 = pt.constant(0.0)
            tau_att_2 = pm.Gamma("tau_att_2", alpha=0.01, beta=0.01)
            tau_def_2 = pm.Gamma("tau_def_2", alpha=0.01, beta=0.01)

            # Group 3: Top teams (good attack, good defense)
            mu_att_3 = pm.TruncatedNormal("mu_att_3", mu=0, sigma=10, lower=0, upper=3)
            mu_def_3 = pm.TruncatedNormal("mu_def_3", mu=0, sigma=10, lower=-3, upper=0)
            tau_att_3 = pm.Gamma("tau_att_3", alpha=0.01, beta=0.01)
            tau_def_3 = pm.Gamma("tau_def_3", alpha=0.01, beta=0.01)

            # Stack parameters
            mu_att_groups = pt.stack([mu_att_1, mu_att_2, mu_att_3])
            mu_def_groups = pt.stack([mu_def_1, mu_def_2, mu_def_3])
            tau_att_groups = pt.stack([tau_att_1, tau_att_2, tau_att_3])
            tau_def_groups = pt.stack([tau_def_1, tau_def_2, tau_def_3])

            # Team-specific effects using Normal distribution (simplified from t-distribution)
            # We'll use a simpler approach due to indexing complexity with StudentT and mixture
            att_raw = pm.Normal("att_raw", mu=0, sigma=1, shape=self.n_teams)
            def_raw = pm.Normal("def_raw", mu=0, sigma=1, shape=self.n_teams)

            # Scale by group-specific parameters
            att_scaled = []
            def_scaled = []

            for t in range(self.n_teams):
                att_mu_t = pt.switch(pt.eq(grp_att[t], 0), mu_att_groups[0],
                                   pt.switch(pt.eq(grp_att[t], 1), mu_att_groups[1], mu_att_groups[2]))
                att_tau_t = pt.switch(pt.eq(grp_att[t], 0), tau_att_groups[0],
                                    pt.switch(pt.eq(grp_att[t], 1), tau_att_groups[1], tau_att_groups[2]))

                def_mu_t = pt.switch(pt.eq(grp_def[t], 0), mu_def_groups[0],
                                   pt.switch(pt.eq(grp_def[t], 1), mu_def_groups[1], mu_def_groups[2]))
                def_tau_t = pt.switch(pt.eq(grp_def[t], 0), tau_def_groups[0],
                                    pt.switch(pt.eq(grp_def[t], 1), tau_def_groups[1], tau_def_groups[2]))

                att_t = att_mu_t + att_raw[t] / pt.sqrt(att_tau_t)
                def_t = def_mu_t + def_raw[t] / pt.sqrt(def_tau_t)

                att_scaled.append(att_t)
                def_scaled.append(def_t)

            att = pt.stack(att_scaled)
            def_ = pt.stack(def_scaled)

            # Apply sum-to-zero constraint
            att_centered = pm.Deterministic("att_centered", att - pt.mean(att))
            def_centered = pm.Deterministic("def_centered", def_ - pt.mean(def_))

            # Create indicator matrices for team assignments
            home_team_matrix = np.zeros((self.n_games, self.n_teams))
            away_team_matrix = np.zeros((self.n_games, self.n_teams))

            for i in range(self.n_games):
                home_team_matrix[i, home_team_idx[i]] = 1
                away_team_matrix[i, away_team_idx[i]] = 1

            # Convert to PyTensor constants
            home_team_matrix = pt.constant(home_team_matrix)
            away_team_matrix = pt.constant(away_team_matrix)

            # Calculate team-specific effects for each game using matrix multiplication
            home_att_effects = pt.dot(home_team_matrix, att_centered)
            away_att_effects = pt.dot(away_team_matrix, att_centered)
            home_def_effects = pt.dot(home_team_matrix, def_centered)
            away_def_effects = pt.dot(away_team_matrix, def_centered)

            # Scoring intensities
            log_theta1 = home_advantage + home_att_effects + away_def_effects
            log_theta2 = away_att_effects + home_def_effects

            theta1 = pm.Deterministic("theta1", pt.exp(log_theta1))
            theta2 = pm.Deterministic("theta2", pt.exp(log_theta2))

            # Likelihood
            y1 = pm.Poisson("y1", mu=theta1, observed=y1_data)
            y2 = pm.Poisson("y2", mu=theta2, observed=y2_data)

        self.mixture_model = model
        return model

    def fit_basic_model(self, draws=2000, tune=1000, chains=2, cores=1):
        """Fit the basic hierarchical model"""
        print("Fitting basic hierarchical model...")

        if self.basic_model is None:
            self.build_basic_model()

        with self.basic_model:
            # Sample from posterior
            self.basic_trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                cores=cores,
                random_seed=42,
                return_inferencedata=True,
                target_accept=0.9  # Higher target acceptance for better sampling
            )

            # Sample posterior predictive
            with self.basic_model:
                self.basic_trace.extend(pm.sample_posterior_predictive(self.basic_trace))

        print("Basic model fitting completed!")
        return self.basic_trace

    def fit_mixture_model(self, draws=2000, tune=1000, chains=2, cores=1):
        """Fit the mixture model"""
        print("Fitting mixture model...")

        if self.mixture_model is None:
            self.build_mixture_model()

        with self.mixture_model:
            # Sample from posterior with simplified parameters
            self.mixture_trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                cores=cores,
                random_seed=42,
                return_inferencedata=True,
                target_accept=0.9  # Slightly lower target acceptance to avoid issues
            )

            # Sample posterior predictive
            with self.mixture_model:
                self.mixture_trace.extend(pm.sample_posterior_predictive(self.mixture_trace))

        print("Mixture model fitting completed!")
        return self.mixture_trace

    def get_home_advantage_summary(self, model_type='basic'):
        """Get summary statistics for home advantage effect"""

        trace = self.basic_trace if model_type == 'basic' else self.mixture_trace

        if trace is None:
            print(f"Please fit the {model_type} model first!")
            return None

        # Extract home advantage samples
        home_adv_samples = trace.posterior['home_advantage']

        # Calculate summary statistics
        home_summary = {
            'parameter': 'home_advantage',
            'mean': float(home_adv_samples.mean()),
            'median': float(home_adv_samples.median()),
            'std': float(home_adv_samples.std()),
            'q025': float(home_adv_samples.quantile(0.025)),
            'q975': float(home_adv_samples.quantile(0.975))
        }

        return home_summary

    def plot_team_effects(self, model_type='basic'):
        """Plot attack vs defense effects for each team"""

        trace = self.basic_trace if model_type == 'basic' else self.mixture_trace

        if trace is None:
            print(f"Please fit the {model_type} model first!")
            return

        # Get posterior means
        if model_type == 'basic':
            att_means = trace.posterior['att'].mean(dim=['chain', 'draw']).values
            def_means = trace.posterior['def'].mean(dim=['chain', 'draw']).values
        else:
            att_means = trace.posterior['att_centered'].mean(dim=['chain', 'draw']).values
            def_means = trace.posterior['def_centered'].mean(dim=['chain', 'draw']).values

        # Create plot
        plt.figure(figsize=(12, 8))
        plt.scatter(att_means, def_means, s=100, alpha=0.7)

        # Add team labels
        for i, team in enumerate(self.teams):
            plt.annotate(team, (att_means[i], def_means[i]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.8)

        plt.xlabel('Attack Effect')
        plt.ylabel('Defense Effect')
        plt.title(f'Team Attack vs Defense Effects ({model_type.title()} Model)')
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        plt.axvline(x=0, color='k', linestyle='--', alpha=0.5)

        # Add quadrant labels
        plt.text(0.02, 0.98, 'Good Attack,\nPoor Defense',
                transform=plt.gca().transAxes, va='top', ha='left',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        plt.text(0.98, 0.98, 'Poor Attack,\nPoor Defense',
                transform=plt.gca().transAxes, va='top', ha='right',
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
        plt.text(0.02, 0.02, 'Good Attack,\nGood Defense',
                transform=plt.gca().transAxes, va='bottom', ha='left',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        plt.text(0.98, 0.02, 'Poor Attack,\nGood Defense',
                transform=plt.gca().transAxes, va='bottom', ha='right',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

        plt.tight_layout()
        plt.show()

    def get_team_summary(self, model_type='basic'):
        """Get summary statistics for team effects"""

        trace = self.basic_trace if model_type == 'basic' else self.mixture_trace

        if trace is None:
            print(f"Please fit the {model_type} model first!")
            return None

        # Get parameter names
        if model_type == 'basic':
            att_param = 'att'
            def_param = 'def'
        else:
            att_param = 'att_centered'
            def_param = 'def_centered'

        # Extract posterior samples
        att_samples = trace.posterior[att_param]
        def_samples = trace.posterior[def_param]

        # Calculate summary statistics
        summary_data = []
        for i, team in enumerate(self.teams):
            # Handle different indexing based on parameter dimensions
            if len(att_samples.dims) == 3:  # chain, draw, team
                att_team_samples = att_samples.isel({list(att_samples.dims)[2]: i})
                def_team_samples = def_samples.isel({list(def_samples.dims)[2]: i})
            else:  # Different structure
                att_team_samples = att_samples[..., i]
                def_team_samples = def_samples[..., i]

            att_mean = float(att_team_samples.mean())
            att_q025 = float(att_team_samples.quantile(0.025))
            att_q975 = float(att_team_samples.quantile(0.975))

            def_mean = float(def_team_samples.mean())
            def_q025 = float(def_team_samples.quantile(0.025))
            def_q975 = float(def_team_samples.quantile(0.975))

            summary_data.append({
                'team': team,
                'att_mean': att_mean,
                'att_q025': att_q025,
                'att_q975': att_q975,
                'def_mean': def_mean,
                'def_q025': def_q025,
                'def_q975': def_q975
            })

        summary_df = pd.DataFrame(summary_data)

        # Sort by attack effect (descending)
        summary_df = summary_df.sort_values('att_mean', ascending=False)

        return summary_df

    def print_model_summary(self, model_type='basic'):
        """Print comprehensive model summary including home advantage"""

        print(f"\n{model_type.upper()} MODEL SUMMARY")
        print("=" * 60)

        # Home advantage
        home_summary = self.get_home_advantage_summary(model_type)
        if home_summary:
            print(f"\nHOME ADVANTAGE EFFECT:")
            print(f"Mean: {home_summary['mean']:.4f}")
            print(f"95% CI: [{home_summary['q025']:.4f}, {home_summary['q975']:.4f}]")
            print(f"Interpretation: Home teams score exp({home_summary['mean']:.4f}) = {np.exp(home_summary['mean']):.3f}x more goals on average")

        # Team effects
        print(f"\nTEAM EFFECTS:")
        team_summary = self.get_team_summary(model_type)
        if team_summary is not None:
            print("\nTop 5 Attack (most goals scored):")
            print(team_summary.head()[['team', 'att_mean', 'att_q025', 'att_q975']].to_string(index=False))

            print(f"\nTop 5 Defense (fewest goals conceded - most negative values):")
            defense_sorted = team_summary.sort_values('def_mean', ascending=True)
            print(defense_sorted.head()[['team', 'def_mean', 'def_q025', 'def_q975']].to_string(index=False))

            print(f"\nBottom 5 Attack (fewest goals scored):")
            attack_sorted = team_summary.sort_values('att_mean', ascending=True)
            print(attack_sorted.head()[['team', 'att_mean', 'att_q025', 'att_q975']].to_string(index=False))

            print(f"\nBottom 5 Defense (most goals conceded - most positive values):")
            print(defense_sorted.tail()[['team', 'def_mean', 'def_q025', 'def_q975']].to_string(index=False))

    def predict_match(self, home_team, away_team, model_type='basic', n_samples=1000):
        """Predict the outcome of a specific match"""

        trace = self.basic_trace if model_type == 'basic' else self.mixture_trace

        if trace is None:
            print(f"Please fit the {model_type} model first!")
            return None

        # Get team indices
        home_idx = self.teams.index(home_team)
        away_idx = self.teams.index(away_team)

        # Get parameter samples
        home_adv_samples = trace.posterior['home_advantage'].values.flatten()

        if model_type == 'basic':
            att_samples = trace.posterior['att'].values
            def_samples = trace.posterior['def'].values
        else:
            att_samples = trace.posterior['att_centered'].values
            def_samples = trace.posterior['def_centered'].values

        # Reshape samples for easier indexing
        att_flat = att_samples.reshape(-1, att_samples.shape[-1])
        def_flat = def_samples.reshape(-1, def_samples.shape[-1])
        home_adv_flat = home_adv_samples.flatten()

        # Take only n_samples
        n_available = min(len(home_adv_flat), len(att_flat))
        n_use = min(n_samples, n_available)

        # Calculate scoring intensities
        theta1_samples = np.exp(home_adv_flat[:n_use] +
                               att_flat[:n_use, home_idx] +
                               def_flat[:n_use, away_idx])

        theta2_samples = np.exp(att_flat[:n_use, away_idx] +
                               def_flat[:n_use, home_idx])

        # Generate predictions
        home_goals = np.random.poisson(theta1_samples)
        away_goals = np.random.poisson(theta2_samples)

        # Calculate probabilities
        home_win = np.mean(home_goals > away_goals)
        draw = np.mean(home_goals == away_goals)
        away_win = np.mean(home_goals < away_goals)

        # Expected goals
        exp_home_goals = np.mean(theta1_samples)
        exp_away_goals = np.mean(theta2_samples)

        return {
            'home_team': home_team,
            'away_team': away_team,
            'expected_home_goals': exp_home_goals,
            'expected_away_goals': exp_away_goals,
            'prob_home_win': home_win,
            'prob_draw': draw,
            'prob_away_win': away_win,
            'home_goals_samples': home_goals,
            'away_goals_samples': away_goals
        }


