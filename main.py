from python import BayesianFootballModel
# Usage example

if __name__ == "__main__":
    # Initialize the model with your data file
    # Replace 'finaldataset2007-08.xlsx' with the actual file path
    model = BayesianFootballModel('finaldataset2007-08.xlsx')

    # Fit the basic model first (simpler and more stable)
    print("=" * 50)
    print("FITTING BASIC MODEL")
    print("=" * 50)
    basic_trace = model.fit_basic_model(draws=1000, tune=500, chains=2)

    # Get comprehensive model summary including home advantage
    print("\n" + "=" * 50)
    print("BASIC MODEL COMPREHENSIVE SUMMARY")
    print("=" * 50)
    model.print_model_summary('basic')

    # Plot team effects
    model.plot_team_effects('basic')

    # Example prediction
    print("\n" + "=" * 50)
    print("MATCH PREDICTION EXAMPLE (BASIC MODEL)")
    print("=" * 50)
    prediction = model.predict_match('Internazionale', 'Milan', 'basic')
    if prediction:
        print(f"Match: {prediction['home_team']} vs {prediction['away_team']}")
        print(f"Expected goals: {prediction['expected_home_goals']:.2f} - {prediction['expected_away_goals']:.2f}")
        print(f"Probabilities: Home {prediction['prob_home_win']:.3f}, Draw {prediction['prob_draw']:.3f}, Away {prediction['prob_away_win']:.3f}")

    # Uncomment the following if you want to try the mixture model:
    # Note: The mixture model is more complex and may take longer to fit

    print("\n" + "=" * 50)
    print("FITTING MIXTURE MODEL")
    print("=" * 50)
    try:
        mixture_trace = model.fit_mixture_model(draws=500, tune=300, chains=2)  # Reduced draws for faster testing

        print("\n" + "=" * 50)
        print("MIXTURE MODEL COMPREHENSIVE SUMMARY")
        print("=" * 50)
        model.print_model_summary('mixture')

        model.plot_team_effects('mixture')

        # Example prediction with mixture model
        print("\n" + "=" * 50)
        print("MATCH PREDICTION EXAMPLE (MIXTURE MODEL)")
        print("=" * 50)
        prediction_mix = model.predict_match('Internazionale', 'Milan', 'mixture')
        if prediction_mix:
            print(f"Match: {prediction_mix['home_team']} vs {prediction_mix['away_team']}")
            print(f"Expected goals: {prediction_mix['expected_home_goals']:.2f} - {prediction_mix['expected_away_goals']:.2f}")
            print(f"Probabilities: Home {prediction_mix['prob_home_win']:.3f}, Draw {prediction_mix['prob_draw']:.3f}, Away {prediction_mix['prob_away_win']:.3f}")

    except Exception as e:
        print(f"Mixture model failed: {e}")
        print("This is expected as mixture models are complex. The basic model results above are still valid.")