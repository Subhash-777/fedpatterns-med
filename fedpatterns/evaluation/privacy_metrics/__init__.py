"""
Privacy evaluation metrics module.

NOTE: When logging Differential Privacy (DP) metrics to MLflow or CSV, we use 
epsilon = -1.0 as a sentinel value to indicate "no DP applied". This is because
true infinity (float("inf")) cannot be cleanly stored in MLflow numeric fields.
Ensure you interpret epsilon=-1.0 correctly when building privacy-accuracy 
tradeoff plots.
"""