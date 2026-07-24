STAGE_ORDER = ["input", "target", "reservoir", "regression", "experiment"]

# MAXSSP/MINSSP (and their *LAST variants) need an "existence" classifier
# at experiment time, on top of their own value regressor: predict_val = 0
# whenever the existence classifier says "no valid answer here". That
# classifier is the regression output of the corresponding DECSSP target
# at the same (U, D, reservoir_method, use_size, reservoir_size) --
# grid.py materializes that extra regression task automatically.
EXISTENCE_DEP = {
    "MAXSSP": "DECSSP",
    "MINSSP": "DECSSP",
    "MAXSSPLAST": "DECSSPLAST",
    "MINSSPLAST": "DECSSPLAST",
}
