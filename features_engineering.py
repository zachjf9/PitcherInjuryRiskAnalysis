import pandas as pd
import numpy as np

def normalize_name(name):
    #Match Statcast names to IL names
    if pd.isna(name):
        return name

    name = str(name).strip()

    if "," in name:
        last, first = name.split(",", 1)
        return f"{first.strip()} {last.strip()}"

    return name

def load_statcast(filepath):
    df = pd.read_csv(filepath)
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df

def clean_statcast(df):
    #Feature engineering columns
    needed_cols = [
        "game_date",
        "pitcher",
        "player_name",
        "pitch_type",
        "release_speed",
        "release_spin_rate",
        "season"]

    df = df[needed_cols].copy()
    #Removes incomplete rows
    df = df.dropna(subset = [
        "game_date",
        "pitcher",
        "pitch_type",
        "release_speed"])

    return df

def create_injury_labels(df, il_df, horizon_games = 5, max_days_before_injury = 60):
    df = df.copy()
    #Initialize all observations as non-injured
    df["injury_label"] = 0

    df["game_date"] = pd.to_datetime(df["game_date"])
    il_df = il_df.copy()
    il_df["injury_date"] = pd.to_datetime(il_df["injury_date"], errors = "coerce")

    #Normalize names in both datasets
    df["name_clean"] = (df["player_name"].apply(normalize_name).str.lower())
    il_df["name_clean"] = (il_df["player_name"].apply(normalize_name).str.lower())

    #Remove unusable injury records
    il_df = il_df.dropna(subset = ["name_clean", "injury_date"])

    #Loop through every injury record
    for _, injury in il_df.iterrows():
        pitcher_name = injury["name_clean"]
        injury_date = injury["injury_date"]

        pitcher_games = (df[df["name_clean"] == pitcher_name].sort_values("game_date"))

        eligible_games = pitcher_games[(pitcher_games["game_date"] < injury_date) &
                                       (pitcher_games["game_date"] >= injury_date - pd.Timedelta(days = max_days_before_injury))]

        #Label last x games before injury
        games_to_label = (eligible_games.tail(horizon_games).index)
        df.loc[games_to_label, "injury_label"] = 1

    #Remove temporary text column
    df = df.drop(columns = ["name_clean"])

    return df

def aggregate_pitcher_game(df):
    game_pitch = (
        df.groupby([
            "game_date",
            "pitcher",
            "player_name",
            "pitch_type",
            "season"])
        .agg(pitch_count = ("pitch_type", "size"), avg_velocity = ("release_speed", "mean"), avg_spin_rate = ("release_spin_rate", "mean"))
        .reset_index())

    #Calculate total pitches thrown in each game
    game_pitch["total_pitches"] = (game_pitch.groupby(["game_date", "pitcher"])["pitch_count"].transform("sum"))

    #Calculate usage % for each pitch type
    game_pitch["pitch_usage_pct"] = (game_pitch["pitch_count"] / game_pitch["total_pitches"])

    return game_pitch

def pivot_pitcher_game_features(game_pitch):
    wide = game_pitch.pivot_table(
        index = [
            "game_date",
            "pitcher",
            "player_name",
            "season",
            "total_pitches"],

        columns = "pitch_type",
        values = [
            "pitch_count",
            "avg_velocity",
            "avg_spin_rate",
            "pitch_usage_pct"])

    wide.columns = [
        f"{metric}_{pitch_type}"
        for metric, pitch_type in wide.columns]
    
    #Restore index columns
    wide = wide.reset_index()

    '''print("Columns after pivot:")
    print(wide.columns.tolist())'''

    #Chronologically by pitcher
    wide = wide.sort_values(["pitcher", "game_date"])

    return wide

#Calculate the linear slope of features across a rolling window
def calculate_slope(values):
    values = np.asarray(values, dtype = float)

    #Keep only non-missing observations
    valid_mask = ~np.isnan(values)
    valid_values = values[valid_mask]

    #Slope requires at least two valid observations
    if len(valid_values) < 2:
        return np.nan

    #Preserve the relative positions of valid games
    game_positions = np.arange(len(values))[valid_mask]

    #Fit a straight line and return its slope
    slope = np.polyfit(game_positions, valid_values,1)[0]

    return slope

#Create game-to-game differences and rolling trend slopes
def create_trend_features(df, window = 5):
    df = df.copy()

    #Ensure dates are datetime objects
    df["game_date"] = pd.to_datetime(df["game_date"])

    #Every pitcher's games are in chronological order
    df = df.sort_values(["pitcher", "game_date"]).reset_index(drop = True)

    #Reset short-term calculations at the start of each season.
    if "season" in df.columns:
        group_cols = ["pitcher", "season"]
    else:
        group_cols = ["pitcher"]

    #Identify the original pitch-type velocity columns
    velocity_cols = [
        col for col in df.columns
        if col.startswith("avg_velocity_")
        and "_previous_game_diff" not in col
        and "_acceleration" not in col
        and "g_slope" not in col
        and "g_std" not in col
        and "g_avg" not in col]

    #Identify the original pitch-type spin-rate columns
    spin_cols = [
        col for col in df.columns
        if col.startswith("avg_spin_rate_")
        and "_previous_game_diff" not in col
        and "_acceleration" not in col
        and "g_slope" not in col
        and "g_std" not in col
        and "g_avg" not in col]

    #-----------------------------------------------
    #Rest features
    #-----------------------------------------------
    #Number of calendar days since the pitcher's previous appearance
    df["days_since_previous_game"] = (df.groupby(group_cols)["game_date"].diff().dt.days)

    #Flag the first recorded appearance within each season
    df["has_previous_game"] = (df["days_since_previous_game"].notna().astype(int))

    #Actual full rest days between appearances
    df["rest_days"] = (df["days_since_previous_game"] - 1).clip(lower = 0)

    #Average rest across the rolling window
    df[f"rest_days_{window}g_avg"] = (
        df.groupby(group_cols)["rest_days"]
        .rolling(window = window, min_periods = 2)
        .mean()
        .reset_index(level = list(range(len(group_cols))), drop = True))

    #Variability in rest across the rolling window
    df[f"rest_days_{window}g_std"] = (
        df.groupby(group_cols)["rest_days"]
        .rolling(window = window, min_periods = 2)
        .std()
        .reset_index(level = list(range(len(group_cols))), drop = True))

    #-----------------------------------------------
    #Velocity Features
    #-----------------------------------------------
    for col in velocity_cols:
        difference_col = f"{col}_previous_game_diff"

        #Change from the immediately previous appearance
        df[difference_col] = (
            df.groupby(group_cols)[col].diff(periods = 1))

        #Change in the game-to-game difference
        df[f"{col}_acceleration"] = (
            df.groupby(group_cols)[difference_col].diff(periods = 1))

        #Overall velocity direction across the rolling window
        df[f"{col}_{window}g_slope"] = (
            df.groupby(group_cols)[col]
            .rolling(window = window, min_periods = 2)
            .apply(calculate_slope, raw = True)
            .reset_index(level = list(range(len(group_cols))), drop = True))

        #Velocity consistency across the rolling window
        df[f"{col}_{window}g_std"] = (
            df.groupby(group_cols)[col]
            .rolling(window = window, min_periods = 2)
            .std()
            .reset_index(level = list(range(len(group_cols))), drop = True))

    #-----------------------------------------------
    #Spin-rate
    #-----------------------------------------------
    for col in spin_cols:
        difference_col = f"{col}_previous_game_diff"

        #Change from the immediately previous appearance
        df[difference_col] = (
            df.groupby(group_cols)[col].diff(periods = 1))

        #Change in the game-to-game spin-rate difference
        df[f"{col}_acceleration"] = (
            df.groupby(group_cols)[difference_col].diff(periods = 1))

        #Overall spin-rate direction across the rolling window
        df[f"{col}_{window}g_slope"] = (
            df.groupby(group_cols)[col]
            .rolling(window = window, min_periods = 2)
            .apply(calculate_slope, raw = True)
            .reset_index(level = list(range(len(group_cols))), drop = True))

        #Spin-rate consistency across the rolling window
        df[f"{col}_{window}g_std"] = (
            df.groupby(group_cols)[col]
            .rolling(window = window, min_periods = 2)
            .std()
            .reset_index(level = list(range(len(group_cols))), drop = True))

    #-----------------------------------------------
    #Pitch Count
    #-----------------------------------------------
    if "total_pitches" in df.columns:
        pitch_difference_col = "total_pitches_previous_game_diff"

        #Pitch-count change from the previous appearance
        df[pitch_difference_col] = (
            df.groupby(group_cols)["total_pitches"].diff(periods = 1))

        #Change in the pitch-count difference
        df["total_pitches_acceleration"] = (
            df.groupby(group_cols)[pitch_difference_col].diff(periods = 1))

        #Overall pitch-count trend
        df[f"total_pitches_{window}g_slope"] = (
            df.groupby(group_cols)["total_pitches"]
            .rolling(window = window, min_periods = 2)
            .apply(calculate_slope, raw = True)
            .reset_index(level = list(range(len(group_cols))), drop = True))

        #Pitch-count variability
        df[f"total_pitches_{window}g_std"] = (
            df.groupby(group_cols)["total_pitches"]
            .rolling(window = window, min_periods = 2)
            .std()
            .reset_index(level = list(range(len(group_cols))), drop = True))

    #-----------------------------------------------
    #Handle missing Values
    #-----------------------------------------------
    generated_suffixes = (
        "_previous_game_diff",
        "_acceleration",
        "g_slope",
        "g_std",
        "g_avg")

    generated_cols = [
        col for col in df.columns
        if col.endswith(generated_suffixes)
           or col in ["days_since_previous_game", "rest_days"]]

    df[generated_cols] = df[generated_cols].fillna(0)

    return df

def create_rolling_features(df, window = 5):
    df = df.copy()
    #Games in chronological order
    df = df.sort_values(["pitcher", "season", "game_date"]).reset_index(drop = True)


    group_cols = ["pitcher", "season"]
    rolling_cols = [col for col in df.columns
                    if (col == "total_pitches"
                        or col.startswith("pitch_count_")
                        or col.startswith("avg_velocity_")
                        or col.startswith("avg_spin_rate_")
                        or col.startswith("pitch_usage_pct_"))
                    and "_previous_game_diff" not in col
                    and "_acceleration" not in col
                    and "g_slope" not in col
                    and "g_std" not in col
                    and "g_avg" not in col]

    #Rolling averages for each feature
    for col in rolling_cols:
        df[f"{col}_{window}g_avg"] = (
            df.groupby(group_cols)[col].rolling(window = window, min_periods = 2).mean().reset_index(level = list(range(len(group_cols))), drop = True))

    return df

def build_game_level_dataset(statcast_file, il_file = None, window = 5, horizon_games = 5, max_days_before_injury = 60):
    #Load and clean data
    df = load_statcast(statcast_file)
    df = clean_statcast(df)

    game_pitch = aggregate_pitcher_game(df)
    wide = pivot_pitcher_game_features(game_pitch)

    #Add game-to-game changes and rolling slopes
    trends = create_trend_features(wide, window = window)

    #Rolling averages
    rolling = create_rolling_features(trends, window = window)

    #Add injury labels
    if il_file is not None:
        il_df = pd.read_csv(il_file)

        rolling = create_injury_labels(rolling, il_df, horizon_games = horizon_games, max_days_before_injury = max_days_before_injury)

    return rolling