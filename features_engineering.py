import pandas as pd

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

def create_injury_labels(df, il_df, horizon_games=5):
    df = df.copy()
    #Initialize all observations as non-injured
    df["injury_label"] = 0

    df["game_date"] = pd.to_datetime(df["game_date"])
    il_df["injury_date"] = pd.to_datetime(il_df["injury_date"])

    #Loop through every injury record
    for _, injury in il_df.iterrows():
        pitcher_name = injury["player_name"]
        injury_date = injury["injury_date"]

        pitcher_games = (
            df[df["player_name"] == pitcher_name]
            .sort_values("game_date"))

        pre_injury_games = pitcher_games[
            pitcher_games["game_date"] < injury_date]

        #Label last x games before injury
        games_to_label = pre_injury_games.tail(horizon_games).index
        df.loc[games_to_label, "injury_label"] = 1

    return df

def aggregate_pitcher_game(df):
    game_pitch = (
        df.groupby([
            "game_date",
            "pitcher",
            "player_name",
            "pitch_type",
            "season"])

        .agg(
            pitch_count=("pitch_type", "size"),
            avg_velocity=("release_speed", "mean"),
            avg_spin_rate=("release_spin_rate", "mean"))
        .reset_index())

    #Calculate total pitches thrown in each game
    game_pitch["total_pitches"] = (
        game_pitch.groupby(["game_date", "pitcher"])["pitch_count"]
        .transform("sum"))

    #Calculate usage % for each pitch type
    game_pitch["pitch_usage_pct"] = (
        game_pitch["pitch_count"] / game_pitch["total_pitches"])

    return game_pitch

def pivot_pitcher_game_features(game_pitch):
    wide = game_pitch.pivot_table(
        index=[
            "game_date",
            "pitcher",
            "player_name",
            "season",
            "total_pitches"],

        columns="pitch_type",
        values=[
            "pitch_count",
            "avg_velocity",
            "avg_spin_rate",
            "pitch_usage_pct"])

    wide = wide.reset_index()
    #Flatten multi-level column names
    wide.columns = [
        f"{metric}_{pitch_type}"
        for metric, pitch_type in wide.columns
    ]
    #Restore index columns
    wide = wide.reset_index()
    #Chronologically by pitcher
    wide = wide.sort_values(["pitcher", "game_date"])

    return wide

def create_rolling_features(df, window=5):
    df = df.copy()
    #Games in chronological order
    df = df.sort_values(["pitcher", "game_date"])

    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    exclude_cols = ["pitcher", "season"]
    rolling_cols = [col for col in numeric_cols if col not in exclude_cols]

    #Rolling averages for each feature
    for col in rolling_cols:
        df[f"{col}_{window}g_avg"] = (
            df.groupby("pitcher")[col]
            .rolling(window)
            .mean()
            .reset_index(level=0, drop=True))

    return df

def build_game_level_dataset(statcast_file, il_file=None, window=5, horizon_games=5):
    #Load and clean data
    df = load_statcast(statcast_file)
    df = clean_statcast(df)

    game_pitch = aggregate_pitcher_game(df)
    wide = pivot_pitcher_game_features(game_pitch)

    #Rolling averages
    rolling = create_rolling_features(
        wide,
        window=window)

    #Add injury labels
    if il_file is not None:
        il_df = pd.read_csv(il_file)

        rolling = create_injury_labels(
            rolling,
            il_df,
            horizon_games=horizon_games)

    return rolling