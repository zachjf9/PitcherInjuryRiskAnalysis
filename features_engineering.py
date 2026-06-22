import pandas as pd

def load_statcast(filepath):
    df = pd.read_csv(filepath)
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df

def clean_statcast(df):
    needed_cols = [
        "game_date",
        "pitcher",
        "player_name",
        "pitch_type",
        "release_speed",
        "release_spin_rate",
        "season"]

    df = df[needed_cols].copy()
    df = df.dropna(subset = [
        "game_date",
        "pitcher",
        "pitch_type",
        "release_speed"])

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

    game_pitch["total_pitches"] = (
        game_pitch.groupby(["game_date", "pitcher"])["pitch_count"]
        .transform("sum"))

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

    wide.columns = [
        f"{metric}_{pitch_type}"
        for metric, pitch_type in wide.columns
    ]

    wide = wide.reset_index()
    wide = wide.sort_values(["pitcher", "game_date"])

    return wide

def create_rolling_features(df, window=5):
    df = df.copy()
    df = df.sort_values(["pitcher", "game_date"])

    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    exclude_cols = ["pitcher", "season"]
    rolling_cols = [col for col in numeric_cols if col not in exclude_cols]

    for col in rolling_cols:
        df[f"{col}_{window}g_avg"] = (
            df.groupby("pitcher")[col]
            .rolling(window)
            .mean()
            .reset_index(level=0, drop=True))

    return df

def build_game_level_dataset(filepath, window=5):
    df = load_statcast(filepath)
    df = clean_statcast(df)
    game_pitch = aggregate_pitcher_game(df)
    wide = pivot_pitcher_game_features(game_pitch)
    rolling = create_rolling_features(wide, window=window)

    return rolling