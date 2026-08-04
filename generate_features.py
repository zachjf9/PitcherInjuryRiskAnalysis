from features_engineering import build_game_level_dataset

WINDOW = 5
HORIZON_GAMES = 5
MAX_DAYS_BEFORE_INJURY = 60
IL_FILE = "all_IL_clean.csv"

def generate_dataset(statcast_file):
    #Generate one feature-engineered dataset from a Statcast file
    return build_game_level_dataset(
        statcast_file = statcast_file,
        il_file = IL_FILE,
        window = WINDOW,
        horizon_games = HORIZON_GAMES,
        max_days_before_injury = MAX_DAYS_BEFORE_INJURY)

def main():
    print("Generating training features...")
    train = generate_dataset("statcast_train.csv")

    print("Generating validation features...")
    val = generate_dataset("statcast_val.csv")

    print("Generating test features...")
    test = generate_dataset("statcast_test.csv")

    train.to_csv("pitcher_game_features_train.csv", index=False)
    val.to_csv("pitcher_game_features_val.csv", index=False)
    test.to_csv("pitcher_game_features_test.csv", index=False)

    print("Files generated successfully")

if __name__ == "__main__":
    main()