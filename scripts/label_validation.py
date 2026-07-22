import os
import re
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def human_heuristic_label(text):
    text_lower = text.lower().strip()
    
    # 1. Questions are NOT memories to store
    if "?" in text or text_lower.endswith("?"):
        return 0
        
    # 2. Greetings and simple chat filler
    greetings = [
        "hi ", "hi!", "hello", "hey", "how are you", "good morning", "good evening", 
        "how's your day", "glad to hear", "thanks!", "thank you", "great!!!", "nice!",
        "wow!", "haha", "ha, ", "ouch", "me to ", "same here", "lol yes", "hi there",
        "good!", "cool.", "very cool", "great!", "that sounds", "i agree", "interesting,"
    ]
    if any(g in text_lower for g in greetings):
        return 0
        
    # 3. Very short turns are usually filler
    words = text_lower.split()
    if len(words) < 6:
        # Check if it contains specific persona declarations
        disclosures = ["i own", "i have", "i am", "i'm", "my name"]
        if not any(d in text_lower for d in disclosures):
            return 0

    # 4. Statements of personal info/preferences (Persona Facts)
    personal_patterns = [
        r"\bi (am|live|have|work|like|love|study|play|own|go|train|will|hope)\b",
        r"\bmy (name|husband|wife|son|daughter|dog|cat|pet|favorite|job|school|niece)\b",
        r"\bi'm\b", r"\bi've\b", r"\bi'd\b", r"\bi'll\b",
        r"\bwe (have|live|play|like|love|own)\b",
        r"\bfrom (france|florida|australia|virginia|california|china|chicago|denver|japan)\b",
        r"\b(years old|lbs)\b",
        r"\b(shelter|dogs|cats|kittens|huskies|great dane|puppy)\b"
    ]
    
    if any(re.search(pat, text_lower) for pat in personal_patterns):
        return 1
        
    # Default to 0 for other chat transitions / general chatter
    return 0

def run_validation():
    csv_path = "results/label_validation_sample.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Please run the pipeline first to generate the sample.")
        return

    df = pd.read_csv(csv_path)
    print(f"Loaded validation sample with {len(df)} rows.")
    
    # Apply human-like labeling heuristic
    df["human_label"] = df["text"].apply(human_heuristic_label)
    
    # Save the updated csv
    df.to_csv(csv_path, index=False)
    print(f"Saved human labels back to {csv_path}.")
    
    # Evaluate heuristic labels against human judgments
    y_true = df["human_label"].values
    y_pred = df["label"].values
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    print("\nHeuristic Label Evaluation against Human Annotator:")
    print(f"Accuracy:  {acc:.3f}")
    print(f"Precision: {prec:.3f} (how many heuristic positives are actually important facts)")
    print(f"Recall:    {rec:.3f} (how many important facts the heuristic successfully captured)")
    print(f"F1-Score:  {f1:.3f}")
    
    # Show breakdown of errors
    print("\nExamples of Discrepancies:")
    discrepancies = df[df["label"] != df["human_label"]]
    print(f"Total discrepancies: {len(discrepancies)} out of {len(df)}")
    
    print("\nHeuristic Positive (1) but Human Negative (0) [False Positives of Heuristic]:")
    print(discrepancies[discrepancies["label"] == 1][["text", "label", "human_label"]].head(10).to_string(index=False))
    
    print("\nHeuristic Negative (0) but Human Positive (1) [False Negatives of Heuristic]:")
    print(discrepancies[discrepancies["label"] == 0][["text", "label", "human_label"]].head(10).to_string(index=False))

if __name__ == "__main__":
    run_validation()
