import pandas as pd

# Numerology mappings
PYTHAGOREAN = {
    'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8, 'I': 9,
    'J': 1, 'K': 2, 'L': 3, 'M': 4, 'N': 5, 'O': 6, 'P': 7, 'Q': 8, 'R': 9,
    'S': 1, 'T': 2, 'U': 3, 'V': 4, 'W': 5, 'X': 6, 'Y': 7, 'Z': 8
}

def name_to_numerology(name):
    name = name.upper().replace(" ", "")
    total = sum(PYTHAGOREAN.get(char, 0) for char in name if char.isalpha())
    return total % 9 or 9  # Reduce to a single digit (1-9)

def vowels_to_numerology(name):
    vowels = "AEIOU"
    name = name.upper()
    return sum(PYTHAGOREAN.get(char, 0) for char in name if char in vowels)

def consonants_to_numerology(name):
    vowels = "AEIOU"
    name = name.upper()
    return sum(PYTHAGOREAN.get(char, 0) for char in name if char.isalpha() and char not in vowels)

# Load dataset
df = pd.read_csv("train.csv")

# Create numerology-based features
df['Name_Numerology'] = df['Name'].apply(name_to_numerology)
df['Soul_Number'] = df['Name'].apply(vowels_to_numerology)
df['Personality_Number'] = df['Name'].apply(consonants_to_numerology)
df['Name_Length'] = df['Name'].str.replace(" ", "").str.len()

# Analyze numerology and survival
numerology_summary = df.groupby('Name_Numerology')['Survived'].mean()

# Save processed dataset
df.to_csv("titanic_numerology.csv", index=False)

# Print some results
print("Numerology Survival Rates:")
print(numerology_summary)
