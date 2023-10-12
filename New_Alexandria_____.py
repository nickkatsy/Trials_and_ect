import random
import matplotlib.pyplot as plt
from collections import Counter

# Number of times you played this stupid level withot reverting
num_trials = 1000

# Lists to store results for each trial
results = []

# Sequence to track
sequence_to_track = ["Hospital", "Whiskey 39er", "Vyrant", "Foxtrot", "Sinoviet"]

# Initialize counter for the sequence
sequence_count = 0

for _ in range(num_trials):
    objectives = []

    # Main objective "Hospital" is always first (besides legendary or something)
    objectives.append('Hospital')

    # Side objectives grouped by categories
    side_objectives = {
        "Helping Marines": ['Vs. Brutes', 'Foxtrot', 'Vs. Jackals'],
        "3 Enemies": ['Whiskey 39er', 'Banshees', 'Snipers'],
        "Covenant Causing Trouble": ['Elites and Engineers', 'Shades', 'Jammer'],
        "Buck": ['Buck', 'Buck2', 'Buck3']
    }

    # Select one side objective from the 12 available
    selected_side_objective = random.choice([
        objective for objectives in side_objectives.values() for objective in objectives
    ])
    objectives.append(selected_side_objective)

    # main objective based on a coin flip(50/50 dudde)
    main_objective = random.choice(["Sinoviet", "Vyrant"])
    objectives.append(main_objective)

    # Select one more side objective from the remaining 9
    remaining_side_objectives = [
        objective for objectives in side_objectives.values() for objective in objectives
        if objective != selected_side_objective
    ]
    objectives.append(random.choice(remaining_side_objectives))

    # Determine the last main objective based on the previous main objective
    last_main_objective = "Sinoviet" if main_objective == "Vyrant" else "Vyrant"
    objectives.append(last_main_objective)

    # Append the generated objectives to the results
    results.append(objectives)

    # Check for the sequence in the objectives
    if objectives == sequence_to_track:
        sequence_count += 1

# Count the frequency of objectives for all trials
all_objectives = [objective for objectives in results for objective in objectives]
most_common_objectives = Counter(all_objectives).most_common()

# Display the most frequent objectives
print("Most Frequent Objectives:")
for objective, frequency in most_common_objectives:
    print(f"{objective}: {frequency} times")

# Display the count of the sequence
print(f"\nCount of Sequence '{sequence_to_track}': {sequence_count} times")