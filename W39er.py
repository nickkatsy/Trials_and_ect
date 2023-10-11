import random

def New_Alexandria():
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
    
    # Determine the main objective based on a coin flip
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
    
    return objectives

# Simulate the "New Alexandria" objectives
mission_sequence = New_Alexandria()

# Print the generated sequence
print("Mission Objectives for New Alexandria:")
for i, objective in enumerate(mission_sequence, start=1):
    print(f"Objective {i}: {objective}")
