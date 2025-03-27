import pandas as pd

path = "Datasets/trades-copy.csv"

# Read the data
trade = pd.read_csv(path, encoding='latin1')

trade.rename(columns={' ': 'Year uncertainty'}, inplace=True)
trade.rename(columns={' .1': 'Number ordered uncertainty'}, inplace=True)
trade.rename(columns={' .2': 'Number delivered uncertainty'}, inplace=True)

# Delete the rows with "?" inside their uncertainty columns
trade = trade[trade['Year uncertainty'] != '?']
trade = trade[trade['Number ordered uncertainty'] != '?']
trade = trade[trade['Number delivered uncertainty'] != '?']

print(trade.columns)
print(trade.shape)

columns_to_delete = [
    'Year uncertainty',
    'Number ordered uncertainty',
    'Number delivered uncertainty',
    'Weapon designation',
    # 'Weapon description',
    'status',
    'Comments',
    'SIPRI TIV per unit',
    'SIPRI TIV for total order',
    'SIPRI TIV of delivered weapons',
    ]

trade.drop(columns=columns_to_delete, inplace=True)

# create new quantity column, which is the minimum value between "Number ordered" and "Number delivered"
trade['Quantity'] = trade[['Number ordered', 'Number delivered']].min(axis=1)
trade.drop(columns=['Number ordered', 'Number delivered'], inplace=True)

# Create new duration column based on Year(s) of delivery format
trade['Duration'] = trade['Year(s) of delivery'].apply(
    lambda x: len(str(x).split('; ')) if isinstance(x, str) and '; ' in x else 1
)
trade.drop(columns=['Year(s) of delivery'], inplace=True)

trade.rename(columns={'Supplier': 'From'}, inplace=True)
trade.rename(columns={'Recipient': 'To'}, inplace=True)

# swap order of "From" and "To" columns
trade = trade[['From', 'To', 'Year of order', 'Duration', 'Quantity', 'Weapon description']]

# save the cleaned data
trade.to_csv('Datasets/trades-clean.csv', index=False)

# print unique values in "Weapons description" column
unique_weapons = trade['Weapon description'].unique()
print(unique_weapons)

# Define Fibonacci weights map - based on your categories and perceived deadliness
weapon_weights = {
    'aircraft engine': 2,  # Component
    'transport helicopter': 3,
    'light helicopter': 2,
    'transport aircraft': 3,
    'combat helicopter': 5,
    'APC': 3,
    'APV': 3,
    'towed gun': 3,
    'anti-tank missile': 5,
    'fighter aircraft': 8,
    'submarine': 8,
    'patrol craft': 2,
    'minesweeper': 2,
    'FAC': 3,
    'AALS': 1, # Ambiguous, assuming light system
    'ship engine': 2, # Component
    'air search radar': 3, # Sensor
    'landing ship': 5,
    'light transport aircraft': 2,
    'frigate': 5,
    'gas turbine': 2, # Component
    'air/sea search radar': 3, # Sensor
    'tank': 8,
    'ASW sonar': 3, # Sensor
    'MCM ship': 3, # Mine Countermeasures
    'armed UAV': 5,
    'helicopter': 3, # General helicopter
    'corvette': 5,
    'MP aircraft radar': 3, # Sensor
    'IFV': 5,
    'trainer aircraft': 2,
    'maritime patrol aircraft': 3,
    'naval SAM system': 5,
    'self-propelled MRL': 5,
    'FGA aircraft': 8, # Fighter Ground Attack
    'bomber aircraft': 8,
    'turbofan': 2, # Component
    'ASW aircraft': 5,
    'destroyer': 8,
    'turbojet': 2, # Component
    'cargo ship': 2, # Support
    'replenishment ship': 2, # Support
    'tank turret': 2, # Component
    'ASW helicopter': 5,
    'fire control radar': 3, # Sensor
    'cruiser': 8,
    'aircraft carrier': 13,
    'light tank': 5,
    'trainer/combat aircraft': 5, # Mid-range
    'tanker/transport aircraft': 3,
    'vehicle engine': 2, # Component
    'naval gun': 3,
    'OPV': 2,
    'tug': 1, # Support
    'light aircraft': 2,
    'SAM': 5,
    'heavy transport aircraft': 5,
    'SAM system': 5,
    'air refuel system': 2, # Support system
    'SIGINT aircraft': 3, # Surveillance
    'training ship': 2, # Support
    'combat heli radar': 3, # Sensor
    'ALV': 1, # Ambiguous, assuming light vehicle
    'self-propelled gun': 5,
    'anti-ship missile': 8,
    'anti-ship/ASW torpedo': 5,
    'ARV': 2, # Armoured Recovery Vehicle
    'ASW mortar': 3,
    'armoured bridgelayer': 2,
    'AEW&C aircraft': 8,
    'tanker': 2, # Support
    'replenishment tanker': 2, # Support
    'tank destroyer': 5,
    'UAV': 3, # General UAV
    'recce satellite': 8, # Strategic recce
    'landing craft': 3,
    'SSM': 8, # Surface to Surface Missile
    'BVRAAM': 8, # Beyond Visual Range Air-to-Air Missile
    'reconnaissance AV': 3, # Reconnaissance Airborne Vehicle (UAV)
    'aircraft recce system': 3, # Sensor system
    'ASW torpedo': 5,
    'ASM': 8, # Air to Surface Missile
    'combat aircraft radar': 3, # Sensor
    'anti-aircraft gun': 3,
    'mobile SAM system': 5,
    'armoured car': 3,
    'SSM launcher': 5, # Launcher, less deadly than missile itself
    'guided bomb': 5,
    'ground surv radar': 3, # Sensor
    'ASW MRL': 3, # Anti-Submarine Warfare Multiple Rocket Launcher
    'transport ship': 2, # Support
    'anti-ship helicopter': 5,
    'portable SAM': 3,
    'AEV': 2, # Armoured Engineering Vehicle
    'support ship': 2, # Support
    'nuclear submarine': 13,
    'SRAAM': 5, # Short Range Air-to-Air Missile
    'reconnaissance aircraft': 3,
    'anti-ship/land-attack missile': 8, # Dual role, high impact
    'trainer/light aircraft': 2,
    'ground attack aircraft': 8,
    'APC turret': 2, # Component
    'AEV/ARV': 2, # Combined support vehicles
    'AGS aircraft': 3, # Airborne Ground Surveillance
    'mortar': 2,
    'minelayer': 3,
    'icebreaker': 1, # Non-combat
    'multi-function radar': 3, # Sensor
    'frigate/landing ship': 5, # Combined
    'APC/APV': 3, # Combined
    'aircraft EO system': 3, # Sensor system
    'artillery locating radar': 3, # Sensor
    'light/trainer aircraft': 2,
    'SIGINT ship': 3, # Surveillance
    'OPV/transport ship': 2, # Combined support/patrol
    'submarine sonar': 3, # Sensor
    'EO search/fire control': 3, # Sensor system
    'height-finding radar': 3, # Sensor
    'self-propelled mortar': 3,
    'AA gun/SAM system': 5, # Combined air defense
    'ground/sea search radar': 3, # Sensor
    'loitering munition': 5, # More advanced UAV munition
    'air search system': 3, # Sensor system
    'coastal defence system': 8, # System, likely missiles and radar
    'minehunter': 3,
    'OPV/support ship': 2, # Combined support/patrol
    'ABM system': 13, # Strategic defense
    'ABM missile': 8, # Strategic defense missile
    'guided glide bomb': 5,
    'SIGINT system': 3, # Sensor system
    'IFV turret': 2, # Component
    'AA gun system': 3, # Air defense system
    'ASW Helicopter': 5, # Assuming same as 'ASW helicopter'
    'SSM/ASM': 8, # Combined missiles
    'AEW radar': 3, # Sensor
    'self-propelled AA gun': 5,
    'maritime patrol UAV': 3,
    'anti-ship missile/SSM': 8, # Combined missiles
    'AEW helicopter': 5,
    'AGS/SIGINT system': 3, # Combined surveillance
    'surveillance satellite': 8, # Strategic surveillance
    'corvette/minesweeper': 5, # Combined
    'support/landing ship': 3, # Combined support/amphibious
    'transport craft': 2, # Support
    'anti-tank missile/ASM': 5, # Combined missiles
    'support craft': 2, # Support
    'SAM system radar': 3, # Sensor
    'gunboat': 3,
    'reconnaissance/SIGINT aircraft': 3, # Combined surveillance
    'AIP engine': 2, # Component
    'sea search radar': 3, # Sensor
    'AGS/MP aircraft radar': 3, # Combined sensor
    'turboprop': 2, # Component
    'guided rocket': 3,
    'survey ship': 1, # Non-combat support
    'cargo craft': 2, # Support
    'OPV/training ship': 2, # Combined support/patrol
    'AA gun': 3,
    'OPV/tug': 1, # Combined support/patrol
    'IFV/APC turret': 2, # Component
    'EO system': 3, # Assuming general sensor system
    'training tank': 3, # For training, less deadly context
    'apc': 3, # Assuming same as 'APC'
    'salvage ship': 1, # Non-combat support
    'guided rocket/SSM': 5, # Combined
    'anti-radar missile': 5,
    'AFSV': 3, # Armoured Fighting Support Vehicle
    'light aircraft/UAV': 2, # Light, dual use
    'BVRAAM/SAM': 8, # Combined missiles
    'airship': 2, # Less common combat role
    'nan': 1, # Assigning lowest weight to NaN, consider handling differently
    'ABM/SAM system': 13, # Combined strategic defense
    'nuclear reactor': 13, # Critical component for nuclear subs/carriers
    'icebreaker/OPV': 2 # Combined non-combat/patrol
}

# Create the 'Weight' column using map
trade['Weight'] = trade['Weapon description'].map(weapon_weights)

# Print value counts of the new 'Weight' column to check distribution
print("\nValue counts for 'Weight' column:")
print(trade['Weight'].value_counts())

# Display some rows with the new 'Weight' column
print("\nSample of DataFrame with 'Weight' column:")
print(trade.head())

# Save the weighted data (optional)
trade.to_csv('Cleaned/trades-weighted.csv', index=False)