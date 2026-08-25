import numpy as np

from main import create_warehouse
from environment.warehouse_env import WarehouseEnv

# CHANGE THIS IMPORT ONLY if order_generalisation.py
# is stored somewhere else in your project.
from utils.order_generation import generate_order

# ============================================================
# SIMPLE TEST HELPER
# ============================================================


def check(condition, message):
    """
    Raise an error if a check fails.
    Print a success message if it passes.
    """
    assert condition, f"FAILED: {message}"
    print(f"PASS: {message}")


# ============================================================
# CREATE WAREHOUSE
# ============================================================

print("\n========================================")
print("1. CREATING WAREHOUSE")
print("========================================\n")

grid, depot, all_pick_locations, aisle_columns = create_warehouse()

print("Grid shape:", grid.shape)
print("Depot:", depot)
print("Aisle columns:", aisle_columns)
print("Number of pick locations:", len(all_pick_locations))


# ============================================================
# TEST 1: BASIC WAREHOUSE DIMENSIONS
# ============================================================

print("\n========================================")
print("2. CHECKING WAREHOUSE DIMENSIONS")
print("========================================\n")

check(
    grid.shape == (14, 17),
    "Grid shape is 14 x 17"
)

check(
    depot == (0, 0),
    "Depot is at (0, 0)"
)

check(
    aisle_columns == [1, 3, 5, 7, 9, 11, 13, 15],
    "Eight picking aisles are in the correct columns"
)


# ============================================================
# TEST 2: PICK LOCATIONS
# ============================================================

print("\n========================================")
print("3. CHECKING PICK LOCATIONS")
print("========================================\n")

check(
    len(all_pick_locations) == 96,
    "There are exactly 96 pick locations"
)

check(
    len(set(all_pick_locations)) == 96,
    "All 96 pick locations are unique"
)

for row, col in all_pick_locations:

    check(
        1 <= row <= 12,
        f"Pick location {(row, col)} has a valid row"
    )

    check(
        col in aisle_columns,
        f"Pick location {(row, col)} is in a valid picking aisle"
    )

    check(
        grid[row, col] == 0,
        f"Pick location {(row, col)} is walkable"
    )


# ============================================================
# TEST 3: CROSS AISLES
# ============================================================

print("\n========================================")
print("4. CHECKING CROSS AISLES")
print("========================================\n")

check(
    np.all(grid[0, :] == 0),
    "Front cross aisle (row 0) is fully walkable"
)

check(
    np.all(grid[13, :] == 0),
    "Rear cross aisle (row 13) is fully walkable"
)


# ============================================================
# TEST 4: VERTICAL PICKING AISLES
# ============================================================

print("\n========================================")
print("5. CHECKING PICKING AISLES")
print("========================================\n")

for col in aisle_columns:

    check(
        np.all(grid[:, col] == 0),
        f"Picking aisle at column {col} is fully walkable"
    )


# ============================================================
# TEST 5: OBSTACLES
# ============================================================

print("\n========================================")
print("6. CHECKING OBSTACLES")
print("========================================\n")

# Interior cells that are NOT picking aisles
# should be obstacles.

for row in range(1, 13):

    for col in range(17):

        if col not in aisle_columns:

            check(
                grid[row, col] == 1,
                f"Interior cell {(row, col)} is correctly blocked"
            )


# ============================================================
# VISUALISE WAREHOUSE
# ============================================================

print("\n========================================")
print("7. WAREHOUSE VISUALISATION")
print("========================================\n")

print("Legend:")
print("D = depot")
print(". = walkable")
print("# = obstacle")
print("P = pick location\n")

pick_set = set(all_pick_locations)

for row in range(grid.shape[0]):

    line = ""

    for col in range(grid.shape[1]):

        pos = (row, col)

        if pos == depot:
            line += "D"

        elif pos in pick_set:
            line += "P"

        elif grid[row, col] == 0:
            line += "."

        else:
            line += "#"

    print(line)


# ============================================================
# TEST 6: UNIFORM ORDER GENERATION
# ============================================================

print("\n========================================")
print("8. CHECKING UNIFORM ORDER GENERATION")
print("========================================\n")

valid_sizes = [5, 10, 15, 20]

for size in valid_sizes:

    order = generate_order(
        all_pick_locations=all_pick_locations,
        size=size,
        distribution="uniform",
        seed=42,
        aisle_columns=aisle_columns
    )

    check(
        len(order) == size,
        f"Uniform order contains exactly {size} picks"
    )

    check(
        len(set(order)) == size,
        f"Uniform order of size {size} contains no duplicates"
    )

    check(
        set(order).issubset(set(all_pick_locations)),
        f"Uniform order of size {size} only uses valid pick locations"
    )


# ============================================================
# TEST 7: UNIFORM REPRODUCIBILITY
# ============================================================

print("\n========================================")
print("9. CHECKING RANDOM-SEED REPRODUCIBILITY")
print("========================================\n")

order_1 = generate_order(
    all_pick_locations,
    10,
    "uniform",
    seed=123,
    aisle_columns=aisle_columns
)

order_2 = generate_order(
    all_pick_locations,
    10,
    "uniform",
    seed=123,
    aisle_columns=aisle_columns
)

check(
    order_1 == order_2,
    "Same random seed generates the same uniform order"
)


# ============================================================
# TEST 8: CLUSTERED ORDER GENERATION
# ============================================================

print("\n========================================")
print("10. CHECKING CLUSTERED ORDERS")
print("========================================\n")

# All possible contiguous 3-aisle clusters
possible_clusters = []

for start in range(len(aisle_columns) - 2):

    possible_clusters.append(
        set(aisle_columns[start:start + 3])
    )


for size in valid_sizes:

    order = generate_order(
        all_pick_locations=all_pick_locations,
        size=size,
        distribution="clustered",
        seed=100 + size,
        aisle_columns=aisle_columns
    )

    check(
        len(order) == size,
        f"Clustered order contains exactly {size} picks"
    )

    check(
        len(set(order)) == size,
        f"Clustered order of size {size} contains no duplicates"
    )

    check(
        set(order).issubset(set(all_pick_locations)),
        f"Clustered order of size {size} only uses valid pick locations"
    )

    # Find which aisle columns appear in this order.
    used_columns = {
        col
        for row, col in order
    }

    # The order should fit entirely inside at least
    # one contiguous block of three aisles.
    fits_cluster = any(
        used_columns.issubset(cluster)
        for cluster in possible_clusters
    )

    check(
        fits_cluster,
        f"Clustered order of size {size} lies inside a contiguous 3-aisle block"
    )


# ============================================================
# CREATE ENVIRONMENT
# ============================================================

print("\n========================================")
print("11. CREATING WAREHOUSE ENVIRONMENT")
print("========================================\n")

env = WarehouseEnv(
    grid=grid,
    depot=depot,
    all_pick_locations=all_pick_locations,
    max_steps=500
)

print("Environment created successfully.")


# ============================================================
# TEST 9: RESET AND STATE VECTOR
# ============================================================

print("\n========================================")
print("12. CHECKING RESET AND STATE")
print("========================================\n")

test_order = generate_order(
    all_pick_locations,
    size=5,
    distribution="uniform",
    seed=42,
    aisle_columns=aisle_columns
)

state = env.reset(test_order)

check(
    env.agent_pos == depot,
    "Picker starts at depot after reset"
)

check(
    env.travel_distance == 0,
    "Travel distance starts at zero"
)

check(
    env.steps == 0,
    "Step count starts at zero"
)

check(
    len(env.remaining_picks) == 5,
    "Five picks remain after resetting with a 5-item order"
)

check(
    len(env.collected_picks) == 0,
    "No items are collected at reset"
)

check(
    state.shape == (98,),
    "State vector has exactly 98 elements"
)

check(
    state[0] == 0.0,
    "Initial normalised row coordinate is zero"
)

check(
    state[1] == 0.0,
    "Initial normalised column coordinate is zero"
)

check(
    int(np.sum(state[2:])) == 5,
    "Remaining-pick vector contains exactly five active picks"
)


# ============================================================
# TEST 10: INVALID MOVEMENT
# ============================================================

print("\n========================================")
print("13. CHECKING INVALID MOVEMENT")
print("========================================\n")

manual_order = [(1, 1)]

env.reset(manual_order)

# Action 0 = UP
# From depot (0,0), UP attempts (-1,0),
# which is outside the warehouse.

state, reward, terminated, truncated, info = env.step(0)

check(
    env.agent_pos == (0, 0),
    "Invalid movement does not change picker position"
)

check(
    env.travel_distance == 0,
    "Invalid movement does not increase travel distance"
)

check(
    env.steps == 1,
    "Invalid movement still counts as one decision step"
)

check(
    terminated is False,
    "Invalid movement does not terminate the episode"
)


# ============================================================
# TEST 11: VALID MOVEMENT
# ============================================================

print("\n========================================")
print("14. CHECKING VALID MOVEMENT")
print("========================================\n")

env.reset([(1, 1)])

# Action 3 = RIGHT
# (0,0) -> (0,1)

state, reward, terminated, truncated, info = env.step(3)

check(
    env.agent_pos == (0, 1),
    "RIGHT moves picker from depot to (0,1)"
)

check(
    env.travel_distance == 1,
    "Valid movement increases travel distance by one"
)

check(
    env.steps == 1,
    "Valid movement increases step count by one"
)


# ============================================================
# TEST 12: ITEM COLLECTION
# ============================================================

print("\n========================================")
print("15. CHECKING ITEM COLLECTION")
print("========================================\n")

# From (0,1), action 1 = DOWN
# This moves to (1,1), where our required pick is stored.

state, reward, terminated, truncated, info = env.step(1)

check(
    env.agent_pos == (1, 1),
    "Picker reaches the required pick location"
)

check(
    (1, 1) not in env.remaining_picks,
    "Collected item is removed from remaining picks"
)

check(
    (1, 1) in env.collected_picks,
    "Collected item appears in collected picks"
)

check(
    len(env.remaining_picks) == 0,
    "No required picks remain"
)

check(
    terminated is False,
    "Episode does NOT terminate until picker returns to depot"
)


# ============================================================
# TEST 13: RETURN TO DEPOT
# ============================================================

print("\n========================================")
print("16. CHECKING RETURN-TO-DEPOT COMPLETION")
print("========================================\n")

# Move from (1,1) -> (0,1)
state, reward, terminated, truncated, info = env.step(0)

check(
    terminated is False,
    "Episode remains active before reaching depot"
)

# Move from (0,1) -> (0,0)
state, reward, terminated, truncated, info = env.step(2)

check(
    env.agent_pos == depot,
    "Picker returns to depot"
)

check(
    terminated is True,
    "Episode terminates after all picks and return to depot"
)

check(
    env.travel_distance == 4,
    "Manual one-item route has travel distance 4"
)


# ============================================================
# TEST 14: MAXIMUM-STEP TRUNCATION
# ============================================================

print("\n========================================")
print("17. CHECKING MAX-STEP TRUNCATION")
print("========================================\n")

short_env = WarehouseEnv(
    grid=grid,
    depot=depot,
    all_pick_locations=all_pick_locations,
    max_steps=2
)

short_env.reset([(12, 15)])

# Two invalid UP movements from the depot.
short_env.step(0)

state, reward, terminated, truncated, info = short_env.step(0)

check(
    terminated is False,
    "Unfinished episode is not marked as successful"
)

check(
    truncated is True,
    "Episode truncates when max_steps is reached"
)

check(
    short_env.travel_distance == 0,
    "Invalid actions during truncation do not add travel distance"
)


# ============================================================
# FINAL RESULT
# ============================================================

print("\n========================================")
print("ALL CHECKS PASSED")
print("========================================\n")

print("Warehouse geometry: PASS")
print("Pick locations: PASS")
print("Uniform orders: PASS")
print("Clustered orders: PASS")
print("State representation: PASS")
print("Movement: PASS")
print("Item collection: PASS")
print("Return-to-depot termination: PASS")
print("Maximum-step truncation: PASS")

print("\nYour warehouse environment passed the basic verification checks.")
