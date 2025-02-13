import pandas as pd
import matplotlib.pyplot as plt

# Load the datasets
users_df = pd.read_csv("users.csv")
items_df = pd.read_csv("items.csv")
interactions_df = pd.read_csv("interactions.csv")

# Merge datasets
merged_df = interactions_df.merge(users_df, on="user_id").merge(items_df, on="item_id")

# Bubble Chart: Interactions by category and average price
bubble_data = merged_df.groupby("category").agg({"interaction_type": "count", "price": "mean"}).reset_index()
plt.figure(figsize=(10, 6))
plt.scatter(
    bubble_data["price"],  # X-axis: Average price
    bubble_data["interaction_type"],  # Y-axis: Interaction counts
    s=bubble_data["interaction_type"] * 10,  # Bubble size proportional to interactions
    alpha=0.6, color="blue"
)
plt.title("Bubble Chart: Interactions by Category and Average Price")
plt.xlabel("Average Price")
plt.ylabel("Number of Interactions")
plt.grid(True)
plt.show()

# Pie Chart: Interaction type distribution.
interaction_counts = merged_df["interaction_type"].value_counts()
interaction_labels = interaction_counts.index.tolist()
plt.figure(figsize=(8, 8))
plt.pie(
    interaction_counts,
    labels=interaction_labels,
    autopct="%1.1f%%",
    startangle=140,
    colors=["lightblue", "lightgreen", "orange"]
)
plt.title("Pie Chart: Interaction Type Distribution")
plt.show()

# Box-Whisker Chart: Age distribution of users.
plt.figure(figsize=(8, 6))
plt.boxplot(users_df["age"], vert=False, patch_artist=True, boxprops=dict(facecolor="lightblue"))
plt.title("Box-Whisker Chart: Age Distribution of Users")
plt.xlabel("Age")
plt.show()

# Scatter Plot: User age vs. average item price
scatter_data = merged_df.groupby("user_id").agg({"age": "first", "price": "mean"}).reset_index()
plt.figure(figsize=(10, 6))
plt.scatter(
    scatter_data["age"],
    scatter_data["price"],
    alpha=0.6, color="green"
)
plt.title("Scatter Plot: User Age vs. Average Item Price")
plt.xlabel("User Age")
plt.ylabel("Average Item Price")
plt.grid(True)
plt.show()
