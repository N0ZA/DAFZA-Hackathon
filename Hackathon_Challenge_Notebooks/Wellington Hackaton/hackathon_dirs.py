import os


base_dir = "Hackathon_Challenge_Notebooks"


challenge_dirs = [
    "Challenge1_Inventory",
    "Challenge2_Routing",
    "Simplified_Chatbot",
    "Simplified_Recommender",
    "Simplified_Returns"
]

print(f"Creating base directory: {base_dir}")
os.makedirs(base_dir, exist_ok=True)


for challenge in challenge_dirs:
    challenge_path = os.path.join(base_dir, challenge)
    data_path = os.path.join(challenge_path, "data")

    print(f"Creating directory: {challenge_path}")
    os.makedirs(challenge_path, exist_ok=True)

    print(f"Creating directory: {data_path}")
    os.makedirs(data_path, exist_ok=True)

   
    notebook_placeholder = os.path.join(challenge_path, f"{challenge}_DataGen.ipynb")
   


print("\nDirectory structure created successfully.")
print("Now, place the corresponding `.ipynb` file content into each challenge folder.")
