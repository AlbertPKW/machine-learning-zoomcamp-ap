import pandas as pd

# 1. Read the CSV
df = pd.read_csv("hr_data.csv")

dept_map = {
    0: "Analytics",
    1: "Finance",
    2: "HR",
    3: "Legal",
    4: "Operations",
    5: "Procurement",
    6: "R&D",
    7: "Sales & Marketing",
    8: "Technology"
}

education_map = {
    0: "Below Secondary",
    1: "Bachelor's",
    2: "Master's & above"
}

gender_map = {
    0: "Female",
    1: "Male"
}

recruitment_channel_map = {
    0: "Other",
    1: "Sourcing",
    2: "Referred"
}

df["department"] = df["department"].map(dept_map)
df["education"] = df["education"].map(education_map)
df["gender"] = df["gender"].map(gender_map)
df["recruitment_channel"] = df["recruitment_channel"].map(recruitment_channel_map)

# 4. Save to a new CSV
df.to_csv("hr_data_mapped.csv", index=False)

