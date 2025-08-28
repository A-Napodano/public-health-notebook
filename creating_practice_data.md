# Generating Practice Datasets

This guide provides instructions on how to generate datasets on which to practice statistical calculations and tests.

---

## Generating a List of Random Integers

```python
import numpy as np
import pandas as pd

# Set a seed for reproducibility (so you get the same "random" numbers every time)
np.random.seed(42)

# Generate an array of 150 random integers between 0 and 500 (inclusive)
random_numbers = np.random.randint(0, 501, size=150)

# Create a pandas DataFrame
df_integers = pd.DataFrame({'random_value': random_numbers})

# Save the DataFrame to a CSV file
df_integers.to_csv('random_integers.csv', index=False)

print("File 'random_integers.csv' created successfully.")
```

---

## Generating Typical Clinical Trial-Style Dataset

```python
import pandas as pd
import numpy as np

# Set a seed for reproducibility
np.random.seed(123)

# Define the number of subjects
n_subjects = 200

# Create categorical data
treatment = np.random.choice(['Drug_A', 'Drug_B', 'Placebo'], n_subjects, p=[0.3, 0.3, 0.4])
sex = np.random.choice(['Male', 'Female'], n_subjects, p=[0.5, 0.5])
outcome = np.random.choice(['Improved', 'No Change'], n_subjects, p=[0.6, 0.4])

# Create a DataFrame
df_prob = pd.DataFrame({
    'treatment_group': treatment,
    'sex': sex,
    'outcome': outcome
})

# Save to a CSV file
df_prob.to_csv('treatment_data.csv', index=False)

print("File 'treatment_data.csv' created successfully.")
```
