import random
import pandas as pd

# Generate data
milliseconds = list(range(1, 30001))
emo = [random.randint(0, 8) if random.random() > 0.4 else 0 for _ in range(30000)]

# Create DataFrame
df = pd.DataFrame({'milisecond': milliseconds, 'EMO': emo})

# Write DataFrame to Excel
df.to_excel('data_for_sma.xlsx', index=False)

