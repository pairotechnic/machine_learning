import pandas as pd

# data = {
#     "calories" : [420, 380, 390],
#     "duration": [50, 40, 45]
# }

# # load  data into a DataFrame object:
# df = pd.DataFrame(data)

# print(df)

student_data = [
    [1, 15],
    [2, 11],
    [3, 11],
    [4, 20]
]

df = pd.DataFrame(student_data, columns=["student_id", "age"])

print(df.shape)

