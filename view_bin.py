import pickle
file_location = "C:\Users\Hp\Documents\ML_CRE Project\training_data\reaction_data.pkl"
with open(file_location,'rb') as f:
    s = pickle.load(f)

print(s)