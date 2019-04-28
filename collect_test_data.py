from functions import *


user = str(input("Enter username: "))
user_ids = pd.read_csv("log/user_ids.csv")
user_id = user_ids.shape[0]

user_id_list = list(user_ids[user_ids['user'] == user]['id'])
if len(user_id_list) > 0:
	user_id = user_id_list[0]
else:
	user_ids.loc[user_id] = [user, user_id]
	user_ids.to_csv("log/user_ids.csv", index=False)


raw_data = CollectData(user_id, "test_data").return_df()

final_data = ExtractFeatures(raw_data, user_id, "test_data")