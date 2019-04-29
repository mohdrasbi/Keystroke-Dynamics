from functions import *


user = str(input("Enter username: "))
user_ids = pd.read_csv("log/user_ids.csv")
user_id = user_ids.shape[0]

user_id_list = list(user_ids[user_ids['user'] == user.lower()]['id'])


while True:
	print("Type:\n1 to collect training data\n2 to validate against existing data")
	user_input = int(input("> "))

	if user_input == 1:
		if len(user_id_list) > 0:
			user_id = user_id_list[0]
		else:
			user_ids.loc[user_id] = [user, user_id]
			user_ids.to_csv("log/user_ids.csv", index=False)
		raw_data = CollectData(user_id, "train_data").return_df()
		final_data = ExtractFeatures(raw_data, user_id, "train_data")
		break

	elif user_input == 2:
		if len(user_id_list) == 0:
			print("The username does not exist")
			continue
		else:
			user_id = user_id_list[0]
			raw_data = CollectData(user_id, "test_data").return_df()
			final_data = ExtractFeatures(raw_data, user_id, "test_data")
			predict = FitAndPredict(user_id)
			match = predict.SVM_classifier()
			print('\n\n\n'+str(match) + '% match'+'\n\n\n')
		break
