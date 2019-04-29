from functions import *
from sklearn import svm

user = str(input("Enter username: "))
user_ids = pd.read_csv("log/user_ids.csv")
user_id = user_ids.shape[0]

user_id_list = list(user_ids[user_ids['user'] == user.lower()]['id'])


while(True):
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
			print(user_id)
			raw_data = CollectData(user_id, "test_data").return_df()
			final_data = ExtractFeatures(raw_data, user_id, "test_data")
			train_path = "train_data/user_" + str(user_id) + "/final_data/"
			test_path = "test_data/user_"+str(user_id)+"/final_data/"
			train_file = sorted(os.listdir(train_path))[-1]
			print(train_path+train_file)
			test_file = sorted(os.listdir(test_path))[-1]
			df_train = pd.read_csv(train_path+train_file)
			df_train = df_train.drop(['user'], axis=1)
			df_test = pd.read_csv(test_path+test_file)
			print(test_path+test_file)
			df_test = df_test.drop(['user'], axis=1)
			x_train = df_train.values
			x_test = df_test.values
			clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.001)
			clf.fit(x_train)
			y_test_pred = clf.predict(x_test)
			unique, counts = np.unique(y_test_pred, return_counts=True)
			counts_dict = dict(zip(unique, counts))

			if 1 in counts_dict:
				match = (counts_dict[1] / len(y_test_pred)) * 100
				print(str(match) + '% match')
			else:
				match = 0
				print('0% match')

		break
