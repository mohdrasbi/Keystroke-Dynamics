from pynput import keyboard
import numpy as np
import pandas as pd
import time
import glob
import os


#########################################


class CollectData:
	def __init__(self, user_id):
		self.timestamps = {}
		self.raw_data = []
		self.user_id = user_id

		with keyboard.Listener(on_press = self.on_press, 
							   on_release = self.on_release) as listener:
			print("Start typing..")
			listener.join()


		self.df = pd.DataFrame(np.array(self.raw_data), columns=['key', 'press_time', 'release_time', 'hold_time'])
		self.df = self.df.sort_values(by=['press_time'])
		
		self.save_file(self.df)


	def on_press(self, key):
		press_time = time.time()
		key_name = self.get_key_name(key)

		print('Key {} pressed.'.format(key_name))

		if key_name != 'Key.esc':
			self.timestamps[key_name] = [press_time]


	def on_release(self, key):
		release_time = time.time()
		key_name = self.get_key_name(key)

		print('Key {} released.'.format(key_name))

		if key_name == 'Key.esc':
			print('Exiting...')
			return False

		try:
			key_info = self.extract_raw_data(key_name, release_time)
		
		except KeyError:
			try:
				key_info = self.extract_raw_data(key_name.upper(), release_time)
			except:
				key_info = None

		if key_info != None:
			self.raw_data.append(key_info)


	def extract_raw_data(self, key_name, release_time):
		stored_vals = self.timestamps.pop(key_name)

		press_time = stored_vals[0]
		hold_time = release_time - press_time
		
		return (key_name, press_time, release_time, hold_time)


	def get_key_name(self, key):
		if isinstance(key, keyboard.KeyCode):
			return key.char
		else:
			return str(key)
	

	def return_df(self):
		return self.df


	def save_file(self, df):
		path = "data/user_{}/raw_data".format(self.user_id)
		if not os.path.exists(path):
			os.makedirs(path)

		file_nums = []
		for file in glob.glob(path + "/*"):
			file_nums.append(int(os.path.basename(file).split('.')[0]))

		index = 0
		if len(file_nums) > 0:
			index = max(file_nums) + 1

		df.to_csv(os.path.join(path, "{}.csv".format(index)), index=False)


#########################################


class ExtractFeatures:
	def __init__(self, df, user_id):
		self.df = df
		self.user_id = user_id
		self.info_dict, self.features = self.initialize()
		self.feat_map = {self.features[i]: i for i in range(len(self.features))}
		self.new_df = self.extract_features()

		self.new_df = self.new_df.fillna(self.new_df.mean())
		self.new_df = self.new_df.fillna(0)
		
		self.save_file(self.new_df)

	def extract_features(self):
		length = self.df.shape[0]
		new_df = pd.DataFrame(columns=self.features)

		start_time = float(self.df.iloc[0]['press_time'])

		bin_size_const = 20
		bin_size = bin_size_const

		feat_values = [[] for i in range(len(self.features))]

		index = 0
		backspaces = 0
		shift = ["Key.shift", "Key.shift_r"]

		info_dict_keys = list(self.info_dict.keys())

		for i in range(1, length):
			prev = self.df.iloc[i-1]
			prev_key = prev['key']
			prev_time = float(prev['press_time'])
			
			curr = self.df.iloc[i]
			curr_key = curr['key']
			curr_time = float(curr['press_time'])
			
			time_diff = curr_time - start_time

			if curr_key == "Key.backspace":
				backspaces += 1
				continue

			if curr_key == None:
				continue

			if curr_key.isupper():
				continue

			if curr_key not in info_dict_keys or prev_key not in info_dict_keys:
				if ((curr_key != "Key.space") and (prev_key != "Key.space") and (curr_key not in shift)) or (curr_key not in info_dict_keys):
					continue
				

			if (curr_key in self.info_dict.keys()) and (prev_key in self.info_dict.keys()) and (self.info_dict[curr_key] == self.info_dict[prev_key]):
				feat_transition = "{}_same".format(self.info_dict[curr_key])
				feat_ht = "{}_ht".format(self.info_dict[curr_key])

				feat_values[self.feat_map[feat_transition]].append(curr_time - prev_time)
				feat_values[self.feat_map[feat_ht]].append(float(curr['hold_time']))
			
			elif curr_key == "Key.space":
				feat_values[self.feat_map["sb_dd"]].append(curr_time - prev_time)
				feat_values[self.feat_map["sb_ht"]].append(float(curr['hold_time']))
			
			elif curr_key in shift:
				next_key = self.df.iloc[i+1]['key'].upper()

				if next_key in info_dict_keys:
					feat_ht = "{}_ht".format(self.info_dict[next_key])
					feat_values[self.feat_map[feat_ht]].append(float(curr['hold_time']))

			elif prev_key == "Key.space":
				feat_transition = 'key_sb'
				feat_ht = "{}_ht".format(self.info_dict[curr_key])
				
				feat_values[self.feat_map[feat_transition]].append(curr_time - prev_time)
				feat_values[self.feat_map[feat_ht]].append(float(curr['hold_time']))
			
			else:		
				temp = ""
				if self.info_dict[prev_key][0] == "l":
					temp = "left"
				elif self.info_dict[prev_key][0] == "r":
					temp = "right"

				feat_transition = "{}_{}".format(self.info_dict[curr_key], temp)
				feat_ht = "{}_ht".format(self.info_dict[curr_key])
				feat_values[self.feat_map[feat_transition]].append(curr_time - prev_time)
				feat_values[self.feat_map[feat_ht]].append(curr_time - prev_time)

			
			if time_diff >= bin_size:
				cpm = i/bin_size*60
				feat_values[self.feat_map['cpm']].append(cpm)
				feat_values[self.feat_map['accuracy']].append(100 - (backspaces/i*100))
				feat_values[self.feat_map['user']].append(self.user_id)

				new_df.loc[index] = list(map(lambda x: np.mean(x) if len(x) > 0 else np.nan, feat_values))

				index += 1
				bin_size += bin_size_const
				feat_values = [[] for i in range(len(self.features))]
				backspaces = 0

		return new_df



	def initialize(self):
		info_dict = dict.fromkeys(["q", "a", "z", "1"], "ll")
		info_dict.update(dict.fromkeys(["w", "s", "x", "2"], "lr"))
		info_dict.update(dict.fromkeys(["e", "d", "c", "3"], "lm"))
		info_dict.update(dict.fromkeys(["r", "t", "f", "g", "v", "b", "4", "5"], 'li'))
		info_dict.update(dict.fromkeys(["!", "Q", "A", "Z"], 'l_cap'))
		info_dict.update(dict.fromkeys(["@", "W", "S", "X"], 'l_cap'))
		info_dict.update(dict.fromkeys(["#", "E", "D", "C"], 'l_cap'))
		info_dict.update(dict.fromkeys(["$", "R", "F", "V", "%", "T", "G", "B"], 'l_cap'))
		info_dict.update(dict.fromkeys(["y", "u", 'h', 'j', 'n', 'm', '6', '7'], 'ri'))
		info_dict.update(dict.fromkeys(['i', 'k', ',', '8'], 'rm'))
		info_dict.update(dict.fromkeys(['.', '9', 'o', 'l'], 'rr'))
		info_dict.update(dict.fromkeys(['0', 'p', ';', '/', '[', "'", "`", ']', '=', '-'], 'rl'))
		info_dict.update(dict.fromkeys(["^", "Y", "H", "N", "&", "U", "J", "M"], 'r_cap'))
		info_dict.update(dict.fromkeys(["*", "I", "K", "<"], 'r_cap'))
		info_dict.update(dict.fromkeys(["(", "O", "L", ">"], 'r_cap'))
		info_dict.update(dict.fromkeys([")", "P", ":", "?", "{", "}", '"', '+', '_'], 'r_cap'))

		features = ["ri_left", "ri_right", "ri_same", "ri_ht",
			"rm_left", "rm_right", "rm_same", "rm_ht", "rr_left", "rr_right", "rr_same", "rr_ht", "r_cap_ht", "rl_left", "rl_right",
			"rl_same", "rl_ht", 'li_left', 'li_right', 'li_same', 'li_ht', 'l_cap_ht', 'lm_left', 'lm_right', 'lm_same', 'lm_ht',
			'lr_left', 'lr_right', 'lr_same', 'lr_ht', 'll_left', 'll_right', 'll_same', 'll_ht', 'cpm',
			'sb_dd', 'sb_ht', 'key_sb', 'accuracy', 'user']

		return info_dict, features


	def return_df(self):
		return self.new_df


	def save_file(self, df):
		path = "data/user_{}/final_data".format(self.user_id)
		if not os.path.exists(path):
			os.makedirs(path)

		file_nums = []
		for file in glob.glob(path + "/*"):
			file_nums.append(int(os.path.basename(file).split('.')[0]))

		index = 0
		if len(file_nums) > 0:
			index = max(file_nums) + 1

		df.to_csv(os.path.join(path, "{}.csv".format(index)), index=False)










	
