sub_dataset="001"
window_size=30
max_life=125#125
scaler="mm" #"mm" max-min ;"ss"standard
scaler_range=(-1,1)
shuffle=True
batch_size=512#适用于你的内存大小
num_iter = 1000
alpha_grid=0.4
alpha_low=0.6
alpha_high=0.9
_COLORS=["green", "teal","pink"]
read_path = r"..\C-MAPSS-Data/"
transform_ln = True