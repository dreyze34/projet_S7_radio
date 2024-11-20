from loadData import ThoraxDataLoader

loader = ThoraxDataLoader("./Data/Data_Medical/MCsimulations")
data = loader.load_all_samples()

print(data.shape)
