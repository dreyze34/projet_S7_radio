from loadData import ThoraxDataLoader

loader = ThoraxDataLoader("./Data/Data_Medical/MCsimulations")
x_dataset = loader.load_all_samples("LS")
y_dataset = loader.load_all_samples("HS")

train_lenght = int(0.80*x_dataset.size()[0])

x_train = x_dataset[:train_lenght,:,:]
y_train = y_dataset[:train_lenght,:,:]

x_test = x_dataset[train_lenght:,:,:]
y_test = y_dataset[train_lenght:,:,:]

print(x_train.size(), x_test.size())