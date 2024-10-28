from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
	def __init__(self, samples, transforms=None, target_transforms=None):
		self.samples = samples
		self.transforms = transforms 
		self.target_transforms = target_transforms

	def __len__(self):
		return len(self.samples)

	def __getitem__(self, idx):
		inp = self.samples[idx][0]
		lbl = self.samples[idx][1]
		if self.transforms:
			inp = self.transforms(inp)
		if self.target_transforms:
			lbl = self.target_transforms(lbl)
		return inp, lbl
