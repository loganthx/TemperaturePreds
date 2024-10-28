import torch, torch.nn as nn


class Block(nn.Module):
	def __init__(self, channels_in, channels_out, num_filters = 3, downsample=True):
		super().__init__()
		self.downsample = downsample        
		if downsample:
			self.conv1 = nn.Conv2d(channels_in, channels_out, num_filters, padding=1)
			self.final = nn.Conv2d(channels_out, channels_out, 4, 2, 1)
		else:
			self.conv1 = nn.Conv2d(2 * channels_in, channels_out, num_filters, padding=1)
			self.final = nn.ConvTranspose2d(channels_out, channels_out, 4, 2, 1)           
		self.bnorm1 = nn.BatchNorm2d(channels_out)
		self.bnorm2 = nn.BatchNorm2d(channels_out)
		self.conv2 = nn.Conv2d(channels_out, channels_out, 3, padding=1)
		self.relu = nn.ReLU()

	def forward(self, x):
		o = self.bnorm1(self.relu(self.conv1(x)))
		o = self.bnorm2(self.relu(self.conv2(o)))
		return self.final(o)



class Net(nn.Module):
	def __init__(self, config, img_channels = 3, sequence_channels = (16, 32), representation_dims=(16, 32)):
		super().__init__()
		c, h, w = config['img_shape']
		sequence_channels_rev = reversed(sequence_channels)		
		self.representation_dims = representation_dims
		self.relu = nn.ReLU()
		self.downsampling = nn.ModuleList([Block(channels_in, channels_out) for channels_in, channels_out in zip(sequence_channels, sequence_channels[1:])])
		self.upsampling = nn.ModuleList([Block(channels_in, channels_out, downsample=False) for channels_in, channels_out in zip(sequence_channels[::-1], sequence_channels[::-1][1:])])
		self.conv1 = nn.Conv2d(img_channels, sequence_channels[0], 3, padding=1)
		self.conv2 = nn.Conv2d(sequence_channels[0], img_channels, 1)

		for n, r_dim in enumerate(representation_dims):
			if n == 0:
				setattr(self, f"finalconv{n+1}", nn.Conv2d(img_channels, representation_dims[n]*img_channels, kernel_size=4, stride=4))
			else:
				setattr(self, f"finalconv{n+1}", nn.Conv2d(representation_dims[n-1]*img_channels, representation_dims[n]*img_channels, kernel_size=4, stride=4))

		self.final_c, self.final_h, self.final_w = self.check_shape(torch.rand(1, c, h, w))
		self.fc1 = nn.Linear(self.final_c * self.final_h * self.final_w, 32)
		self.fc2 = nn.Linear(32, 16)
		self.fc_out = nn.Linear(16, 1)

	@torch.inference_mode()
	def check_shape(self, x):
		residuals = []
		o = self.conv1(x)
		for ds in self.downsampling:
			o = ds(o)
			residuals.append(o)
		for us, res in zip(self.upsampling, reversed(residuals)):
			o = us(torch.cat((o, res), dim=1))    
		o = self.conv2(o)
		for n in range(len(self.representation_dims)):
			o = getattr(self, f"finalconv{n+1}")(o)
		return o.shape[1:]



	def forward(self, x):
		residuals = []
		o = self.conv1(x)
		for ds in self.downsampling:
			o = ds(o)
			residuals.append(o)
		for us, res in zip(self.upsampling, reversed(residuals)):
			o = us(torch.cat((o, res), dim=1))    
		o = self.conv2(o)
		for n in range(len(self.representation_dims)):
			o = getattr(self, f"finalconv{n+1}")(o)
		o = o.view(-1, self.final_c * self.final_h * self.final_w)
		o = self.relu(self.fc1(o))
		o = self.relu(self.fc2(o))
		return self.fc_out(o)

	