...
batches_per_epoch = floor(dataset_size / batch_size)
total_iterations = batches_per_epoch * total_epochs
# gan training algorithm
def train_gan(dataset, n_epochs, n_batch):
	# calculate the number of batches per epoch
	batches_per_epoch = int(len(dataset) / n_batch)
	# calculate the number of training iterations
	n_steps = batches_per_epoch * n_epochs
	# gan training algorithm
	for i in range(n_steps):
		# update the discriminator model
		# ...
		# update the generator model
		# ...
    ...
# generate points in the latent space
z = randn(latent_dim * n_batch)
# reshape into a batch of inputs for the network
z = x_input.reshape(n_batch, latent_dim)
# generate fake images
fake = generator.predict(z)
...
# select a batch of random real images
ix = randint(0, len(dataset), n_batch)
# retrieve real images
real = dataset[ix]
# gan training algorithm
def train_gan(generator, discriminator, dataset, latent_dim, n_epochs, n_batch):
	# calculate the number of batches per epoch
	batches_per_epoch = int(len(dataset) / n_batch)
	# calculate the number of training iterations
	n_steps = batches_per_epoch * n_epochs
	# gan training algorithm
	for i in range(n_steps):
		# generate points in the latent space
		z = randn(latent_dim * n_batch)
		# reshape into a batch of inputs for the network
		z = z.reshape(n_batch, latent_dim)
		# generate fake images
		fake = generator.predict(z)
		# select a batch of random real images
		ix = randint(0, len(dataset), n_batch)
		# retrieve real images
		real = dataset[ix]
		# update weights of the discriminator model
		# ...

		# update the generator model
		# ...
    ...
# generate points in the latent space
z = randn(latent_dim * n_batch)
# reshape into a batch of inputs for the network
z = z.reshape(n_batch, latent_dim)
# generate fake images
fake = generator.predict(z)
# classify as real or fake
result = discriminator.predict(fake)
# gan training algorithm
def train_gan(generator, discriminator, dataset, latent_dim, n_epochs, n_batch):
	# calculate the number of batches per epoch
	batches_per_epoch = int(len(dataset) / n_batch)
	# calculate the number of training iterations
	n_steps = batches_per_epoch * n_epochs
	# gan training algorithm
	for i in range(n_steps):
		# generate points in the latent space
		z = randn(latent_dim * n_batch)
		# reshape into a batch of inputs for the network
		z = z.reshape(n_batch, latent_dim)
		# generate fake images
		fake = generator.predict(z)
		# select a batch of random real images
		ix = randint(0, len(dataset), n_batch)
		# retrieve real images
		real = dataset[ix]
		# update weights of the discriminator model
		# ...
		# generate points in the latent space
		z = randn(latent_dim * n_batch)
		# reshape into a batch of inputs for the network
		z = z.reshape(n_batch, latent_dim)
		# generate fake images
		fake = generator.predict(z)
		# classify as real or fake
		result = discriminator.predict(fake)
		# update weights of the generator model
		# ...
