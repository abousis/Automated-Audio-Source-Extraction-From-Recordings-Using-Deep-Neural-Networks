import numpy as np

# Data generator
def generator(X_data, y_data, batch_size):

  samples_per_epoch = X_data.shape[0]
  number_of_batches = samples_per_epoch/batch_size
  counter=0

  while 1:

    X_batch = np.array(X_data[batch_size*counter:batch_size*(counter+1)])
    y_batch = np.array(y_data[batch_size*counter:batch_size*(counter+1)])
    counter += 1
    yield X_batch, y_batch

    #restart counter to yeild data in the next epoch as well
    if counter >= number_of_batches:
        counter = 0