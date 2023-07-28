import pickle
from UNet_StableDiffusion import *

## TOUT ##
with open('data_label/xx_label_melange.pkl','rb') as f:
    X_train = pickle.load(f)

###############
# PARAMETERS #
##############

nb_shapes = len(X_train)
timesteps = 450 # number of steps
learning_rate = 1e-4
iterations = 750000
epochs = nb_shapes * 10
batch_size = 200 #125
num_class = 22 #pomme, chauves souris, marteau, cloche, tortue
dim1 = 32
dim2 = 32


#########
# MODEL #
#########

# create our unet model
unet = Unet_conditional(
    num_classes=num_class,
    in_res=32,
    channels=1)

test_images = np.ones([batch_size, dim1, dim2, 1])
test_timestamps = generate_timestamp(0, 1)
test_class = np.ones(batch_size)
k = unet(test_images, test_timestamps, test_class)
# opt = keras.optimizers.Adam(learning_rate=1e-4)

# create our optimizer, we will use adam with a Learning rate
boundaries = [400, 800]
values = [0.0001, 0.00001, 0.0001]

learning_rate_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)
opt = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn)

# unet.load_weights('save/SD/2shapes/1500weights')

def loss_fn(real, generated):
    mse = tf.keras.losses.MeanSquaredError()
    loss = mse(real, generated) 
    return loss


def train_step(batch, n_class):
    rng, tsrng = np.random.randint(0, iterations, size=(2,))
    timestep_values = generate_timestamp(tsrng, batch.shape[0])

    noised_image, noise = forward_noise(rng, batch, tf.cast(timestep_values, tf.int32))
    
    with tf.GradientTape() as tape:
        prediction = unet(noised_image, timestep_values, n_class)
        
        loss_value = loss_fn(tf.reshape(noise, [batch_size, dim1*dim2]), tf.reshape(prediction, [batch_size, dim1*dim2]))
        # loss_value = loss_fn(noise, prediction)

    gradients = tape.gradient(loss_value, unet.trainable_variables)
    opt.apply_gradients(zip(gradients, unet.trainable_variables))

    return loss_value


t1 = time.time()
print('- '*50)
print('   TRAINING   ')
print('- '*50)
list_avg = []

for e in range(1, epochs+1):
    # this is cool utility in Tensorflow that will create a nice looking progress bar
    bar = tf.keras.utils.Progbar(nb_shapes)
    losses = []
    
    for i in range(0, nb_shapes, batch_size):
        # print(X_train[0][0].shape)
        loss = train_step(X_train[0][i : i + batch_size].reshape(batch_size, dim1, dim2, 1), np.array([X_train[1][i : i + batch_size]], dtype=np.int32).reshape(batch_size,2)) ###
        losses.append(loss)
        bar.update(i, values=[("loss", loss)])
        
    avg = np.mean(losses)
    list_avg.append(avg)
    print('\n' + str(e) + '/' + str(epochs) + '  loss : ' + str(avg))
    # print(f"Average loss for epoch {e}/{epochs}: {avg}")
t2 = time.time()
print('- '*50)
print('   END   ')
print('- '*50)


########
# SAVE #
########

# weights
unet.save_weights(path_weights)
print('Weights saved')

# loss
with open(path_loss,'wb') as f:
    pickle.dump(np.array(list_avg), f)
    
print('Loss saved')

# time
with open(path_time,'wb') as f:
    pickle.dump(np.array(t2-t1), f)
    
print('time', round((t2-t1)/60/60,3),'h')