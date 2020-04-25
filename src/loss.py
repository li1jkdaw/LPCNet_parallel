from keras import backend as K


C = 10.0


def separator_loss(y_true, y_pred):
    loss = K.sum((y_pred*y_true[:,:,0:1] + (1.0 - y_pred)*C)*y_true[:,:,1:2])
    num_resets = K.sum(y_true[:,:,1])
    return loss/num_resets
