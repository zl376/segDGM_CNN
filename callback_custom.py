from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau

class EarlyStoppingLowLR(EarlyStopping):
    """Stop training when a monitored quantity has stopped improving
    and learning rate smaller (or equal) than threshold
    """

    def __init__(self, monitor='val_loss',
                 min_delta=0, patience=0, verbose=0, mode='auto', thresh_LR=0):
        super().__init__(monitor, min_delta, patience, verbose, mode)
        self.thresh_LR = thresh_LR
        self.last_LR = None
        
    def _checkSameLR(self):
        """Check if learning rate has changed since last check
        """
        LR = K.get_value(self.model.optimizer.lr)
        if self.last_LR == LR:
            return True
        else:
            self.last_LR = LR
            return False

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn(
                'Early stopping conditioned on metric `%s` '
                'which is not available. Available metrics are: %s' %
                (self.monitor, ','.join(list(logs.keys()))), RuntimeWarning
            )
            return
        if self.monitor_op(current - self.min_delta, self.best) or \
           not self._checkSameLR():
            self.best = current
            self.wait = 0
        else:
            print(self.last_LR, self.thresh_LR)
            self.wait += 1
            if self.wait >= self.patience and self.last_LR <= self.thresh_LR*1.01:
                self.stopped_epoch = epoch
                self.model.stop_training = True

                
class ReduceLROnPlateauBestWeight(ReduceLROnPlateau):
    """Reduce learning rate when a metric has stopped improving.
    When LR changes, load the best weight
    """

    def __init__(self, filepath, monitor='val_loss', factor=0.1, patience=10,
                 verbose=0, mode='auto', epsilon=1e-4, cooldown=0, min_lr=0):
        super().__init__(monitor, factor, patience, verbose, mode, epsilon, cooldown, min_lr)

        self.filepath = filepath

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn(
                'Reduce LR on plateau conditioned on metric `%s` '
                'which is not available. Available metrics are: %s' %
                (self.monitor, ','.join(list(logs.keys()))), RuntimeWarning
            )

        else:
            if self.in_cooldown():
                self.cooldown_counter -= 1
                self.wait = 0

            if self.monitor_op(current, self.best):
                self.best = current
                self.wait = 0
            elif not self.in_cooldown():
                if self.wait >= self.patience:
                    old_lr = float(K.get_value(self.model.optimizer.lr))
                    if old_lr > self.min_lr:
                        new_lr = old_lr * self.factor
                        new_lr = max(new_lr, self.min_lr)
                        K.set_value(self.model.optimizer.lr, new_lr)
                        # load best weight
                        self.model.load_weights(self.filepath)
                        if self.verbose > 0:
                            print('\nEpoch %05d: ReduceLROnPlateau reducing learning '
                                  'rate to %s.' % (epoch + 1, new_lr))
                        self.cooldown_counter = self.cooldown
                        self.wait = 0
                self.wait += 1
