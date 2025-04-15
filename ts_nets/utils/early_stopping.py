class EarlyStopping:
    
    def __init__(self, patience=5, verbose=True, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 5
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.best_epoch = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.best_model_weights = None

    def __call__(self, val_loss, model, epoch, n_epochs, train_loss):
        score = -val_loss

        post_message = ''

        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch+1
            self.train_loss_best_epoch = train_loss
            self.save_checkpoint(val_loss, model)

        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                # Restore the best model weights if early stopping is triggered
                if -self.best_score < 0.001 :
                    post_message += f"\nRestoring best model weights from epoch {self.best_epoch} with Validation Loss = {-self.best_score:.4e}"
                else :
                    post_message += f"\nRestoring best model weights from epoch {self.best_epoch} with Validation Loss = {-self.best_score:.4f}"
                model.load_state_dict(self.best_model_weights)

        else:
            self.best_score = score
            self.best_epoch = epoch+1
            self.train_loss_best_epoch = train_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0

        message = f"Epoch {epoch+1}/{n_epochs} - Training Loss: {train_loss:.4e}, Validation Loss: {val_loss:.4e} - Early Stopping counter: {self.counter}/{self.patience}"

        return message + post_message

    def save_checkpoint(self, val_loss, model):
        """Saves the model when the validation loss decreases."""
        # Save the current best model's state_dict
        self.best_model_weights = model.state_dict().copy()
        self.val_loss_min = val_loss
