import logging

import numpy as np
import tensorflow as tf
import pandas as pd
import os
from datetime import datetime
from gc import collect
from keras import Input, Model
from keras.src.callbacks import EarlyStopping, ModelCheckpoint, Callback
from keras.src.layers import Dense, BatchNormalization, Bidirectional, GlobalAveragePooling1D, GRU, Conv1D, Dropout, \
    MultiHeadAttention, LSTM, GlobalMaxPooling1D, Concatenate
from keras.src.metrics import Precision, Recall, Metric
from keras.src.optimizers import Adam
from keras.src.optimizers.schedules import CosineDecay, ExponentialDecay, PiecewiseConstantDecay
from keras.src.regularizers import L2
from keras.src.saving import load_model
from tensorflow.python.keras.backend import clear_session
from tensorflow.python.keras.metrics import AUC
from keras_tuner import Hyperband, Objective, BayesianOptimization
from sklearn.metrics import classification_report, confusion_matrix
from config import memory_watchdog

###############################################################################
# TRADING METRICS & PER-CLASS AUC
###############################################################################
class PerClassAUC(Metric):
    def __init__(self, class_id, name='per_class_auc', **kwargs):
        super().__init__(name=name, **kwargs)
        self.class_id = class_id
        self.auc = AUC(curve='PR')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_class = y_true[:, self.class_id]
        y_pred_class = y_pred[:, self.class_id]
        self.auc.update_state(y_true_class, y_pred_class, sample_weight)

    def result(self):
        return self.auc.result()

    def reset_states(self):
        self.auc.reset_states()


class TradingMetrics:
    def __init__(self, num_classes=5, classes_to_monitor=None):
        if classes_to_monitor is None:
            classes_to_monitor = [0, 1, 2, 3, 4]
        self.num_classes = num_classes
        self.classes_to_monitor = classes_to_monitor
        self.weighted_accuracy = self._weighted_accuracy()
        self.accuracy = "accuracy"
        self.precision = {class_id: Precision(class_id=class_id, name=f"precision_class_{class_id}") for class_id in
                          self.classes_to_monitor}
        self.recall = {class_id: Recall(class_id=class_id, name=f"recall_class_{class_id}") for class_id in
                       self.classes_to_monitor}
        self.f1 = {class_id: self._f1_score_class(class_id) for class_id in self.classes_to_monitor}
        self.pr_auc_metrics = {class_id: PerClassAUC(class_id, name=f"pr_auc_class_{class_id}") for class_id in
                               self.classes_to_monitor}

    def _weighted_accuracy(self):
        def weighted_accuracy(y_true, y_pred):
            weights = tf.constant([1.5, 1.0, 0.5, 1.0, 1.5], dtype=tf.float32)
            y_true_idx = tf.argmax(y_true, axis=1)
            y_pred_idx = tf.argmax(y_pred, axis=1)
            correct = tf.cast(tf.equal(y_true_idx, y_pred_idx), tf.float32)
            weighted_correct = correct * tf.gather(weights, y_true_idx)
            return tf.reduce_sum(weighted_correct) / tf.reduce_sum(tf.gather(weights, y_true_idx))

        weighted_accuracy.__name__ = "weighted_accuracy"
        return weighted_accuracy

    def _f1_score_class(self, class_id):
        def f1(y_true, y_pred):
            prec = self.precision[class_id](y_true, y_pred)
            rec = self.recall[class_id](y_true, y_pred)
            return 2 * (prec * rec) / (prec + rec + tf.keras.backend.epsilon())

        f1.__name__ = f"f1_class_{class_id}"
        return f1

    def get_metrics(self):
        metrics = [self.weighted_accuracy, self.accuracy]
        for class_id in self.classes_to_monitor:
            metrics.extend(
                [self.precision[class_id], self.recall[class_id], self.f1[class_id], self.pr_auc_metrics[class_id]])
        return metrics


###############################################################################
# CUSTOM CALLBACK FOR RISK-ADJUSTED TRADING METRIC
###############################################################################
class RiskAdjustedTradeMetric(Callback):
    def __init__(self, X_val, y_val, fwd_returns_val, df_val, initial_balance=10000, kelly_fraction=0.5,
                 reward_risk_ratio=2.5, partial_close_ratio=0.5, atr_period=14, atr_multiplier_sl=1.5):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.fwd_returns_val = fwd_returns_val
        self.df_val = df_val
        self.initial_balance = initial_balance
        self.kelly_fraction = kelly_fraction
        self.reward_risk_ratio = reward_risk_ratio
        self.partial_close_ratio = partial_close_ratio
        self.atr_period = atr_period
        self.atr_multiplier_sl = atr_multiplier_sl

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        y_pred_probs = self.model.predict(self.X_val, verbose=0)
        y_pred_classes = np.argmax(y_pred_probs, axis=1)  # For classification; adjust for regression
        current_balance = self.initial_balance
        trade_returns = []

        for i in range(len(y_pred_classes)):
            pred_class = y_pred_classes[i]
            confidence = y_pred_probs[i][pred_class]
            actual_return = self.fwd_returns_val[i]
            if pred_class in [3, 4]:
                direction = 'long'
            elif pred_class in [0, 1]:
                direction = 'short'
            else:
                trade_returns.append(0)
                continue

            atr = self._compute_atr(self.df_val, self.atr_period).iloc[i]
            if np.isnan(atr) or atr <= 0:
                trade_returns.append(0)
                continue
            distance = self.atr_multiplier_sl * atr
            b = self.reward_risk_ratio
            p = confidence
            q = 1 - p
            f = max((b * p - q) / b if b > 0 else 0, 0)
            risk_fraction = f * self.kelly_fraction
            risk_amount = current_balance * risk_fraction
            quantity = risk_amount / distance

            entry_price = self.df_val['close'].iloc[i]
            if direction == 'long':
                stop_loss = entry_price - distance
                take_profit = entry_price + (self.reward_risk_ratio * distance)
                exit_price = min(entry_price * (1 + actual_return), take_profit) if actual_return >= 0 else max(
                    entry_price * (1 + actual_return), stop_loss)
            else:
                stop_loss = entry_price + distance
                take_profit = entry_price - (self.reward_risk_ratio * distance)
                exit_price = max(entry_price * (1 + actual_return), take_profit) if actual_return <= 0 else min(
                    entry_price * (1 + actual_return), stop_loss)

            pnl = quantity * (exit_price - entry_price) if direction == 'long' else quantity * (
                    entry_price - exit_price)
            trade_returns.append(pnl)
            current_balance += pnl

        avg_trade_return = np.mean(trade_returns) if trade_returns else 0
        logs['val_avg_risk_adj_return'] = avg_trade_return
        print(f"Epoch {epoch + 1}: val_avg_risk_adj_return = {avg_trade_return:.4f}")

    def _compute_atr(self, df, period):
        high_low = df['high'] - df['low']
        high_close_prev = (df['high'] - df['close'].shift(1)).abs()
        low_close_prev = (df['low'] - df['close'].shift(1)).abs()
        tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()


###############################################################################
# ENHANCED MODEL CLASS WITH KERAS TUNER
###############################################################################
class EnhancedCryptoModel:
    def __init__(self, project_name="enhanced_crypto_model", max_trials=100,
                 tuner_type="bayesian", model_save_path="best_enhanced_model.keras",
                 label_smoothing=0.1, ensemble_size=3):
        self.project_name = project_name
        self.max_trials = max_trials
        self.tuner_type = tuner_type
        self.output_classes = 5  # For classification; 1 for regression
        self.seed = 42
        self.model_save_path = model_save_path
        self.best_model = None
        self.best_hp = None
        self.label_smoothing = label_smoothing
        self.ensemble_size = ensemble_size
        self.ensemble_models = []
        tf.random.set_seed(self.seed)
        np.random.seed(self.seed)
        self.logger = logging.getLogger("EnhancedCryptoModel")

    def _transformer_block(self, x, units, num_heads):
        """Implement a transformer block with residual connections"""
        # Get input dimensions to ensure matching shapes for residual connections
        input_dim = x.shape[-1]

        # Multi-head self attention
        attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=units // num_heads)(x, x)

        # Project attention output to match input dimension if needed
        if attn_output.shape[-1] != input_dim:
            attn_output = Dense(input_dim)(attn_output)

        x = x + attn_output  # Residual connection
        x = BatchNormalization()(x)  # Normalization

        # Feed-forward network
        ffn_output = Dense(units * 2, activation='relu')(x)
        ffn_output = Dropout(0.1)(ffn_output)
        ffn_output = Dense(input_dim)(ffn_output)  # Ensure output matches input dimension

        x = x + ffn_output  # Second residual connection
        x = BatchNormalization()(x)  # Normalization

        return x

    def _build_model(self, hp, input_shape, total_steps):
        """Enhanced model architecture with transformer blocks"""
        inputs = Input(shape=input_shape, dtype=tf.float32)

        # Initial convolution layer
        filter0 = hp.Int("conv_filter_0", min_value=32, max_value=128, step=32)
        kernel_size = hp.Choice("kernel_size", values=[3, 5, 7])
        x = Conv1D(filters=filter0, kernel_size=kernel_size, padding='same', activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = Dropout(rate=hp.Float("dropout_rate_0", min_value=0.1, max_value=0.3, step=0.1))(x)

        # Option to use multi-scale convolutions
        use_multiscale = hp.Boolean("use_multiscale")
        if use_multiscale:
            # Multi-scale convolutional block
            conv3 = Conv1D(filters=filter0 // 2, kernel_size=3, padding='same', activation='relu')(x)
            conv5 = Conv1D(filters=filter0 // 2, kernel_size=5, padding='same', activation='relu')(x)
            conv7 = Conv1D(filters=filter0 // 2, kernel_size=7, padding='same', activation='relu')(x)
            x = Concatenate()([conv3, conv5, conv7])
            x = BatchNormalization()(x)

        # First recurrent block with LSTM
        lstm_type = hp.Choice("lstm_type", values=["LSTM", "GRU"])
        dropout_rate1 = hp.Float("dropout_rate_1", min_value=0.1, max_value=0.5, step=0.1)
        l2_reg1 = hp.Float("l2_reg_1", min_value=1e-5, max_value=1e-2, sampling='log')
        unit1 = hp.Int("unit_1", min_value=32, max_value=128, step=32)

        if lstm_type == "LSTM":
            x = Bidirectional(LSTM(unit1, return_sequences=True, dropout=dropout_rate1,
                                   kernel_regularizer=L2(l2_reg1)))(x)
        else:
            x = Bidirectional(GRU(unit1, return_sequences=True, dropout=dropout_rate1,
                                  kernel_regularizer=L2(l2_reg1)))(x)

        x = BatchNormalization()(x)

        # Transformer blocks
        num_transformer_blocks = hp.Int("num_transformer_blocks", min_value=1, max_value=3)
        transformer_units = hp.Int("transformer_units", min_value=32, max_value=128, step=32)
        num_heads = hp.Int("num_heads", min_value=2, max_value=8, step=2)

        for i in range(num_transformer_blocks):
            x = self._transformer_block(x, transformer_units, num_heads)

        # Global pooling strategy
        pooling_type = hp.Choice("pooling_type", values=["average", "max"])

        if pooling_type == "average":
            x = GlobalAveragePooling1D()(x)
        else:
            x = GlobalMaxPooling1D()(x)

        # Final classification layer
        x = Dense(128, activation="relu")(x)
        x = Dropout(rate=hp.Float("final_dropout", min_value=0.1, max_value=0.5, step=0.1))(x)
        outputs = Dense(self.output_classes, activation="softmax", dtype=tf.float32)(x)

        # Compile with custom metrics and loss
        trading_metrics = TradingMetrics(num_classes=5, classes_to_monitor=[0, 1, 2, 3, 4])
        metrics = trading_metrics.get_metrics()

        # Focal loss with adjustable gamma
        gamma = hp.Float("gamma", min_value=1.0, max_value=5.0, step=0.5)
        loss_fn = tf.keras.losses.CategoricalFocalCrossentropy(gamma=gamma, label_smoothing=self.label_smoothing)

        # Learning rate schedule
        lr_schedule_type = hp.Choice("lr_schedule", values=["cosine", "exponential", "step"])
        initial_lr = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling='log')

        if lr_schedule_type == "cosine":
            lr_schedule = CosineDecay(initial_learning_rate=initial_lr, decay_steps=total_steps, alpha=0.1)
        elif lr_schedule_type == "exponential":
            lr_schedule = ExponentialDecay(initial_learning_rate=initial_lr, decay_steps=total_steps // 4,
                                           decay_rate=0.9)
        else:
            lr_schedule = PiecewiseConstantDecay(
                boundaries=[total_steps // 3, total_steps * 2 // 3],
                values=[initial_lr, initial_lr * 0.1, initial_lr * 0.01]
            )

        optimizer = Adam(learning_rate=lr_schedule)

        model = Model(inputs, outputs)
        model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)

        return model

    def tune_and_train(self, iteration, X_train, y_train, X_val, y_val, df_val, fwd_returns_val, epochs=32,
                       batch_size=256, class_weight=None):
        """Train model without small dataset adjustments"""
        if len(X_train) == 0:
            self.logger.warning("No training data. Skipping tuner.")
            return None, None

        input_shape = (X_train.shape[1], X_train.shape[2])
        steps_per_epoch = len(X_train) // batch_size + (1 if len(X_train) % batch_size != 0 else 0)
        total_steps = steps_per_epoch * epochs
        objective = Objective("val_avg_risk_adj_return", direction="max")

        # Use the originally specified tuner type
        if self.tuner_type.lower() == "hyperband":
            tuner = Hyperband(
                hypermodel=lambda hp: self._build_model(hp, input_shape, total_steps),
                objective=objective,
                max_epochs=epochs,
                factor=3,
                executions_per_trial=1,
                project_name=self.project_name,
                overwrite=True,
                seed=self.seed
            )
        else:
            tuner = BayesianOptimization(
                hypermodel=lambda hp: self._build_model(hp, input_shape, total_steps),
                objective=objective,
                max_trials=self.max_trials,
                executions_per_trial=1,
                project_name=self.project_name,
                overwrite=True,
                seed=self.seed
            )

        # Standard early stopping with reasonable patience
        patience = 10
        es = EarlyStopping(monitor='val_avg_risk_adj_return', patience=patience, restore_best_weights=True, mode='max')
        checkpoint = ModelCheckpoint(self.model_save_path, monitor='val_avg_risk_adj_return', save_best_only=True,
                                    mode='max')
        callback = RiskAdjustedTradeMetric(X_val, y_val, fwd_returns_val, df_val)

        # Use shuffled dataset with full batch size
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(
            buffer_size=min(len(X_train), 10000), seed=self.seed).batch(
            batch_size).prefetch(tf.data.AUTOTUNE)

        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

        tuner.search(train_dataset, validation_data=val_dataset, epochs=epochs, callbacks=[callback, es, checkpoint],
                    class_weight=class_weight, verbose=2)

        try:
            best_trial = tuner.oracle.get_best_trials(1)[0]
            best_hp = best_trial.hyperparameters
            self.best_hp = best_hp.values
            self.best_model = tuner.hypermodel.build(best_hp)

            # Fixed the bug with the hyperparameter access
            row = {
                'iteration': iteration,
                'trial_id': best_trial.trial_id,
                'learning_rate': best_hp.values.get('lr'),
                'total_steps': total_steps,
                'conv_filter_0': best_hp.values.get('conv_filter_0'),
                'dropout_rate_0': best_hp.values.get('dropout_rate_0'),
                'lstm_unit_1': best_hp.values.get('unit_1'),
                'dropout_rate_1': best_hp.values.get('dropout_rate_1'),
                'l2_reg_1': best_hp.values.get('l2_reg_1'),
                'transformer_blocks': best_hp.values.get('num_transformer_blocks', 1),
                'transformer_units': best_hp.values.get('transformer_units', 64),
                'num_heads': best_hp.values.get('num_heads', 4),
                'gamma': best_hp.values.get('gamma'),
                'val_loss': best_trial.metrics.get_last_value('val_loss'),
                'val_accuracy': best_trial.metrics.get_last_value('val_accuracy'),
                'val_weighted_accuracy': best_trial.metrics.get_last_value('val_weighted_accuracy'),
                'val_avg_risk_adj_return': best_trial.metrics.get_last_value('val_avg_risk_adj_return'),
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'message': 'completed'
            }

            df_best = pd.DataFrame([row])
            csv_path = "EnhancedTrainingResults/best_trials_log.csv"
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            if not os.path.isfile(csv_path):
                df_best.to_csv(csv_path, index=False)
            else:
                df_best.to_csv(csv_path, mode='a', header=False, index=False)
        except (IndexError, ValueError) as e:
            self.logger.error(f"Error getting best trial: {e}")

            # If no best trial found, create a simple model with default parameters
            if len(X_train) > 0 and self.best_model is None:
                self.logger.info("Creating default model since no best model was found")
                input_shape = (X_train.shape[1], X_train.shape[2])

                # Create a simple model with minimal hyperparameters
                inputs = Input(shape=input_shape, dtype=tf.float32)
                x = Conv1D(filters=64, kernel_size=5, padding='same', activation='relu')(inputs)
                x = BatchNormalization()(x)
                x = Dropout(rate=0.2)(x)
                x = Bidirectional(LSTM(64, return_sequences=True, dropout=0.2))(x)
                x = BatchNormalization()(x)
                x = GlobalAveragePooling1D()(x)
                x = Dense(128, activation="relu")(x)
                x = Dropout(rate=0.2)(x)
                outputs = Dense(self.output_classes, activation="softmax", dtype=tf.float32)(x)

                self.best_model = Model(inputs, outputs)

                # Compile with basic settings
                trading_metrics = TradingMetrics(num_classes=5, classes_to_monitor=[0, 1, 2, 3, 4])
                metrics = trading_metrics.get_metrics()
                loss_fn = tf.keras.losses.CategoricalFocalCrossentropy(gamma=2.0, label_smoothing=self.label_smoothing)

                self.best_model.compile(
                    optimizer=Adam(learning_rate=0.001),
                    loss=loss_fn,
                    metrics=metrics
                )

                # Fit the model
                self.best_model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=min(20, epochs),
                    batch_size=batch_size,
                    class_weight=class_weight,
                    callbacks=[es, checkpoint],
                    verbose=1
                )

                # Save the model
                self.best_model.save(self.model_save_path)
                self.logger.info(f"Default model trained and saved to {self.model_save_path}")

        del tuner
        clear_session()
        collect()
        return self.best_model, self.best_hp

    def build_ensemble(self, X_train, y_train, X_val, y_val, df_val, fwd_returns_val, epochs=32, batch_size=256,
                       class_weight=None):
        """Train an ensemble of models with different seeds and architectures"""
        # Save original seed
        original_seed = self.seed

        for i in range(self.ensemble_size):
            # Use different seeds for ensemble diversity
            self.seed = original_seed + i * 100
            tf.random.set_seed(self.seed)
            np.random.seed(self.seed)

            # Custom model save path for each ensemble member
            ensemble_path = f"{self.model_save_path.replace('.keras', '')}_ensemble_{i}.keras"

            clear_session()
            collect()
            # Train each model
            model, hp = self.tune_and_train(i, X_train, y_train, X_val, y_val, df_val, fwd_returns_val, epochs,
                                            batch_size, class_weight)

            if model is not None:
                # Save model
                model.save(ensemble_path)
                self.ensemble_models.append((model, hp))
                print(f"Ensemble model {i + 1}/{self.ensemble_size} trained and saved.")

        # Restore original seed
        self.seed = original_seed
        tf.random.set_seed(self.seed)
        np.random.seed(self.seed)

        return self.ensemble_models

    def predict_with_ensemble(self, X_new, batch_size=256):
        """Make predictions using the ensemble of models"""
        if not self.ensemble_models:
            raise RuntimeError("No ensemble models found. Train ensemble first.")

        # Add memory logging
        memory_watchdog(threshold_gb=20)

        all_predictions = []
        for i, (model, _) in enumerate(self.ensemble_models):

            # Check memory before each model prediction
            memory_watchdog(threshold_gb=20)

            pred = model.predict(X_new, batch_size=batch_size, verbose=0)
            all_predictions.append(pred)

            # Clear individual model from memory if not last one
            if i < len(self.ensemble_models) - 1:
                # Only keep the model we need in memory
                del model
                clear_session()

        # Average predictions from all models
        ensemble_pred = np.mean(all_predictions, axis=0)

        # Calculate uncertainty (standard deviation across models)
        uncertainty = np.std(all_predictions, axis=0)

        # Cleanup after prediction
        clear_session()
        memory_watchdog(threshold_gb=14)
        return ensemble_pred, uncertainty

    def load_ensemble(self, base_path=None, num_models=None):
        """Load a previously trained ensemble"""
        if base_path is None:
            base_path = self.model_save_path.replace('.keras', '')

        if num_models is None:
            num_models = self.ensemble_size

        self.ensemble_models = []

        for i in range(num_models):
            # Clear memory before loading each model
            clear_session()
            collect()
            model_path = f"{base_path}_ensemble_{i}.keras"
            if os.path.exists(model_path):
                model = load_model(model_path)
                self.ensemble_models.append((model, None))  # We don't have hyperparameters here
            # Monitor memory after loading each model
            memory_watchdog(threshold_gb=20)

        print(f"Loaded {len(self.ensemble_models)} ensemble models.")
        return len(self.ensemble_models) > 0

    def evaluate(self, X_val, y_val, batch_size=256):
        if hasattr(self, 'ensemble_models') and self.ensemble_models:
            print("Evaluating ensemble model")
            ensemble_preds, uncertainties = self.predict_with_ensemble(X_val, batch_size)
            y_pred, y_true = np.argmax(ensemble_preds, axis=1), np.argmax(y_val, axis=1)
            print("Classification Report:\n", classification_report(y_true, y_pred, digits=4))
            print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
            print(f"Average uncertainty: {np.mean(uncertainties):.4f}")
            return

        if self.best_model is None:
            print("No model to evaluate.")
            return
        metrics_vals = self.best_model.evaluate(X_val, y_val, batch_size=batch_size, verbose=0)
        print("Validation metrics:", metrics_vals)
        y_probs = self.best_model.predict(X_val, batch_size=batch_size, verbose=0)
        y_pred, y_true = np.argmax(y_probs, axis=1), np.argmax(y_val, axis=1)
        print("Classification Report:\n", classification_report(y_true, y_pred, digits=4))
        print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

    def load_best_model(self):
        if os.path.exists(self.model_save_path):
            self.best_model = load_model(self.model_save_path)
            print(f"Loaded model from {self.model_save_path}")
        else:
            print(f"No model found at {self.model_save_path}")

    def predict_signals(self, X_new, batch_size=256):
        if hasattr(self, 'ensemble_models') and self.ensemble_models:
            preds, _ = self.predict_with_ensemble(X_new, batch_size)
            return preds

        if self.best_model is None:
            raise RuntimeError("No model found. Train or load a model first.")
        return self.best_model.predict(X_new, batch_size=batch_size, verbose=0)