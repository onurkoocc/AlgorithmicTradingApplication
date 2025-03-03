import logging
import os
from gc import collect

import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Input, Model
from keras.src.callbacks import EarlyStopping, ModelCheckpoint, Callback
from keras.src.layers import (
    Dense, BatchNormalization, GlobalAveragePooling1D,
    GRU, Conv1D, Dropout
)
from keras.src.metrics import Precision, Recall, Metric
from keras.src.optimizers import Adam
from keras.src.saving import load_model
from keras_tuner import BayesianOptimization, Objective
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.python.keras.backend import clear_session
from tensorflow.python.keras.metrics import AUC

from memory_utils import memory_watchdog

###############################################################################
# LOGGER CONFIG
###############################################################################
logger = logging.getLogger("ModelManagement")
if not logger.handlers:
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(sh)
logger.setLevel(logging.INFO)

###############################################################################
# CUSTOM METRICS
###############################################################################
class PerClassAUC(Metric):
    """Custom metric for per-class AUC (PR curve)."""

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
    """Collection of trading-specific metrics."""

    def __init__(self, num_classes=5, classes_to_monitor=None):
        if classes_to_monitor is None:
            classes_to_monitor = [0, 1, 2, 3, 4]
        self.num_classes = num_classes
        self.classes_to_monitor = classes_to_monitor
        self.weighted_accuracy = self._weighted_accuracy()
        self.accuracy = "accuracy"
        self.precision = {
            cid: Precision(class_id=cid, name=f"precision_class_{cid}")
            for cid in self.classes_to_monitor
        }
        self.recall = {
            cid: Recall(class_id=cid, name=f"recall_class_{cid}")
            for cid in self.classes_to_monitor
        }
        self.f1 = {
            cid: self._f1_score_class(cid)
            for cid in self.classes_to_monitor
        }
        self.pr_auc_metrics = {
            cid: PerClassAUC(cid, name=f"pr_auc_class_{cid}")
            for cid in self.classes_to_monitor
        }

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
        m = [self.weighted_accuracy, self.accuracy]
        for cid in self.classes_to_monitor:
            m.extend([
                self.precision[cid],
                self.recall[cid],
                self.f1[cid],
                self.pr_auc_metrics[cid],
            ])
        return m

###############################################################################
# CUSTOM CALLBACK
###############################################################################
class RiskAdjustedTradeMetric(Callback):
    """Custom callback for evaluating risk-adjusted return at epoch end."""

    def __init__(self, X_val, y_val, fwd_returns_val, df_val,
                 initial_balance=10000, kelly_fraction=0.5, reward_risk_ratio=2.5,
                 partial_close_ratio=0.5, atr_period=14, atr_multiplier_sl=1.5):
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
        try:
            batch_size = 32
            preds = []
            for i in range(0, len(self.X_val), batch_size):
                batch_end = min(i + batch_size, len(self.X_val))
                pr = self.model.predict(self.X_val[i:batch_end], verbose=0)
                preds.append(pr)
            y_pred_probs = np.vstack(preds)
            y_pred_classes = np.argmax(y_pred_probs, axis=1)

            current_balance = self.initial_balance
            trade_returns = []
            atr_series = self._compute_atr(self.df_val, self.atr_period)

            for i, pred_class in enumerate(y_pred_classes):
                conf = y_pred_probs[i][pred_class]
                act_ret = self.fwd_returns_val[i]
                if pred_class in [3, 4]:
                    direction = 'long'
                elif pred_class in [0, 1]:
                    direction = 'short'
                else:
                    trade_returns.append(0)
                    continue

                atr_val = atr_series.iloc[i] if i < len(atr_series) else np.nan
                if np.isnan(atr_val) or atr_val <= 0:
                    trade_returns.append(0)
                    continue

                distance = self.atr_multiplier_sl * atr_val
                b = self.reward_risk_ratio
                p = conf
                q = 1 - p
                f = max((b * p - q) / b, 0) if b > 0 else 0
                risk_fraction = f * self.kelly_fraction
                risk_amount = current_balance * risk_fraction
                quantity = risk_amount / distance

                entry_price = self.df_val['close'].iloc[i]
                if direction == 'long':
                    stop_loss = entry_price - distance
                    take_profit = entry_price + b * distance
                    exit_price = min(entry_price * (1 + act_ret), take_profit) if act_ret >= 0 \
                        else max(entry_price * (1 + act_ret), stop_loss)
                    pnl = quantity * (exit_price - entry_price)
                else:
                    stop_loss = entry_price + distance
                    take_profit = entry_price - b * distance
                    exit_price = max(entry_price * (1 + act_ret), take_profit) if act_ret <= 0 \
                        else min(entry_price * (1 + act_ret), stop_loss)
                    pnl = quantity * (entry_price - exit_price)

                trade_returns.append(pnl)
                current_balance += pnl

            del y_pred_probs, y_pred_classes, preds
            collect()

            avg_trade_return = np.mean(trade_returns) if trade_returns else 0
            logs['val_avg_risk_adj_return'] = avg_trade_return
            print(f"Epoch {epoch + 1}: val_avg_risk_adj_return = {avg_trade_return:.4f}")
        except Exception as e:
            print(f"Error in RiskAdjustedTradeMetric: {e}")
            logs['val_avg_risk_adj_return'] = 0.0

    def _compute_atr(self, df, period):
        hl = df['high'] - df['low']
        hc = (df['high'] - df['close'].shift(1)).abs()
        lc = (df['low'] - df['close'].shift(1)).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

###############################################################################
# MODEL BUILDER
###############################################################################
class ModelBuilder:
    """Builds model architectures for short-term trading."""

    def __init__(self, output_classes=5, seed=42):
        self.output_classes = output_classes
        self.seed = seed
        tf.random.set_seed(self.seed)
        np.random.seed(self.seed)
        self.logger = logging.getLogger("ModelBuilder")

    def build_model(self, hp, input_shape, total_steps=None):
        """Simplified model builder for short-term trading."""
        try:
            if len(input_shape) != 2 or any(dim <= 0 for dim in input_shape):
                raise ValueError(f"Invalid input shape: {input_shape}")

            inputs = Input(shape=input_shape, dtype=tf.float32)
            filters = hp.Choice("conv_filters", [16, 32])
            kernel_size = hp.Choice("kernel_size", [3])
            x = Conv1D(filters=filters, kernel_size=kernel_size, padding='same', activation='relu')(inputs)
            x = BatchNormalization()(x)
            x = Dropout(rate=0.2)(x)

            gru_units = hp.Choice("gru_units", [16, 32])
            x = GRU(gru_units, return_sequences=True,
                    dropout=0.2, activation='tanh', recurrent_activation='sigmoid')(x)
            x = BatchNormalization()(x)

            x = GlobalAveragePooling1D()(x)
            dense_units = hp.Choice("dense_units", [16, 32])
            x = Dense(dense_units, activation="relu")(x)
            x = Dropout(rate=0.2)(x)

            outputs = Dense(self.output_classes, activation="softmax", dtype=tf.float32)(x)

            loss_fn = tf.keras.losses.CategoricalFocalCrossentropy(gamma=2.0, label_smoothing=0.1)
            initial_lr = hp.Float("lr", min_value=1e-4, max_value=1e-3, sampling='log')

            if total_steps:
                lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
                    initial_learning_rate=initial_lr,
                    decay_steps=total_steps,
                    alpha=0.1
                )
            else:
                lr_schedule = initial_lr

            optimizer = Adam(learning_rate=lr_schedule, beta_1=0.9, beta_2=0.999, epsilon=1e-7)

            model = Model(inputs, outputs)
            model.compile(
                optimizer=optimizer,
                loss=loss_fn,
                metrics=['accuracy'],
                jit_compile=True,
                steps_per_execution=8
            )
            return model

        except Exception as e:
            self.logger.error(f"Error building model: {e}")
            return self._create_minimal_model(input_shape)

    def _create_minimal_model(self, input_shape):
        """Fallback minimal model."""
        try:
            inputs = Input(shape=input_shape, dtype=tf.float32)
            x = Conv1D(filters=16, kernel_size=3, padding='same', activation='relu')(inputs)
            x = BatchNormalization()(x)
            x = Dropout(rate=0.2)(x)
            x = GRU(16, return_sequences=False)(x)
            x = BatchNormalization()(x)
            x = Dense(16, activation="relu")(x)
            x = Dropout(rate=0.2)(x)
            outputs = Dense(self.output_classes, activation="softmax", dtype=tf.float32)(x)

            model = Model(inputs, outputs)
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy'],
                jit_compile=True
            )
            return model
        except Exception as e:
            self.logger.error(f"Error creating minimal model: {e}")
            return None

###############################################################################
# MODEL TRAINER
###############################################################################
class ModelTrainer:
    """Handles training, tuning, and evaluation of models."""

    def __init__(self, model_builder=None, project_name="enhanced_crypto_model",
                 max_trials=3, tuner_type="bayesian", seed=42):
        self.model_builder = model_builder or ModelBuilder()
        self.project_name = project_name
        self.max_trials = max_trials
        self.tuner_type = tuner_type
        self.seed = seed
        self.best_model = None
        self.best_hp = None
        self.logger = logging.getLogger("ModelTrainer")

    def train_model(self, X_train, y_train, X_val, y_val, df_val, fwd_returns_val,
                    epochs=10, batch_size=64, class_weight=None, model_save_path="best_model.keras"):
        if len(X_train) == 0:
            self.logger.warning("No training data. Skipping training.")
            return None, None

        input_shape = (X_train.shape[1], X_train.shape[2])
        self.logger.info(f"Training with input shape: {input_shape}")

        # Adjust batch size
        if X_train.shape[2] > 100:
            adjusted_batch_size = min(32, batch_size)
        elif len(X_train) < 1000:
            adjusted_batch_size = min(64, len(X_train) // 8)
        else:
            adjusted_batch_size = batch_size
        adjusted_batch_size = min(adjusted_batch_size, len(X_train) // 16)
        adjusted_batch_size = max(adjusted_batch_size, 16)

        use_tuning = (len(X_train) > 2000) and (self.max_trials > 1)
        memory_watchdog(threshold_gb=20, component="before_model_training")

        if use_tuning:
            return self._tune_model_bayesian(
                input_shape, X_train, y_train, X_val, y_val,
                fwd_returns_val, df_val, epochs, adjusted_batch_size,
                class_weight, model_save_path
            )
        else:
            return self._train_with_walk_forward(
                input_shape, X_train, y_train, X_val, y_val,
                fwd_returns_val, df_val, epochs, adjusted_batch_size,
                class_weight, model_save_path
            )

    def _train_with_walk_forward(self, input_shape, X_train, y_train, X_val, y_val,
                                 fwd_returns_val, df_val, epochs, batch_size,
                                 class_weight, model_save_path):
        """Walk-forward validation approach."""
        try:
            memory_watchdog(threshold_gb=20, component="before_walk_forward")

            from keras_tuner.engine.hyperparameters import HyperParameters
            hp = HyperParameters()
            hp.Fixed("conv_filters", 32)
            hp.Fixed("kernel_size", 3)
            hp.Fixed("gru_units", 32)
            hp.Fixed("dense_units", 32)
            hp.Fixed("lr", 0.0005)

            model = self.model_builder.build_model(hp, input_shape)
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, min_delta=0.001),
                ModelCheckpoint(model_save_path, monitor='val_loss', save_best_only=True),
                RiskAdjustedTradeMetric(X_val, y_val, fwd_returns_val, df_val)
            ]

            train_len = len(X_train)
            window_size = max(train_len // 4, 100)
            step_size = window_size // 2

            for start_idx in range(0, train_len - window_size, step_size):
                end_idx = start_idx + window_size
                val_end_idx = min(end_idx + step_size, train_len)
                if len(X_train[end_idx:val_end_idx]) < 10:
                    continue
                X_window, y_window = X_train[start_idx:end_idx], y_train[start_idx:end_idx]
                X_val_window, y_val_window = X_train[end_idx:val_end_idx], y_train[end_idx:val_end_idx]
                model.fit(
                    X_window, y_window,
                    validation_data=(X_val_window, y_val_window),
                    epochs=max(3, epochs // 3),
                    batch_size=batch_size,
                    class_weight=class_weight,
                    callbacks=callbacks,
                    verbose=1
                )
                memory_watchdog(threshold_gb=20, component=f"after_window_{start_idx}")

            self.logger.info("Final short training on full dataset")
            model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=3,
                batch_size=batch_size,
                class_weight=class_weight,
                callbacks=callbacks,
                verbose=1
            )

            model.save(model_save_path)
            self.logger.info(f"Model saved to {model_save_path}")
            self.best_model = model
            self.best_hp = hp
            return model, hp

        except Exception as e:
            self.logger.error(f"Error in walk-forward training: {e}")
            try:
                m = self.model_builder._create_minimal_model(input_shape)
                from keras_tuner.engine.hyperparameters import HyperParameters
                mhp = HyperParameters()
                m.save(model_save_path)
                return m, mhp
            except:
                return None, None

    def _tune_model_bayesian(self, input_shape, X_train, y_train, X_val, y_val,
                             fwd_returns_val, df_val, epochs, batch_size,
                             class_weight, model_save_path):
        """Tune model hyperparameters using Bayesian optimization."""
        try:
            memory_watchdog(threshold_gb=20, component="before_bayesian_tuning")
            steps_per_epoch = len(X_train) // batch_size + (1 if len(X_train) % batch_size != 0 else 0)
            total_steps = steps_per_epoch * epochs
            objective = Objective("val_avg_risk_adj_return", direction="max")

            tuner = BayesianOptimization(
                lambda hp: self.model_builder.build_model(hp, input_shape, total_steps),
                objective=objective,
                max_trials=self.max_trials,
                seed=self.seed,
                project_name=self.project_name,
                overwrite=True
            )

            train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(
                buffer_size=min(len(X_train), 5000), seed=self.seed
            ).batch(batch_size).prefetch(tf.data.AUTOTUNE)
            val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

            risk_callback = RiskAdjustedTradeMetric(X_val, y_val, fwd_returns_val, df_val)
            es = EarlyStopping(monitor='val_avg_risk_adj_return', patience=3,
                               restore_best_weights=True, mode='max')

            tuner.search(
                train_dataset,
                validation_data=val_dataset,
                epochs=min(5, epochs),
                callbacks=[risk_callback, es],
                class_weight=class_weight,
                verbose=1
            )

            memory_watchdog(threshold_gb=20, component="after_bayesian_tuning", force_cleanup=True)

            best_hp = tuner.get_best_hyperparameters(1)[0]
            self.logger.info(f"Best hyperparameters: {best_hp.values}")

            self.best_model = tuner.hypermodel.build(best_hp)
            self.best_hp = best_hp

            checkpoint = ModelCheckpoint(model_save_path, monitor='val_avg_risk_adj_return',
                                         save_best_only=True, mode='max')
            self.best_model.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=epochs,
                callbacks=[risk_callback, es, checkpoint],
                class_weight=class_weight,
                verbose=1
            )

            self.best_model.save(model_save_path)
            self.logger.info(f"Best model saved to {model_save_path}")

            del tuner
            clear_session()
            collect()

            return self.best_model, self.best_hp

        except Exception as e:
            self.logger.error(f"Error in Bayesian tuning: {e}")
            return self._train_with_walk_forward(
                input_shape, X_train, y_train, X_val, y_val,
                fwd_returns_val, df_val, epochs, batch_size,
                class_weight, model_save_path
            )

###############################################################################
# MODEL PREDICTOR
###############################################################################
class ModelPredictor:
    """Handles loading models and making predictions."""

    def __init__(self, model_path="best_model.keras", ensemble_paths=None):
        self.model_path = model_path
        self.ensemble_paths = ensemble_paths or []
        self.model = None
        self.ensemble_models = []
        self.logger = logging.getLogger("ModelPredictor")

    def load_model(self):
        if os.path.exists(self.model_path):
            try:
                self.model = load_model(self.model_path)
                self.logger.info(f"Loaded model from {self.model_path}")
                return True
            except Exception as e:
                self.logger.error(f"Error loading model: {e}")
                return False
        else:
            self.logger.warning(f"No model found at {self.model_path}")
            return False

    def load_ensemble(self):
        self.ensemble_models = []
        if os.path.exists(self.model_path):
            try:
                m = load_model(self.model_path)
                self.ensemble_models.append(m)
            except Exception as e:
                self.logger.error(f"Error loading primary model: {e}")

        for path in self.ensemble_paths:
            if os.path.exists(path):
                try:
                    m = load_model(path)
                    self.ensemble_models.append(m)
                    self.logger.info(f"Loaded ensemble model from {path}")
                except Exception as e:
                    self.logger.error(f"Error loading ensemble model {path}: {e}")

        self.logger.info(f"Loaded {len(self.ensemble_models)} ensemble models")
        return len(self.ensemble_models) > 0

    def find_ensemble_models(self, base_path=None, max_models=2):
        if base_path is None:
            base_path = self.model_path.replace('.keras', '')
        max_models = min(max_models, 2)
        ensemble_paths = []
        for i in range(max_models):
            path = f"{base_path}_ensemble_{i}.keras"
            if os.path.exists(path):
                ensemble_paths.append(path)
        self.ensemble_paths = ensemble_paths
        return ensemble_paths

    def predict(self, X_new, batch_size=32, use_ensemble=True):
        batch_size = min(batch_size, 32)
        if not self.ensemble_models and not self.model:
            loaded = self.load_ensemble() if use_ensemble else self.load_model()
            if not loaded:
                self.logger.error("No models could be loaded for prediction.")
                fallback = np.zeros((len(X_new), 5))
                fallback[:, 2] = 1.0  # neutral
                return (fallback, np.zeros((len(X_new), 5))) if use_ensemble else fallback

        if use_ensemble and self.ensemble_models:
            return self._predict_with_ensemble(X_new, batch_size)
        else:
            return self._predict_with_single_model(X_new, batch_size)

    def _predict_with_ensemble(self, X_new, batch_size=32):
        if not self.ensemble_models:
            if len(self.ensemble_paths) > 0:
                self.load_ensemble()
            else:
                self.find_ensemble_models()
                self.load_ensemble()
        if not self.ensemble_models:
            self.logger.warning("Ensemble not available. Falling back to single model.")
            return self._predict_with_single_model(X_new, batch_size)

        try:
            memory_watchdog(threshold_gb=20)
            all_preds = []
            for model in self.ensemble_models:
                pred_batches = []
                for j in range(0, len(X_new), batch_size):
                    end_j = min(j + batch_size, len(X_new))
                    try:
                        batch_pred = model.predict(X_new[j:end_j], verbose=0)
                    except (tf.errors.InternalError, tf.errors.ResourceExhaustedError):
                        with tf.device('/CPU:0'):
                            batch_pred = model.predict(X_new[j:end_j], verbose=0)
                    except Exception as e:
                        self.logger.error(f"Ensemble batch prediction error: {e}")
                        batch_pred = np.zeros((end_j - j, 5))
                        batch_pred[:, 2] = 1.0
                    pred_batches.append(batch_pred)
                model_preds = np.vstack(pred_batches) if pred_batches else np.zeros((len(X_new), 5))
                all_preds.append(model_preds)
                clear_session()
                collect()
                memory_watchdog(threshold_gb=20)

            if all_preds:
                arr = np.array(all_preds)
                ensemble_pred = np.mean(arr, axis=0)
                uncertainty = np.std(arr, axis=0)
            else:
                ensemble_pred = np.zeros((len(X_new), 5))
                ensemble_pred[:, 2] = 1.0
                uncertainty = np.zeros((len(X_new), 5))

            memory_watchdog(threshold_gb=14)
            return ensemble_pred, uncertainty
        except Exception as e:
            self.logger.error(f"Ensemble prediction error: {e}")
            return self._predict_with_single_model(X_new, batch_size)

    def _predict_with_single_model(self, X_new, batch_size=32):
        if self.model is None:
            if not self.load_model():
                if self.ensemble_models:
                    self.model = self.ensemble_models[0]
                else:
                    fallback = np.zeros((len(X_new), 5))
                    fallback[:, 2] = 1.0
                    return fallback

        expected_features = self.model.input_shape[2]
        actual_features = X_new.shape[2]
        if actual_features != expected_features:
            self.logger.warning(f"Feature mismatch: model={expected_features}, data={actual_features}")
            if actual_features > expected_features:
                X_new = X_new[:, :, :expected_features]
            else:
                pad = np.zeros((X_new.shape[0], X_new.shape[1], expected_features - actual_features))
                X_new = np.concatenate([X_new, pad], axis=2)

        predictions_list = []
        try:
            memory_watchdog(threshold_gb=15)
            for i in range(0, len(X_new), batch_size):
                end_i = min(i + batch_size, len(X_new))
                batch_input = X_new[i:end_i]
                try:
                    batch_pred = self.model.predict(batch_input, verbose=0)
                except (tf.errors.InternalError, tf.errors.ResourceExhaustedError):
                    with tf.device('/CPU:0'):
                        batch_pred = self.model.predict(batch_input, verbose=0)
                except Exception as e:
                    self.logger.error(f"Prediction batch error: {e}")
                    batch_pred = np.zeros((end_i - i, 5))
                    batch_pred[:, 2] = 1.0
                predictions_list.append(batch_pred)
                if i % (5 * batch_size) == 0:
                    memory_watchdog(threshold_gb=15)

            return np.vstack(predictions_list) if predictions_list else np.zeros((len(X_new), 5))

        except Exception as e:
            self.logger.error(f"Single-model prediction error: {e}")
            try:
                tf.keras.backend.clear_session()
                preds = []
                with tf.device('/CPU:0'):
                    for i in range(0, len(X_new), 8):
                        end_i = min(i + 8, len(X_new))
                        batch_pred = self.model.predict(X_new[i:end_i], verbose=0)
                        preds.append(batch_pred)
                return np.vstack(preds) if preds else np.zeros((len(X_new), 5))
            except Exception as cpu_error:
                self.logger.error(f"CPU fallback prediction failed: {cpu_error}")
                fallback = np.zeros((len(X_new), 5))
                fallback[:, 2] = 1.0
                return fallback

###############################################################################
# INTEGRATED FACADE
###############################################################################
class EnhancedCryptoModel:
    """Facade for simplified usage of the training and prediction pipeline."""

    def __init__(self, project_name="enhanced_crypto_model", max_trials=3, tuner_type="bayesian",
                 model_save_path="best_enhanced_model.keras", label_smoothing=0.1,
                 ensemble_size=2, use_mixed_precision=True):
        self.project_name = project_name
        self.max_trials = max_trials
        self.tuner_type = tuner_type
        self.model_save_path = model_save_path
        self.label_smoothing = label_smoothing
        self.ensemble_size = min(ensemble_size, 2)
        self.seed = 42

        if use_mixed_precision:
            try:
                tf.keras.mixed_precision.set_global_policy('mixed_float16')
                logger.info("Enabled mixed precision with 'mixed_float16'.")
            except Exception as e:
                logger.warning(f"Failed to set mixed precision: {e}")

        self.model_builder = ModelBuilder(output_classes=5, seed=self.seed)
        self.model_trainer = ModelTrainer(
            model_builder=self.model_builder,
            project_name=project_name,
            max_trials=max_trials,
            tuner_type=tuner_type,
            seed=self.seed
        )
        self.model_predictor = ModelPredictor(model_path=model_save_path)
        self.ensemble_models = []

    def tune_and_train(self, iteration, X_train, y_train, X_val, y_val,
                       df_val, fwd_returns_val, epochs=10, batch_size=64,
                       class_weight=None):
        logger.info(f"Starting model training for iteration {iteration}")
        adjusted_epochs = min(epochs, 8)
        adjusted_batch_size = min(batch_size, 64)

        best_model, best_hp = self.model_trainer.train_model(
            X_train, y_train, X_val, y_val, df_val, fwd_returns_val,
            adjusted_epochs, adjusted_batch_size, class_weight, self.model_save_path
        )
        return best_model, best_hp

    def build_ensemble(self, X_train, y_train, X_val, y_val, df_val, fwd_returns_val,
                       epochs=8, batch_size=64, class_weight=None):
        """
        This method is kept to avoid breaking any reference, but implementation
        can be project-specific if you want to actually create multiple models.
        """
        # If you previously had a separate logic for building multiple models,
        # you can reintroduce or keep it simple here. For now, it’s a placeholder.
        logger.info("No specialized ensemble build logic implemented; returning a single model reference.")
        return [(self.model_save_path, self.model_trainer.best_hp)]

    def predict_signals(self, X_new, batch_size=32):
        batch_size = min(batch_size, 32)
        return self.model_predictor.predict(X_new, batch_size, use_ensemble=False)

    def predict_with_ensemble(self, X_new, batch_size=32):
        batch_size = min(batch_size, 32)
        if not self.model_predictor.ensemble_paths:
            self.model_predictor.find_ensemble_models(
                self.model_save_path.replace('.keras', ''),
                self.ensemble_size
            )
        return self.model_predictor.predict(X_new, batch_size, use_ensemble=True)

    def load_best_model(self):
        loaded = self.model_predictor.load_model()
        return self.model_predictor.model if loaded else None

    def load_ensemble(self):
        self.model_predictor.find_ensemble_models(
            self.model_save_path.replace('.keras', ''),
            self.ensemble_size
        )
        success = self.model_predictor.load_ensemble()
        return self.model_predictor.ensemble_models if success else []

    def evaluate(self, X_val, y_val, batch_size=32):
        batch_size = min(batch_size, 32)
        if not self.model_predictor.model and not self.model_predictor.ensemble_models:
            self.load_best_model() or self.load_ensemble()

        if self.model_predictor.ensemble_models:
            logger.info("Evaluating ensemble model")
            preds, uncertainties = self.predict_with_ensemble(X_val, batch_size)
            y_true = np.argmax(y_val, axis=1)
            y_pred = np.argmax(preds, axis=1)
            logger.info(f"Classification Report:\n{classification_report(y_true, y_pred, digits=4)}")
            logger.info(f"Confusion Matrix:\n{confusion_matrix(y_true, y_pred)}")
            logger.info(f"Average uncertainty: {np.mean(uncertainties):.4f}")
        else:
            if self.model_predictor.model:
                # If you want a formal evaluation method, you could do something like:
                # result = self.model_predictor.model.evaluate(X_val, y_val, batch_size=batch_size)
                # logger.info(f"Evaluation result: {result}")
                # For now, we'll just do a quick classification report:
                preds = self.predict_signals(X_val, batch_size)
                y_true = np.argmax(y_val, axis=1)
                y_pred = np.argmax(preds, axis=1)
                logger.info(f"Classification Report:\n{classification_report(y_true, y_pred, digits=4)}")
                logger.info(f"Confusion Matrix:\n{confusion_matrix(y_true, y_pred)}")
            else:
                logger.warning("No model available for evaluation")
