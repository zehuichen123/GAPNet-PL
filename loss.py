import tensorflow as tf

def f1(y_true, y_pred):
    eps = 1e-10
    tp = tf.reduce_sum(tf.cast(y_true*y_pred, dtype=tf.float32), 0)
    tn = tf.reduce_sum(tf.cast((1-y_true)*(1-y_pred), dtype=tf.float32), 0)
    fp = tf.reduce_sum(tf.cast((1-y_true)*y_pred, dtype=tf.float32), 0)
    fn = tf.reduce_sum(tf.cast(y_true*(1-y_pred), dtype=tf.float32), 0)

    p = tp / (tp + fp + eps)
    r = tp / (tp + fn + eps)

    f1 = 2 * p * r / (p + r + eps)
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return tf.reduce_mean(f1)

def f1_loss(y_true, y_pred):
    eps = 1e-10
    tp = tf.reduce_sum(tf.cast(y_true*y_pred, dtype=tf.float32), 0)
    tn = tf.reduce_sum(tf.cast((1-y_true)*(1-y_pred), dtype=tf.float32), 0)
    fp = tf.reduce_sum(tf.cast((1-y_true)*y_pred, dtype=tf.float32), 0)
    fn = tf.reduce_sum(tf.cast(y_true*(1-y_pred), dtype=tf.float32), 0)

    p = tp / (tp + fp + eps)
    r = tp / (tp + fn + eps)

    f1 = 2 * p * r / (p + r + eps)
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return 1 - tf.reduce_mean(f1)