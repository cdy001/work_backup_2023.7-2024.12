import tensorflow as tf
import sklearn
import os
import time
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix


def training_model(strategy, model, train_dataset, val_dataset, learn_rate, global_batch_size, epochs):
    train_loss = tf.keras.metrics.Mean()

    # loss and metrics
    with strategy.scope():
        #  train
        loss_object_t = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)

        def compute_loss(labels, predictions):
            per_example_loss = loss_object_t(labels, predictions)
            return tf.nn.compute_average_loss(per_example_loss, global_batch_size=global_batch_size)

        train_ac = tf.keras.metrics.Accuracy()

        # validation
        loss_object_v = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        val_loss = tf.keras.metrics.Mean()
        val_ac = tf.keras.metrics.Accuracy()

    # optimizers
    with strategy.scope():
        optimizer = tf.keras.optimizers.Adam(learning_rate=learn_rate)

    def train_step(x, y):
        # forward + backward + optimizer
        with tf.GradientTape() as tape:
            logits = model(x, training=True)
            loss_value = compute_loss(y, logits)
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        pre = tf.math.argmax(logits, axis=1)
        train_ac.update_state(y, pre)

        return loss_value

    def val_step(x, y):
        logits = model(x, training=False)
        loss_value = loss_object_v(y, logits)

        pre = tf.math.argmax(logits, axis=1)

        val_loss.update_state(loss_value)
        val_ac.update_state(y, pre)

    @tf.function
    def distributed_train_step(x, y):
        per_replica_losses = strategy.run(train_step, args=(x, y))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

    @tf.function
    def distributed_test_step(x, y):
        strategy.run(val_step, args=(x, y))

    def train_validation():
        # logs
        logdir_train = os.path.join("./logs/train/")
        logdir_val = os.path.join("./logs/validation/")
        writer_train = tf.summary.create_file_writer(logdir_train)
        writer_val = tf.summary.create_file_writer(logdir_val)

        latest_peak_ac = 0
        epoch_save = list(range(0, epochs, 10))
        for epoch in range(epochs):
            print("\nStart of epoch %d" % epoch)
            start_time = time.time()

            train_loss.reset_states()
            train_ac.reset_states()
            val_loss.reset_states()
            val_ac.reset_states()
            # train loop
            for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
                loss = distributed_train_step(x_batch_train, y_batch_train)
                train_loss.update_state(loss)
                if step % 10 == 0:
                    print("Epoch {:03d}: Step: {:04d},Loss: {:.4f},ac: {:04f} ".format(epoch, step,
                                                                                       train_loss.result(),
                                                                                       train_ac.result()))
            with writer_train.as_default():
                tf.summary.scalar('loss', train_loss.result(), step=epoch)
                tf.summary.scalar('ac', train_ac.result(), step=epoch)

            # test loop
            for x_batch_val, y_batch_val in val_dataset:
                distributed_test_step(x_batch_val, y_batch_val)

            print("\nEpoch {:03d}: Loss: {:.4f},ac: {:04f} ".format(epoch, val_loss.result(), val_ac.result()))
            with writer_val.as_default():
                tf.summary.scalar('loss', val_loss.result(), step=epoch)
                tf.summary.scalar('ac', val_ac.result(), step=epoch)

            # save model
            if epoch in epoch_save:
                model_save_path = os.path.join('./tmp', str(epoch) + '_epoch.h5')
                model.save(model_save_path)
            if val_ac.result() > latest_peak_ac:
                latest_peak_ac = val_ac.result()
                model.save("./tmp/best_model.h5")

            print("Time taken:%.2fs" % (time.time() - start_time))

    train_validation()


def testing_model(model, test_dataset):
    pre_labels = []
    test_labels = []
    for x_batch_test, y_batch_test in test_dataset:
        pred_pro = model(x_batch_test, training=False)
        pred_label = tf.math.argmax(pred_pro, axis=1)

        pre_labels.extend(pred_label)
        test_labels.extend(y_batch_test)

    accuracy = accuracy_score(test_labels, pre_labels)
    report = classification_report(test_labels, pre_labels)
    con = confusion_matrix(test_labels, pre_labels)
    print(accuracy)
    print(report)
    print(con)
