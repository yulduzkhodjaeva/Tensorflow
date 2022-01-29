import tensorflow as tf
import pandas as pd

csv_column_names = ['dir', 'numBytesSnt', 'minPktSz', 'stdIAT', 'ipMindIPID', 'ipMaxTTL', 'tcpPSeqCnt', 'tcpInitWinSz', 'tcpAveWinSz', 'tcpMSS', 'tcpWS', 'tcpRTTAckTripMax', 'entropy', 'class']
class_values = [0, 1]

train = pd.read_csv('UNB_GT_IDS_12000_14.csv', names=csv_column_names, header=0)
test = pd.read_csv('UNB_benign_attack_9000_14.csv', names=csv_column_names, header=0)

train_y = train.pop('class')
test_y = test.pop('class')


def input_fn(features, labels, training=True,batch_size=256):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    if training:
        dataset = dataset.shuffle(1000).repeat()
    return dataset.batch(batch_size)


my_feature_columns = []
for key in train.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))

classifier = tf.estimator.DNNClassifier(feature_columns=my_feature_columns, hidden_units=[30, 10], n_classes=2)
classifier.train(input_fn=lambda: input_fn(train, train_y, training=True), steps=5000)

eval_results = classifier.evaluate(input_fn=lambda: input_fn(test, test_y, training=False))

print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_results))
