import tensorflow as tf

training_samples_file_path = tf.keras.utils.get_file("trainingSamples.csv",
                                                     "file:///Users/WigginsLu/Workspace/FinancialProductsRecSys/data/trainingSamples.csv")

test_samples_file_path = tf.keras.utils.get_file("testSamples.csv",
                                                 "file:///Users/WigginsLu/Workspace/FinancialProductsRecSys/data/testSamples.csv")


# load sample as tf dataset
def get_dataset(file_path):
    dataset = tf.data.experimental.make_csv_dataset(
        file_path,
        batch_size=12,
        label_name='label',
        na_value="0",
        num_epochs=1,
        ignore_errors=True)
    return dataset


# split as test dataset and training dataset
train_dataset = get_dataset(training_samples_file_path)
test_dataset = get_dataset(test_samples_file_path)

# type features vocabulary
type_vocab = ['PropertyInsurance', 'LifeInsurance', 'MedicalInsurance', 'AccidentInsurance',
              'ShortTermBonds', 'LediumTermBonds', 'LongTermBonds',
              'StockFunds', 'BondFunds', 'MoneyMarketFunds', 'MixedFunds',
              'Gold', 'Silver', 'Platinum', 'Palladium',
              'LDW', 'LDZW', 'ZDLW']

TYPE_FEATURES = {
    'userType1': type_vocab,
    'userType2': type_vocab,
    'userType3': type_vocab,
    'userType4': type_vocab,
    'userType5': type_vocab,
    'productType1': type_vocab,
    'productType2': type_vocab,
    'productType3': type_vocab
}

# all categorical features
categorical_columns = []
for feature, vocab in TYPE_FEATURES.items():
    cat_col = tf.feature_column.categorical_column_with_vocabulary_list(
        key=feature, vocabulary_list=vocab)
    emb_col = tf.feature_column.embedding_column(cat_col, 10)
    categorical_columns.append(emb_col)
# product id embedding feature
product_col = tf.feature_column.categorical_column_with_identity(
    key='productId', num_buckets=1001)
product_emb_col = tf.feature_column.embedding_column(product_col, 10)
categorical_columns.append(product_emb_col)

# user id embedding feature
user_col = tf.feature_column.categorical_column_with_identity(
    key='userId', num_buckets=30001)
user_emb_col = tf.feature_column.embedding_column(user_col, 10)
categorical_columns.append(user_emb_col)

# all numerical features
numerical_columns = [tf.feature_column.numeric_column('releaseYear'),
                     tf.feature_column.numeric_column('productRatingCount'),
                     tf.feature_column.numeric_column('productAvgRating'),
                     tf.feature_column.numeric_column('productRatingStddev'),
                     tf.feature_column.numeric_column('userRatingCount'),
                     tf.feature_column.numeric_column('userAvgRating'),
                     tf.feature_column.numeric_column('userRatingStddev')]

# embedding + MLP model architecture
model = tf.keras.Sequential([
    tf.keras.layers.DenseFeatures(numerical_columns + categorical_columns),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])

# compile the model, set loss function, optimizer and evaluation metrics
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy', tf.keras.metrics.AUC(curve='ROC'), tf.keras.metrics.AUC(curve='PR')])

# train the model
model.fit(train_dataset, epochs=5)

# evaluate the model
test_loss, test_accuracy, test_roc_auc, test_pr_auc = model.evaluate(
    test_dataset)
print('\n\nTest Loss {}, Test Accuracy {}, Test ROC AUC {}, Test PR AUC {}'.format(test_loss, test_accuracy,
                                                                                   test_roc_auc, test_pr_auc))

# print some predict results
predictions = model.predict(test_dataset)
for prediction, goodRating in zip(predictions[:12], list(test_dataset)[0][1][:12]):
    print("Predicted good rating: {:.2%}".format(prediction[0]),
          " | Actual rating label: ",
          ("Good Rating" if bool(goodRating) else "Bad Rating"))
