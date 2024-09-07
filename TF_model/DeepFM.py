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

# define input for keras model
inputs = {
    'productAvgRating': tf.keras.layers.Input(name='productAvgRating', shape=(), dtype='float32'),
    'productRatingStddev': tf.keras.layers.Input(name='productRatingStddev', shape=(), dtype='float32'),
    'productRatingCount': tf.keras.layers.Input(name='productRatingCount', shape=(), dtype='int32'),
    'userAvgRating': tf.keras.layers.Input(name='userAvgRating', shape=(), dtype='float32'),
    'userRatingStddev': tf.keras.layers.Input(name='userRatingStddev', shape=(), dtype='float32'),
    'userRatingCount': tf.keras.layers.Input(name='userRatingCount', shape=(), dtype='int32'),
    'releaseYear': tf.keras.layers.Input(name='releaseYear', shape=(), dtype='int32'),

    'productId': tf.keras.layers.Input(name='productId', shape=(), dtype='int32'),
    'userId': tf.keras.layers.Input(name='userId', shape=(), dtype='int32'),
    'userRatedproduct1': tf.keras.layers.Input(name='userRatedproduct1', shape=(), dtype='int32'),

    'userType1': tf.keras.layers.Input(name='userType1', shape=(), dtype='string'),
    'userType2': tf.keras.layers.Input(name='userType2', shape=(), dtype='string'),
    'userType3': tf.keras.layers.Input(name='userType3', shape=(), dtype='string'),
    'userType4': tf.keras.layers.Input(name='userType4', shape=(), dtype='string'),
    'userType5': tf.keras.layers.Input(name='userType5', shape=(), dtype='string'),
    'productType1': tf.keras.layers.Input(name='productType1', shape=(), dtype='string'),
    'productType2': tf.keras.layers.Input(name='productType2', shape=(), dtype='string'),
    'productType3': tf.keras.layers.Input(name='productType3', shape=(), dtype='string'),
}

# product id embedding feature
product_col = tf.feature_column.categorical_column_with_identity(key='productId', num_buckets=1001)
product_emb_col = tf.feature_column.embedding_column(product_col, 10)
product_ind_col = tf.feature_column.indicator_column(product_col) # movid id indicator columns

# user id embedding feature
user_col = tf.feature_column.categorical_column_with_identity(key='userId', num_buckets=30001)
user_emb_col = tf.feature_column.embedding_column(user_col, 10)
user_ind_col = tf.feature_column.indicator_column(user_col) # user id indicator columns

# type features vocabulary
type_vocab = ['PropertyInsurance', 'LifeInsurance', 'MedicalInsurance', 'AccidentInsurance',
              'ShortTermBonds', 'LediumTermBonds', 'LongTermBonds',
              'StockFunds', 'BondFunds', 'MoneyMarketFunds', 'MixedFunds',
              'Gold', 'Silver', 'Platinum', 'Palladium',
              'LDW', 'LDZW', 'ZDLW']
# user type embedding feature
user_type_col = tf.feature_column.categorical_column_with_vocabulary_list(key="userType1",
                                                                           vocabulary_list=type_vocab)
user_type_emb_col = tf.feature_column.embedding_column(user_type_col, 10)
user_type_ind_col = tf.feature_column.indicator_column(user_type_col) # user type indicator columns
# item type embedding feature
item_type_col = tf.feature_column.categorical_column_with_vocabulary_list(key="productType1",
                                                                           vocabulary_list=type_vocab)
item_type_emb_col = tf.feature_column.embedding_column(item_type_col, 10)
item_type_ind_col = tf.feature_column.indicator_column(item_type_col) # item type indicator columns

# fm first-order term columns: without embedding and concatenate to the output layer directly
fm_first_order_columns = [product_ind_col, user_ind_col, user_type_ind_col, item_type_ind_col]

deep_feature_columns = [tf.feature_column.numeric_column('releaseYear'),
                        tf.feature_column.numeric_column('productRatingCount'),
                        tf.feature_column.numeric_column('productAvgRating'),
                        tf.feature_column.numeric_column('productRatingStddev'),
                        tf.feature_column.numeric_column('userRatingCount'),
                        tf.feature_column.numeric_column('userAvgRating'),
                        tf.feature_column.numeric_column('userRatingStddev'),
                        product_emb_col,
                        user_emb_col]

item_emb_layer = tf.keras.layers.DenseFeatures([product_emb_col])(inputs)
user_emb_layer = tf.keras.layers.DenseFeatures([user_emb_col])(inputs)
item_type_emb_layer = tf.keras.layers.DenseFeatures([item_type_emb_col])(inputs)
user_type_emb_layer = tf.keras.layers.DenseFeatures([user_type_emb_col])(inputs)

# The first-order term in the FM layer
fm_first_order_layer = tf.keras.layers.DenseFeatures(fm_first_order_columns)(inputs)

# FM part, cross different categorical feature embeddings
product_layer_item_user = tf.keras.layers.Dot(axes=1)([item_emb_layer, user_emb_layer])
product_layer_item_type_user_type = tf.keras.layers.Dot(axes=1)([item_type_emb_layer, user_type_emb_layer])
product_layer_item_type_user = tf.keras.layers.Dot(axes=1)([item_type_emb_layer, user_emb_layer])
product_layer_user_type_item = tf.keras.layers.Dot(axes=1)([item_emb_layer, user_type_emb_layer])

# deep part, MLP to generalize all input features
deep = tf.keras.layers.DenseFeatures(deep_feature_columns)(inputs)
deep = tf.keras.layers.Dense(64, activation='relu')(deep)
deep = tf.keras.layers.Dense(64, activation='relu')(deep)

# concatenate fm part and deep part
concat_layer = tf.keras.layers.concatenate([fm_first_order_layer, product_layer_item_user, product_layer_item_type_user_type,
                                            product_layer_item_type_user, product_layer_user_type_item, deep], axis=1)
output_layer = tf.keras.layers.Dense(1, activation='sigmoid')(concat_layer)

model = tf.keras.Model(inputs, output_layer)
# compile the model, set loss function, optimizer and evaluation metrics
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy', tf.keras.metrics.AUC(curve='ROC'), tf.keras.metrics.AUC(curve='PR')])

# train the model
model.fit(train_dataset, epochs=5)

# evaluate the model
test_loss, test_accuracy, test_roc_auc, test_pr_auc = model.evaluate(test_dataset)
print('\n\nTest Loss {}, Test Accuracy {}, Test ROC AUC {}, Test PR AUC {}'.format(test_loss, test_accuracy,
                                                                                   test_roc_auc, test_pr_auc))

# print some predict results
predictions = model.predict(test_dataset)
for prediction, goodRating in zip(predictions[:12], list(test_dataset)[0][1][:12]):
    print("Predicted good rating: {:.2%}".format(prediction[0]),
          " | Actual rating label: ",
          ("Good Rating" if bool(goodRating) else "Bad Rating"))