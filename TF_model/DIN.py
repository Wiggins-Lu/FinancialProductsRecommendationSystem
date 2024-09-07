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

# Config
RECENT_productS = 5  # userRatedproduct{1-5}
EMBEDDING_SIZE = 10

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
    'userRatedproduct2': tf.keras.layers.Input(name='userRatedproduct2', shape=(), dtype='int32'),
    'userRatedproduct3': tf.keras.layers.Input(name='userRatedproduct3', shape=(), dtype='int32'),
    'userRatedproduct4': tf.keras.layers.Input(name='userRatedproduct4', shape=(), dtype='int32'),
    'userRatedproduct5': tf.keras.layers.Input(name='userRatedproduct5', shape=(), dtype='int32'),

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
# product_col = tf.feature_column.categorical_column_with_identity(key='productId', num_buckets=1001)
# product_emb_col = tf.feature_column.embedding_column(product_col, EMBEDDING_SIZE)

# user id embedding feature
user_col = tf.feature_column.categorical_column_with_identity(key='userId', num_buckets=30001)
user_emb_col = tf.feature_column.embedding_column(user_col, EMBEDDING_SIZE)

# type features vocabulary
type_vocab = ['PropertyInsurance', 'LifeInsurance', 'MedicalInsurance', 'AccidentInsurance',
              'ShortTermBonds', 'LediumTermBonds', 'LongTermBonds',
              'StockFunds', 'BondFunds', 'MoneyMarketFunds', 'MixedFunds',
              'Gold', 'Silver', 'Platinum', 'Palladium',
              'LDW', 'LDZW', 'ZDLW']
# user type embedding feature
user_type_col = tf.feature_column.categorical_column_with_vocabulary_list(key="userType1",
                                                                           vocabulary_list=type_vocab)
user_type_emb_col = tf.feature_column.embedding_column(user_type_col, EMBEDDING_SIZE)
# item type embedding feature
item_type_col = tf.feature_column.categorical_column_with_vocabulary_list(key="productType1",
                                                                           vocabulary_list=type_vocab)
item_type_emb_col = tf.feature_column.embedding_column(item_type_col, EMBEDDING_SIZE)


'''
candidate_product_col = [tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_identity(key='productId', num_buckets=1001,default_value=0))]
recent_rate_col = [
    tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_identity(key='userRatedproduct1', num_buckets=1001,default_value=0)),
    tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_identity(key='userRatedproduct2', num_buckets=1001,default_value=0)),
    tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_identity(key='userRatedproduct3', num_buckets=1001,default_value=0)),
    tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_identity(key='userRatedproduct4', num_buckets=1001,default_value=0)),
    tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_identity(key='userRatedproduct5', num_buckets=1001,default_value=0)),
]
'''


candidate_product_col = [ tf.feature_column.numeric_column(key='productId', default_value=0),   ]

recent_rate_col = [
    tf.feature_column.numeric_column(key='userRatedproduct1', default_value=0),
    tf.feature_column.numeric_column(key='userRatedproduct2', default_value=0),
    tf.feature_column.numeric_column(key='userRatedproduct3', default_value=0),
    tf.feature_column.numeric_column(key='userRatedproduct4', default_value=0),
    tf.feature_column.numeric_column(key='userRatedproduct5', default_value=0),
]



# user profile
user_profile = [
    user_emb_col,
    user_type_emb_col,
    tf.feature_column.numeric_column('userRatingCount'),
    tf.feature_column.numeric_column('userAvgRating'),
    tf.feature_column.numeric_column('userRatingStddev'),
]

# context features
context_features = [
    item_type_emb_col,
    tf.feature_column.numeric_column('releaseYear'),
    tf.feature_column.numeric_column('productRatingCount'),
    tf.feature_column.numeric_column('productAvgRating'),
    tf.feature_column.numeric_column('productRatingStddev'),
]

candidate_layer = tf.keras.layers.DenseFeatures(candidate_product_col)(inputs)
user_behaviors_layer = tf.keras.layers.DenseFeatures(recent_rate_col)(inputs)
user_profile_layer = tf.keras.layers.DenseFeatures(user_profile)(inputs)
context_features_layer = tf.keras.layers.DenseFeatures(context_features)(inputs)

# Activation Unit

product_emb_layer = tf.keras.layers.Embedding(input_dim=1001,output_dim=EMBEDDING_SIZE,mask_zero=True)# mask zero

user_behaviors_emb_layer = product_emb_layer(user_behaviors_layer) 

candidate_emb_layer = product_emb_layer(candidate_layer) 
candidate_emb_layer = tf.squeeze(candidate_emb_layer,axis=1)

repeated_candidate_emb_layer = tf.keras.layers.RepeatVector(RECENT_productS)(candidate_emb_layer)

activation_sub_layer = tf.keras.layers.Subtract()([user_behaviors_emb_layer,
                                                   repeated_candidate_emb_layer])  # element-wise sub
activation_product_layer = tf.keras.layers.Multiply()([user_behaviors_emb_layer,
                                                       repeated_candidate_emb_layer])  # element-wise product

activation_all = tf.keras.layers.concatenate([activation_sub_layer, user_behaviors_emb_layer,
                                              repeated_candidate_emb_layer, activation_product_layer], axis=-1)

activation_unit = tf.keras.layers.Dense(32)(activation_all)
activation_unit = tf.keras.layers.PReLU()(activation_unit)
activation_unit = tf.keras.layers.Dense(1, activation='sigmoid')(activation_unit)
activation_unit = tf.keras.layers.Flatten()(activation_unit)
activation_unit = tf.keras.layers.RepeatVector(EMBEDDING_SIZE)(activation_unit)
activation_unit = tf.keras.layers.Permute((2, 1))(activation_unit)
activation_unit = tf.keras.layers.Multiply()([user_behaviors_emb_layer, activation_unit])

# sum pooling
user_behaviors_pooled_layers = tf.keras.layers.Lambda(lambda x: tf.keras.backend.sum(x, axis=1))(activation_unit)

# fc layer
concat_layer = tf.keras.layers.concatenate([user_profile_layer, user_behaviors_pooled_layers,
                                            candidate_emb_layer, context_features_layer])
output_layer = tf.keras.layers.Dense(128)(concat_layer)
output_layer = tf.keras.layers.PReLU()(output_layer)
output_layer = tf.keras.layers.Dense(64)(output_layer)
output_layer = tf.keras.layers.PReLU()(output_layer)
output_layer = tf.keras.layers.Dense(1, activation='sigmoid')(output_layer)

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
