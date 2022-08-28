import ktrain

# import os #bypass GPU memory

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # use only CPU in prediction

predictor = ktrain.load_predictor('BERT_IMDB_Movie_Flask')

def get_prediction(x):
	sent = predictor.predict([x])
	return sent[0]