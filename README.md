## SpotLLM
Repository to store the ML model which predicts whether the text is generated by the LLM or the human itself.

Currently, <br>
I have tried two approach, one is based on the dataset which contains around 4.4L rows (having text and respective label)
1) **With pre-trained DistilBertClassifier**
2) **Zero-shot classifier from huggingface**: <br>
    -> facebook/bart-large-mnli (Using pipeline) <br>
    -> MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli <br>

My notebooks :  <br>
https://www.kaggle.com/code/jaypanchal14/spot6 <br>
https://colab.research.google.com/drive/1Dk2cUt08qFoJLLlldCz118OLceFn5UjF?usp=sharing <br>
https://www.kaggle.com/code/jaypanchal14/spotllm <br>

Darshit's Notebook : <br>
https://www.kaggle.com/code/darshit2582/fine-tuning-using-bert <br>
https://www.kaggle.com/code/darshit2582/logistic-regression <br>
https://www.kaggle.com/code/darshit2582/neural-network
