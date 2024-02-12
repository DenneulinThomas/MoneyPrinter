# Connector for Hugging Face's model hub

import requests
import json
import os
import sys
import time
import logging
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import pipeline


# Install the transformers library


class HuggingFace:

    def __init__(self, model_name, task):
        self.model_name = model_name
        self.task = task
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.load_model()

    def load_model(self):
        try:
            if self.task == 'text-classification':
                print(f'Loading model {self.model_name} for text classification...')
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.pipeline = pipeline(
                    model="lxyuan/distilbert-base-multilingual-cased-sentiments-student",
                    return_all_scores=False
                )
            else:
                print('Invalid task type. Please choose from text-classification or text-generation.')
        except Exception as e:
            print(f'Error loading model: {e}')

    def predict(self, text):
        try:
            if self.task == 'text-classification':
                result = self.pipeline(text)
                return result
            elif self.task == 'text-generation':
                result = self.pipeline(text, max_length=50, num_return_sequences=1)
                return result
            else:
                print('Invalid task type. Please choose from text-classification or text-generation.')
        except Exception as e:
            print(f'Error predicting: {e}')
            return None
