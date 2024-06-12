import torch
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration

tokenizer = BlenderbotTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
model = BlenderbotForConditionalGeneration.from_pretrained("facebook/blenderbot-400M-distill")

def blenderbot(text):
   input = tokenizer(text, return_tensors="pt")
   res = model.generate(**input)
   output = tokenizer.decode(res[0]).replace('<s>', '').replace('</s>', '')
   return output

