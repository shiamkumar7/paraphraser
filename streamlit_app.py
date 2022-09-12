
# Loading
# import numpy as np
# import pandas as pd
import torch


import joblib
import requests
import streamlit as st

st.title('Paraphraser Demo')
st.markdown('T5-Large model')

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

#Loading the t5-large model and the tokenizer from hugging face
model = AutoModelForSeq2SeqLM.from_pretrained("ramsrigouthamg/t5-large-paraphraser-diverse-high-quality")
tokenizer = AutoTokenizer.from_pretrained("ramsrigouthamg/t5-large-paraphraser-diverse-high-quality")


#t5-large customised code

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print ("device ",device)
model = model.to(device)


@st.cache(allow_output_mutation=True)
def t5_large_paraphraser(data,num_return_sequences):
  outputs = []
  df = pd.DataFrame(columns=['input','output'])
  df1 = df.copy()
  c= 0
  for text in data:
      input = "paraphrase: "+text + " </s>"
      encoding = tokenizer.encode_plus(input,max_length =128, padding=True, return_tensors="pt")
      input_ids,attention_mask  = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)

      model.eval()
      beam_outputs = model.generate(
          input_ids=input_ids,attention_mask=attention_mask,
          max_length=128,
          early_stopping=True,
          num_beams=15,
          num_return_sequences=num_return_sequences
      )
      for beam_output in beam_outputs:
          sent = tokenizer.decode(beam_output, skip_special_tokens=True,clean_up_tokenization_spaces=True)
          # outputs.append(sent)
          outputs.append(sent.split('paraphrasedoutput: ')[1])
  # return outputs
  st.write('Outputs')
  for op in outputs:
      st.success(op)


text = st.text_input('Enter input text')
num_return_seq = st.number_input('Enter number of return sequence to generate')


if st.button('Submit'):
    t5_large_paraphraser([text],num_return_seq)



