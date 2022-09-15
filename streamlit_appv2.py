
# Loading
# import numpy as np
# import pandas as pd
import torch


# import joblib
import requests
import streamlit as st

st.title('Paraphraser Demo')
# st.header('T5-Large model')
st.markdown('A paraphraser is a model that generates different variations of same sentence without changing the intent of the sentence. The underlying model is a T5 transformer.')

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# #Loading the t5-large model and the tokenizer from hugging face
# model = AutoModelForSeq2SeqLM.from_pretrained("ramsrigouthamg/t5-large-paraphraser-diverse-high-quality")
# tokenizer = AutoTokenizer.from_pretrained("ramsrigouthamg/t5-large-paraphraser-diverse-high-quality")
#
#
# #t5-large customised code
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print ("device ",device)
# model = model.to(device)
@st.cache(allow_output_mutation=True)
def load_model():
    print('loading model')
    model = AutoModelForSeq2SeqLM.from_pretrained("ramsrigouthamg/t5-large-paraphraser-diverse-high-quality")
    tokenizer = AutoTokenizer.from_pretrained("ramsrigouthamg/t5-large-paraphraser-diverse-high-quality")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device ", device)
    model = model.to(device)
    print('model is loaded')
    return model,tokenizer,device

# @st.cache(allow_output_mutation=True,suppress_st_warning=True)
def t5_large_paraphraser(data,num_return_sequences):
  outputs = []
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

  for op in outputs:
      st.text(op)
  # st.success(outputs)

#adding new
menu = ['Upload a txt file','Single text']
choice = st.sidebar.selectbox("Menu",menu)

if choice =='Single text':
    st.subheader('Enter your input below')
    text = st.text_input('Enter input text')
    text = [text]
    num_return_seq = st.number_input('Enter number of return sequence to generate',min_value=1, max_value=5, value=3, step=1)
elif choice == 'Upload a txt file':
    st.subheader('Text File')
    txt_file = st.file_uploader('Upload your text file',type=['txt'])
    if txt_file is not None:
        text=[]
        text1 = txt_file.read().splitlines()
       #convering bytes to string
        for sent in text1:
            print(sent)
            text.append(sent.decode("utf-8"))
        print(text)
        num_return_seq = 1

st.write('Outputs')

if st.button('Submit'):
    model,tokenizer,device = load_model()
    t5_large_paraphraser(text,num_return_seq)



