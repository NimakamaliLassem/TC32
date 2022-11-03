import pandas as pd
import numpy as np
import tensorflow as tf
from transformers.models.bert import BertTokenizer
from transformers import TFBertModel
import streamlit as st
import pandas as pd
from transformers import TFAutoModel


hist_loss= [0.1971,0.0732,0.0465,0.0319,0.0232,0.0167,0.0127,0.0094,0.0073,0.0058,0.0049,0.0042]
hist_acc = [0.9508,0.9811,0.9878,0.9914,0.9936,0.9954,0.9965,0.9973,0.9978,0.9983,0.9986,0.9988]
hist_val_acc = [0.9804,0.9891,0.9927,0.9956,0.9981,0.998,0.9991,0.9997,0.9991,0.9998,0.9998,0.9998]
hist_val_loss = [0.0759,0.0454,0.028,0.015,0.0063,0.0064,0.004,0.0011,0.0021,0.00064548,0.0010,0.00042896]
Epochs = [i for i in range(1,13)]

hist_loss[:] = [x * 100 for x in hist_loss]
hist_acc[:] = [x * 100 for x in hist_acc]
hist_val_acc[:] = [x * 100 for x in hist_val_acc]
hist_val_loss[:] = [x * 100 for x in hist_val_loss]
d = {'val_acc':hist_val_acc, 'acc':hist_acc,'loss':hist_loss, 'val_loss':hist_val_loss, 'Epochs': Epochs}
chart_data = pd.DataFrame(d)
chart_data.index = range(1,13)

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def load_model(show_spinner=True):
    yorum_model = TFAutoModel.from_pretrained("NimaKL/TC32")
    tokenizer = BertTokenizer.from_pretrained('dbmdz/bert-base-turkish-128k-uncased')
    return yorum_model, tokenizer

st.set_page_config(layout='wide', initial_sidebar_state='expanded')
col1, col2= st.columns(2)
with col1:
    st.title("TC32 Multi-Class Text Classification")
    st.subheader('Model Loss and Accuracy')
    st.area_chart(chart_data, x = 'Epochs')
    yorum_model, tokenizer = load_model()



with col2:
    st.title("Sınıfı bulmak için bir şikayet girin.")
    st.subheader("Şikayet")
    text = st.text_area('',"Jandarma Genel Komutanlığı Alo 156 'yı Aradığımda Yardımcı Olunmadı!",label_visibility ='collapsed', height=240)
    aButton = st.button('Ara')

def prepare_data(input_text, tokenizer):
    token = tokenizer.encode_plus(
        input_text,
        max_length=256, 
        truncation=True, 
        padding='max_length', 
        add_special_tokens=True,
        return_tensors='tf'
    )
    return {
        'input_ids': tf.cast(token.input_ids, tf.float64),
        'attention_mask': tf.cast(token.attention_mask, tf.float64)
    }

def make_prediction(model, processed_data, classes=['Alışveriş','Anne-Bebek','Beyaz Eşya','Bilgisayar','Cep Telefonu','Eğitim','Elektronik','Emlak ve İnşaat','Enerji','Etkinlik ve Organizasyon','Finans','Gıda','Giyim','Hizmet','İçecek','İnternet','Kamu','Kargo-Nakliyat','Kozmetik','Küçük Ev Aletleri','Medya','Mekan ve Eğlence','Mobilya - Ev Tekstili','Mücevher Saat Gözlük','Mutfak Araç Gereç','Otomotiv','Sağlık','Sigorta','Spor','Temizlik','Turizm','Ulaşım']):
    probs = model.predict(processed_data)[0]
    return classes[np.argmax(probs)]
    
    
if text or aButton:
    with col2:
        with st.spinner('Wait for it...'):
            processed_data = prepare_data(text, tokenizer)
            result = make_prediction(yorum_model, processed_data=processed_data)
            description = '<table style="border: collapse;"><tr><div style="height: 62px;"></div></tr><tr><p style="border-width: medium; border-color: #aa5e70; border-radius: 10px;padding-top: 1px;padding-left: 20px;background:#20212a;font-family:Courier New; color: white;font-size: 36px; font-weight: boldest;">'+result+'</p></tr><table>'
            st.markdown(description, unsafe_allow_html=True)
    with col1:
        st.success("Tahmin başarıyla tamamlandı!")

