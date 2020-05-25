import sys
import pandas as pd
import nltk
import string 
import re
import numpy as np
from pprint import pprint
from nltk import tokenize
from nltk.corpus import stopwords 
import textcleaner as tc
from PyQt5.QtWidgets import  QInputDialog,QFileDialog
from nltk.tag import StanfordNERTagger
import mysql.connector
from PyQt5 import QtWidgets
from textAnalyzerDesing import Ui_MainWindow
from nltk import word_tokenize
import os
import datefinder
from itertools import groupby
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import speech_recognition as sr

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmatization
import spacy

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt
#%matplotlib inline

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
import os
from gensim.models.wrappers import LdaMallet


class TextAnalyzer(QtWidgets.QMainWindow):
    def __init__(self):
        super(TextAnalyzer,self).__init__()

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.btn_analyze.clicked.connect(self.btn_click)
        self.ui.btn_clear.clicked.connect(self.btn_clear)
        self.ui.btn_addfile.clicked.connect(self.add_file)
        self.ui.cb_uppercase.stateChanged.connect(self.redflag)
        self.ui.cb_lowercase.stateChanged.connect(self.redflag)
        self.ui.txt_autodetails.hide()
        self.ui.txt_manualdetails.hide()
        self.ui.txt_cleantext.hide()
        self.ui.txt_rawtext.textChanged.connect(self.clickable)
        self.ui.btn_voice.clicked.connect(self.add_voice)

        self.ui.btn_analyze.setEnabled(False)
    def clickable(self):
        a=self.ui.txt_rawtext.toPlainText()
        if a!="":
            self.ui.btn_analyze.setEnabled(True)
        else:
            self.ui.btn_analyze.setEnabled(False)
    def btn_click(self,value):
        self.ui.txt_cleantext.setText('')
        self.ui.txt_autodetails.setText('')
        self.ui.txt_manualdetails.setText('')
        checkboxes=self.ui.gb_detail.findChildren(QtWidgets.QCheckBox)
        self.ui.txt_autodetails.show()
        self.ui.txt_cleantext.show()
        stopwords.words('english')
        # grup boxtaki checkboxların hepsini items a atadı.
        text=str(self.ui.txt_rawtext.toPlainText())
        
        for cb in checkboxes: 
            if cb.isChecked():
                if cb.text()=="Lower Case":
                    self.ui.txt_cleantext.setText(text.lower())
                elif cb.text()=="Upper Case":
                    self.ui.txt_cleantext.setText(text.upper())
                elif cb.text()=="Extra Space Remover":
                    self.ui.txt_cleantext.setText(" ".join(text.split()))
                elif cb.text()=="Remove Punctuations":
                     self.ui.txt_cleantext.setText(text.translate(str.maketrans('', '', string.punctuation)))
                elif cb.text()=="Number Remover":
                     self.ui.txt_cleantext.setText(text.translate(str.maketrans('', '', string.digits)))
                elif cb.text()=="Stop Words Remover":
                    data = tc.document(text)
                    a=data.remove_stpwrds()
                    self.ui.txt_cleantext.setText(str(a))
                elif cb.text()=="Take Dates":
                    self.ui.txt_manualdetails.show()
                    dates=self.ui.txt_rawtext.toPlainText()
                    matches = datefinder.find_dates(dates)
                    dates_times=""
                    for match in matches:
                        dates_times=dates_times+str(match)+"\n"
                    if self.ui.txt_manualdetails.toPlainText()!="":
                        self.ui.txt_manualdetails.setText(self.ui.txt_manualdetails.toPlainText()+"\n"+"Dates"+"\n"+dates_times)
                    else:
                        self.ui.txt_manualdetails.setText("Dates"+"\n"+dates_times)
                elif cb.text()=="Emotion Analysis":
                    self.ui.txt_manualdetails.show()
                    sid = SentimentIntensityAnalyzer()
                    message_text =self.ui.txt_rawtext.toPlainText()
                    scores = sid.polarity_scores(message_text)
                    emotion=(scores.get('compound'))
                    
                    state=''
                    if (emotion < (-0.1)) & (emotion>=(-0.5)):
                        state='Negative'
                    elif (emotion < (-0.5)) :
                        state='So negative'
                    elif (emotion>=(-0.1)) & (emotion<=(0.1)):
                        state='Neutral'
                    elif (emotion>(0.1)) & (emotion<(0.5)):
                        state='Positive'
                    elif (emotion>=(0.5)):
                        state='So positive'
                    self.ui.txt_manualdetails.setText(self.ui.txt_manualdetails.toPlainText()+"\n"+"Emotion \n"+state)
                elif cb.text()=="Topic Analysis":
                    self.ui.txt_manualdetails.show()
                    take_subject(self)

        if self.ui.txt_cleantext.toPlainText()=='':
            self.ui.txt_cleantext.setText(text)
            
        text=self.ui.txt_cleantext.toPlainText()
        #Most common 5 word       
        self.ui.txt_cleantext.setText(text)
        b=self.ui.txt_rawtext.toPlainText()
        b=b.lower()
        data = tc.document(b)
        none_stop=data.remove_stpwrds()
        none_stop=str(none_stop)
        clean=(none_stop.translate(str.maketrans('', '', string.punctuation)))
        clean=tc.document(clean)
        most_common_five=pd.Series(" ".join(clean).split()).value_counts().nlargest(5)
        most_common_five_vector=most_common_five[0:len(most_common_five)]
        df=pd.DataFrame(most_common_five,columns=[""])
        self.ui.txt_autodetails.setText("Most Common Five Words"+str(df)+"\n")
       
        lines = self.ui.txt_rawtext.toPlainText()
        sentences = nltk.sent_tokenize(lines)
        verbs = [] #empty to array to hold all verbs

        for sentence in sentences:
            for verb,pos in nltk.pos_tag(nltk.word_tokenize(str(sentence))):
                if (pos == 'VB'  or pos == 'VBD'or pos=='VBG'or pos=='VBN' or pos=='VBZ' or pos=='VBP'):
                    verbs.append(verb)
        most_common_five_verb=(pd.Series(" ".join(verbs).split()).value_counts().nlargest(5))
        most_common_five_verb_vector=most_common_five_verb[0:len(most_common_five_verb)]
        df_verb=pd.DataFrame(most_common_five_verb,columns=[""])
        text_verb=self.ui.txt_autodetails.toPlainText()
        text_verb=text_verb+'\n'+"Most Common Five Verbs"+str(df_verb)+"\n"

        # Add the jar and model via their path (instead of setting environment variables):
        jar = 'C:/Users/zeyne/OneDrive/Masaüstü/stanford-ner-2018-10-16/stanford-ner.jar'
        model = 'C:/Users/zeyne/OneDrive/Masaüstü/stanford-ner-2018-10-16/classifiers/english.all.3class.distsim.crf.ser.gz'

        java_path = "C:/Program Files/Java/jre1.8.0_251/bin/java.exe"
        os.environ['JAVAHOME'] = java_path
        st = StanfordNERTagger(model_filename=model,path_to_jar=jar,encoding='utf-8')
        text = self.ui.txt_rawtext.toPlainText()
        tokenized_text = word_tokenize(text)
        classified_text = st.tag(tokenized_text)
        s=''           
        for tag, chunk in groupby(classified_text, lambda x:x[1]):
            if tag != "O":
                olcak="%-12s"%tag, " ".join(w for w, t in chunk)
                s=s+'\n'+str(olcak)
        a=(str(s).translate(str.maketrans('','', string.punctuation)))        
        text_verb=text_verb+'\n'+"Persons/Locations/Entities"+a
        self.ui.txt_autodetails.setText(text_verb)
        #self.database_save()
        
    def database_save(self):
        connection=mysql.connector.connect(
            host="localhost",
            user="root",
            password="zeynep.baykan95",
            database="text_analyzer"
        )
        cursor=connection.cursor()
        raw_text=self.ui.txt_rawtext.toPlainText()
        clean_text=self.ui.txt_cleantext.toPlainText()
        text=self.ui.txt_autodetails.toPlainText()
        self.ui.txt_autodetails.setText(" ".join(text.split()))
        details=self.ui.txt_autodetails.toPlainText()
         
        sql="INSERT INTO text(raw_text,clean_text,details) VALUES(%s,%s,%s)"
        values=(raw_text,clean_text,details)

        cursor.execute(sql,values)

        try:
            connection.commit()
        except mysql.connector.Error as err:
            print('Error',err)
        finally:
            connection.close()
     
    def redflag(self):
        item=self.sender()
        if item.text()=="Lower Case":
            self.ui.cb_uppercase.setChecked(False)
        elif item.text()=="Upper Case":
            self.ui.cb_lowercase.setChecked(False)
            
    def btn_clear(self):
        items=self.ui.gb_detail.findChildren(QtWidgets.QCheckBox)
        for i in items:
            i.setChecked(False)
        self.ui.txt_rawtext.setText('')
        self.ui.txt_cleantext.setText('')
        self.ui.txt_autodetails.setText('')
        self.ui.txt_manualdetails.setText('')
        self.ui.txt_autodetails.hide()
        self.ui.txt_manualdetails.hide()
        self.ui.txt_cleantext.hide()

    def add_file(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file', '/home')
        if fname[0]:
            f = open(fname[0], 'r')
            with f:
                data = f.read()
                self.ui.txt_rawtext.setText(data)
    def add_voice(self):
        def main():
            r=sr.Recognizer()

            with sr.Microphone() as source:
                r.adjust_for_ambient_noise(source)

                print("Say anything")

                audio=r.listen(source)


                try:
                    self.ui.txt_rawtext.setText(r.recognize_google(audio))

                except Exception as e:
                    print("Error:"+str(e))

        if __name__=="__main__":
            main() 
def take_subject(self):
    stop_words = stopwords.words('english')
    stop_words.extend(['from', 'subject', 're', 'edu', 'use'])
    text=self.ui.txt_rawtext.toPlainText()
    a=tokenize.sent_tokenize(text)
    v = pd.Series(a)
    text_vector=v[0:len(v)]
    df=pd.DataFrame(text_vector,columns=["content"])

    # Convert to list
    data = df.content.values.tolist()

    # Remove Emails
    data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]

    # Remove new line characters
    data = [re.sub('\s+', ' ', sent) for sent in data]

    # Remove distracting single quotes
    data = [re.sub("\'", "", sent) for sent in data]
        
    def sent_to_words(sentences):
        for sentence in sentences:
            yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

    data_words = list(sent_to_words(data))

    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  

    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)
        
    # Define functions for stopwords, bigrams, trigrams and lemmatization
    def remove_stopwords(texts):
        return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

    def make_bigrams(texts):
        return [bigram_mod[doc] for doc in texts]

    def make_trigrams(texts):
        return [trigram_mod[bigram_mod[doc]] for doc in texts]

    def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        """https://spacy.io/api/annotation"""
        texts_out = []
        for sent in texts:
            doc = nlp(" ".join(sent)) 
            texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        return texts_out
    # Remove Stop Words
    data_words_nostops = remove_stopwords(data_words)

    # Form Bigrams
    data_words_bigrams = make_bigrams(data_words_nostops)

    # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
    # python3 -m spacy download en
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

    # Do lemmatization keeping only noun, adj, vb, adv
    data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

    # Create Dictionary
    id2word = corpora.Dictionary(data_lemmatized)

    # Create Corpus
    texts = data_lemmatized

    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]
    
    # Download File: http://mallet.cs.umass.edu/dist/mallet-2.0.8.zip
    os.environ['MALLET_HOME'] = 'C:\\Mallet'
    mallet_path = 'C:/Mallet/bin/mallet' # update this path
    ldamallet = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=2, id2word=id2word)

    #Compute Coherence Score
    coherence_model_ldamallet = CoherenceModel(model=ldamallet, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
    coherence_ldamallet = coherence_model_ldamallet.get_coherence()

    #Select the model and print the topics
    optimal_model = ldamallet
    model_topics = optimal_model.show_topics(formatted=False)
        
    def format_topics_sentences(ldamodel=ldamallet ,corpus=corpus, texts=data):
    #Init output
        sent_topics_df = pd.DataFrame()

    #Get main topic in each document
        for i, row in enumerate(ldamodel[corpus]):
            row = sorted(row, key=lambda x: (x[1]), reverse=True)
            # Get the Dominant topic, Perc Contribution and Keywords for each document
            for j, (topic_num, prop_topic) in enumerate(row):
                if j == 0:  # => dominant topic
                    wp = ldamodel.show_topic(topic_num)
                    topic_keywords = ", ".join([word for word, prop in wp])
                    sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
                else:
                    break
        sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
        contents = pd.Series(texts)
        sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
        return(sent_topics_df)

    df_topic_sents_keywords = format_topics_sentences(ldamodel=optimal_model, corpus=corpus, texts=data)

    # Format
    df_dominant_topic = df_topic_sents_keywords.reset_index()
    df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

    # Group top 5 sentences under each topic
    sent_topics_sorteddf_mallet = pd.DataFrame()

    sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('Dominant_Topic')

    for i, grp in sent_topics_outdf_grpd:
        sent_topics_sorteddf_mallet = pd.concat([sent_topics_sorteddf_mallet, 
                                                grp.sort_values(['Perc_Contribution'], ascending=[0]).head(1)], 
                                                axis=0)

    # Reset Index    
    sent_topics_sorteddf_mallet.reset_index(drop=True, inplace=True)

    # Format
    sent_topics_sorteddf_mallet.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Text"]

    #Show
    sent_topics_sorteddf_mallet=sent_topics_sorteddf_mallet.sort_values('Topic_Perc_Contrib',ascending=True)
   
    subject=sent_topics_sorteddf_mallet["Keywords"].iloc[0]
    subject2=sent_topics_sorteddf_mallet["Keywords"].iloc[1]

    subject=subject.split(",")
    subject2=subject2.split(",")
    a=""
    
    for i in range(0,2):
        if i==0:
            a=a+"Probably : "+subject[i]+", "+subject2[i]+"\n"         
        else:
            a=a+"Might Be :"+subject[i]+","+subject2[i]+"\n"
    if self.ui.txt_manualdetails.toPlainText()!="":
        self.ui.txt_manualdetails.setText(self.ui.txt_manualdetails.toPlainText()+"\n"+"Topics \n"+a)
    else:
        self.ui.txt_manualdetails.setText("Topics \n"+a)
def app():
    app = QtWidgets.QApplication(sys.argv)
    window = TextAnalyzer()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    app()


