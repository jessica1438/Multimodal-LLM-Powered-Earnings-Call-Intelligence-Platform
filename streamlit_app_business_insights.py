# -*- coding: utf-8 -*-
"""Streamlit_App_Business_insights.ipynb
"""

!pip install langchain openai faiss-cpu youtube_transcript_api deep_translator
!pip install pytube moviepy torch torchvision torchaudio
!pip install fastapi uvicorn
!pip install transformers
!pip install yt-dlp
!pip install langchain-community langchain-core
!pip install openai-whisper
!pip install youtube_transcript_api pytube moviepy torch torchvision torchaudio
!pip install opencv-python transformers fastapi uvicorn
!pip install git+https://github.com/openai/whisper.git
!pip install vaderSentiment
!pip install streamlit pyngrok yt-dlp openai

!pip install spacy

pip install streamlit wordcloud matplotlib

import spacy
nlp = spacy.load("en_core_web_sm")

!wget -q -O - ipv4.icanhazip.com

Commented out IPython magic to ensure Python compatibility.
%%writefile app.py
# Load English NLP model

from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from transformers import pipeline
import torch
from openai import OpenAI
import yt_dlp
import cv2
from PIL import Image
import os
from transformers import BlipProcessor, BlipForConditionalGeneration
import whisper
import streamlit as st
from yt_dlp import YoutubeDL
import os
import openai
from pyngrok import ngrok
import smtplib
from email.mime.text import MIMEText
import pandas as pd
import plotly.express as px
from textblob import TextBlob
import spacy
import networkx as nx
import matplotlib.pyplot as plt
nlp = spacy.load("en_core_web_sm")
from spacy import displacy
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io
import mimetypes
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders


model_whisper = whisper.load_model("base")


processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model_blip = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

client = OpenAI(
		base_url = "https://qh5204dobmavfx9y.us-east-1.aws.endpoints.huggingface.cloud/v1/",
		api_key = "hf_XX"
	)

def download_video_yt_dlp(video_url, output_path="temp_video.mp4"):
    """Downloads the full video from YouTube using yt-dlp."""
    ydl_opts = {
        'format': 'bestvideo+bestaudio/best',
        'outtmpl': output_path,  # Save as MP4
        'merge_output_format': 'mp4'  # Ensure it's merged correctly
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])

    return output_path  # Return the downloaded video path

def extract_video_frames(video_path, output_dir="frames", frame_interval=5):
    """Extracts keyframes from a downloaded video at regular intervals."""

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    images = []

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % (frame_interval * 30) == 0:  # Extract every 5 seconds
            frame_path = os.path.join(output_dir, f"frame_{frame_count}.jpg")
            cv2.imwrite(frame_path, frame)
            images.append(Image.open(frame_path))
        frame_count += 1

    cap.release()
    os.remove(video_path)  # Delete the video after extracting frames

    return images


def image_captioning(folder_path):
    # Get list of all image files in the folder
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png', '.jpeg'))]

    # Dictionary to store captions
    results = {}

    # Process every 10th image
    for i in range(0, len(image_files), 5):  # Step of 10
        image_file = image_files[i]
        image_path = os.path.join(folder_path, image_file)

        # Load an image
        image = Image.open(image_path)

        # Process the image
        inputs = processor(images=image, return_tensors="pt")

        # Generate captions
        with torch.no_grad():
            outputs = model_blip.generate(**inputs)

        # Decode caption
        caption = processor.decode(outputs[0], skip_special_tokens=True)

        # Store in results dictionary
        results[image_file] = caption

    return results  # Return dictionary of filenames and captions



def download_audio_yt_dlp(video_url, output_path="temp_audio.mp4"):
    """Downloads only the audio from a YouTube video using yt-dlp."""
    ydl_opts = {
        'format': 'bestaudio/best',
        'extract_audio': True,
        'audio_format': 'mp3',
        'outtmpl': output_path,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])

    return output_path  # Return the downloaded audio path



def transcribe_audio(file_path):
    """Transcribes the audio file using OpenAI Whisper."""
    result = model_whisper.transcribe(file_path)
    return result["text"]  # Extract transcribed text



def format_input_data(captions_dict, transcription_text):
    """Formats multimodal data (captions + transcription) into a structured input for Mistral."""
    formatted_text = "### Video Summary Data\n\n"

    # Adding captions first
    formatted_text += "#### 🔹 Key Visual Elements:\n"
    for image, caption in captions_dict.items():
        formatted_text += f"- {image}: {caption}\n"

    # Adding transcriptions
    formatted_text += "\n#### 🔹 Spoken Content (Transcription):\n"
    formatted_text += transcription_text + "\n"

    return formatted_text



def choose_model(input, is_video):
     global EarningsCallSummary

     if is_video is True:


        download_video=download_video_yt_dlp(input)
        frames=extract_video_frames(download_video)
        folder_path= "/content/frames"
        image_caption=image_captioning(folder_path)
        download_audio=download_audio_yt_dlp(input)
        transcribe_audio_result=transcribe_audio(download_audio)
        format_data=format_input_data(image_caption, transcribe_audio_result)



        chat_completion = client.chat.completions.create(
        model="tgi",
        messages=[
        {"role": "user", "content": f"""Task: Analyze the content of the provided business earnings call video and
                                        summarize it in a structured format. The content includes both **spoken words** and **visual elements** from keyframes.
                                        Extract key financial insights, company performance, and strategic discussions.

                                        ### **Expected Output Format**:
                                        Ensure that the output follows this structured format:


                                        1.Company Overview
                                        [Summary of company introduction]
                                        [General sentiment of the earnings call]
                                        2.Key Financial Highlights
                                        Revenue: [Revenue details]
                                        Growth trends: [Trends and key financial metrics]
                                        Expenses: [Expense changes compared to previous periods]
                                        3.Market & Industry Trends
                                        [Mention major industry trends affecting the company]
                                        [Competitor insights, economic conditions]
                                        4.Future Outlook & Strategic Plans
                                        [Company's future plans, expansions, and strategies]
                                        [Management's expectations for the upcoming periods]
                                        5.Investor Sentiment & Market Reactions
                                        [CEO/CFO tone and confidence in the market]
                                        [Investor reactions, stock performance insights]
                                        \n\n

                                        Now generate the structured summary based on the input data.

                                        Input Data:
                                        {format_data}

                                        Output Summary:
                                        Provide a concise, factual summary based on the given information.
                                        DO NOT CONTINUE GENERATION AFTER GIVING THE OUTPUT.

            """}
        ],
        top_p=None,
        temperature=0,
        max_tokens=500,
        stream=False,
        seed=None,
        stop=None,
        frequency_penalty=None,
        presence_penalty=None
        )



     else:


        chat_completion = client.chat.completions.create(
        model="tgi",
        messages=[
        {"role": "user", "content": f"""Task: Analyze the content of the provided business earnings call Article and
                                        summarize it in a structured format. The content includes both **spoken words** and **visual elements** from keyframes.
                                        Extract key financial insights, company performance, and strategic discussions.

                                        ### **Expected Output Format**:
                                        Ensure that the output follows this structured format:


                                        1.Company Overview
                                        [Summary of company introduction]
                                        [General sentiment of the earnings call]
                                        2.Key Financial Highlights
                                        Revenue: [Revenue details]
                                        Growth trends: [Trends and key financial metrics]
                                        Expenses: [Expense changes compared to previous periods]
                                        3.Market & Industry Trends
                                        [Mention major industry trends affecting the company]
                                        [Competitor insights, economic conditions]
                                        4.Future Outlook & Strategic Plans
                                        [Company's future plans, expansions, and strategies]
                                        [Management's expectations for the upcoming periods]
                                        5.Investor Sentiment & Market Reactions
                                        [CEO/CFO tone and confidence in the market]
                                        [Investor reactions, stock performance insights]
                                        \n\n

                                        Now generate the structured summary based on the input data.

                                        Input Data:
                                        {input}

                                        Output Summary:
                                        Provide a concise, factual summary based on the given information.
                                        DO NOT CONTINUE GENERATION AFTER GIVING THE OUTPUT.

            """}
        ],
        	top_p=None,
          temperature=0,
          max_tokens=500,
          stream=False,
          seed=None,
          stop=None,
          frequency_penalty=None,
          presence_penalty=None
        )



     return (chat_completion.choices[0].message.content)


# UI Customization
st.set_page_config(page_title="AI Summarizer & Analysis", layout="centered")
st.title("📝 AI Corporate Insights Report")
st.markdown("🚀 **Decoding Business Earnings by Summarization and Visualization**")
st.divider()

# Ensure session state for summary
if "summary" not in st.session_state:
    st.session_state["summary"] = None

# User Input Options
option = st.radio("Select Input Type:", ("📺 Video", "📰 Text"))

if option == "📺 Video":
    video_url = st.text_input("🔗 Enter YouTube Video URL:")
    if st.button("🔍 Summarize Video"):
        with st.spinner("Generating summary..."):
            st.session_state["summary"] = choose_model(video_url, is_video=True)
        st.success("✅ Summary Generated!")
        st.markdown(f"<div class='summary-box'>{st.session_state['summary']}</div>", unsafe_allow_html=True)

elif option == "📰 Text":
    article_text = st.text_area("📄 Paste your text here:", height=200)
    if st.button("🔍 Summarize Text"):
        with st.spinner("Generating summary..."):
            st.session_state["summary"] = choose_model(article_text, is_video=False)
        st.success("✅ Summary Generated!")
        st.markdown(f"<div class='summary-box'>{st.session_state['summary']}</div>", unsafe_allow_html=True)




# Perform analysis only if summary exists
if st.session_state["summary"]:
    if st.button("📈 Analyze Entities and Words"):
        with st.spinner("Analyzing data..."):
            # Function to perform NER analysis
            def extract_entities(text):
                doc = nlp(text)
                entities = {"ORG": [], "MONEY": [], "DATE": [], "PERCENT": [], "GPE": []}
                entity_relations = []
                for ent in doc.ents:
                    if ent.label_ in entities:
                        entities[ent.label_].append(ent.text)
                        entity_relations.append((ent.text, ent.label_))
                return entities, entity_relations


            entities, entity_relations = extract_entities(st.session_state["summary"])
            df_entities = pd.DataFrame([(key, len(set(value))) for key, value in entities.items()], columns=["Entity Type", "Count"])
            st.subheader("🧬 Named Entity Recognition (NER)")
            st.metric(label="📌 Total Named Entities Identified", value=sum(df_entities["Count"]))
            fig = px.bar(df_entities, x="Entity Type", y="Count", color="Entity Type", text=df_entities["Count"], template="plotly_dark")
            fig.update_layout(title="🔍 Entity Distribution", xaxis_title="Entity Type", yaxis_title="Count")
            fig.update_traces(textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
            st.subheader("📌 Extracted Entities")
            for key, values in entities.items():
                st.write(f"**{key}:** {', '.join(set(values)) if values else 'None'}")
            st.subheader("📌 Entity Relationship Network")
            G = nx.Graph()
            color_map = {"ORG": "blue", "MONEY": "green", "DATE": "red", "PERCENT": "purple", "GPE": "orange"}
            node_colors = []
            node_category_map = {}
            for entity, label in entity_relations:
                G.add_node(entity)
                node_category_map[entity] = label
                node_colors.append(color_map.get(label, "gray"))
            for entity, label in entity_relations:
                G.add_edge(label, entity)
            plt.figure(figsize=(14, 8))
            pos = nx.spring_layout(G, k=0.8)
            node_color_map = [color_map.get(node_category_map.get(n, "gray"), "gray") for n in G.nodes]
            nx.draw(G, pos, with_labels=True, node_size=2500, node_color=node_color_map, edge_color="lightgray", font_size=10, font_weight="bold")
            st.pyplot(plt)
            st.subheader("📄 Annotated Text")
            html = displacy.render(nlp(st.session_state["summary"]), style="ent", jupyter=False)
            st.components.v1.html(html, height=500, scrolling=True)

            st.subheader("☁️ Word Cloud")

            # Generate the word cloud
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(st.session_state["summary"])

            # Display the word cloud using matplotlib
            fig, ax = plt.subplots()
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')

            # Render the matplotlib figure in Streamlit
            st.pyplot(fig)


# ---- Email Feature ----
send_email = st.checkbox("✉️ Send summary to your email?")

if send_email:
    email_id = st.text_input("📩 Enter your email:")

    if st.button("📤 Send Email"):
        if "summary" in st.session_state:  # ✅ Ensure summary exists before sending
            try:
                sender_email = "jessannajames@gmail.com"  # Replace with your Gmail
                sender_password = "rkiw qnkc wixb wgys"  # Use the working App Password

                subject = "Your AI Generated Business Insights"
                body = f"Here is your summary:\n\n{st.session_state['summary']}"  # ✅ Get summary from session state

                msg = MIMEText(body)
                msg["From"] = sender_email
                msg["To"] = email_id
                msg["Subject"] = subject

                # Connect to Gmail SMTP server
                server = smtplib.SMTP("smtp.gmail.com", 587)
                server.starttls()
                server.login(sender_email, sender_password)
                server.sendmail(sender_email, email_id, msg.as_string())
                server.quit()

                st.success(f"📧 Summary sent successfully to {email_id}!")
            except Exception as e:
                st.error(f"⚠️ Failed to send email: {e}")
        else:
            st.error("⚠️ Please generate a summary first before sending an email!")



!streamlit run app.py & npx localtunnel --port 8501