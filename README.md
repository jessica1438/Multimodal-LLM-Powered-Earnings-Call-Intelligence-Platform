# Multimodal-LLM-Powered-Earnings-Call-Intelligence-Platform
Developed a Streamlit web app for business earnings video analysis, integrating Whisper AI for transcription, CV2 for frame extraction, BLIP-2 for image captioning, and Mistral for summarization. Used NER for key entity extraction, mapped relationships with NetworkX, and visualized insights with word clouds, enabling data-driven decisions.

## Business Earnings Video Analysis Web Application – Detailed Explanation Overview

This project is a Streamlit-based web application designed to analyze business earnings videos by combining multiple AI techniques for transcription, image processing, summarization, and entity extraction. The goal is to extract actionable insights from video content, making it easier for businesses to interpret financial discussions efficiently.
Key Components & Workflow

### 1)Video Transcription using Whisper AI

Model Used: OpenAI’s Whisper AI (Automatic Speech Recognition - ASR). Purpose: Converts speech in earnings videos into text with high accuracy, enabling further analysis. Implementation: The video’s audio is extracted. Whisper processes the audio to generate a timestamped transcript. The transcript is stored for summarization and entity recognition.

### 2)Video Frame Extraction using CV2

Library Used: OpenCV (CV2) Purpose: Captures key video frames for image-based analysis. Implementation: Frames are extracted every 5 seconds to capture significant moments. The 10th frame is selected for image captioning using BLIP-2.

### 3)Image Captioning using BLIP-2 (Multimodal LLM)

Model Used: BLIP-2 (Bootstrapped Language-Image Pretraining) Purpose: Generates meaningful captions for the extracted frames, adding context to visual content. Implementation: Each frame is processed through BLIP-2 to produce captions. Captions provide insights into visual elements, such as graphs, charts, or speaker expressions.

### 4)Summarization using Mistral LLM

Model Used: Mistral (Efficient Transformer for Text Summarization) Purpose: Generates high-precision summaries of the transcribed text. Implementation: The transcript is fed into Mistral LLM. It produces a coherent, concise summary, highlighting key financial points.

### 5)Named Entity Recognition (NER) for Key Entity Extraction

Library Used: SpaCy or Transformers-based NER Purpose: Extracts important entities such as companies, stock names, earnings figures, and financial terms. Implementation: The summarization output is analyzed using an NER model. Entities are tagged and stored for further processing.

### 6)Relationship Mapping using NetworkX

Library Used: NetworkX Purpose: Creates a graph-based structure to map relationships between extracted entities. Implementation: Entities from the NER process are linked based on context (e.g., a company and its reported revenue). A network visualization is generated to show connections between key financial terms.

### 7)Insights Visualization using Word Clouds

Library Used: WordCloud Purpose: Provides an intuitive visual representation of frequently mentioned terms. Implementation: The summarized text is tokenized and analyzed for frequency. A word cloud is generated, emphasizing the most significant topics discussed in the earnings report. 

Final Outcome & Business Impact This application enables businesses and investors to quickly extract insights from earnings videos, helping them: 

✅ Understand key financial trends without watching lengthy videos. 

✅ Identify major stakeholders, companies, and financial terms using entity recognition. 

✅ Visualize connections between different financial entities through network mapping. 

✅ Extract image-based insights that complement textual analysis.

This AI-powered solution significantly enhances data-driven decision-making by converting unstructured video content into a structured, easy-to-analyze format. 🚀

<img width="841" alt="Screenshot 2025-03-16 at 1 19 50 PM" src="https://github.com/user-attachments/assets/a4dd135f-5ad3-48ef-a800-2242c5aea4fe" />
<img width="848" alt="Screenshot 2025-03-16 at 1 19 39 PM" src="https://github.com/user-attachments/assets/eb64a95b-9d5e-4678-a0f8-ba44444b557d" />
<img width="851" alt="Screenshot 2025-03-16 at 1 19 31 PM" src="https://github.com/user-attachments/assets/05c4daff-6168-493a-808c-8c5594200637" />
<img width="885" alt="Screenshot 2025-03-16 at 1 19 15 PM" src="https://github.com/user-attachments/assets/5cf9cba3-4888-4509-8134-bb3e63429e25" />
<img width="830" alt="Screenshot 2025-03-16 at 1 19 00 PM" src="https://github.com/user-attachments/assets/85155afd-b221-4d75-a108-525bd96351f7" />
<img width="859" alt="Screenshot 2025-03-16 at 1 18 46 PM" src="https://github.com/user-attachments/assets/b4e03389-5ecc-4dc0-a7df-33aa19e7efd4" />
<img width="870" alt="Screenshot 2025-03-16 at 1 17 58 PM" src="https://github.com/user-attachments/assets/70d3413b-e4df-466d-9207-a77c26157746" />
<img width="860" alt="Screenshot 2025-03-16 at 1 17 29 PM" src="https://github.com/user-attachments/assets/fe5677ae-30e2-4833-8bb3-3956e8328d9f" />
<img width="830" alt="Screenshot 2025-03-16 at 1 16 46 PM" src="https://github.com/user-attachments/assets/f5c035c9-6a11-4a3b-bc4e-6b6285d09635" />
<img width="834" alt="Screenshot 2025-03-16 at 1 16 15 PM" src="https://github.com/user-attachments/assets/b3911938-36f4-44a9-88cd-fb13f54934a4" />
<img width="807" alt="Screenshot 2025-03-16 at 1 16 03 PM" src="https://github.com/user-attachments/assets/a319d534-1263-48af-861a-abe85fdd9752" />
<img width="811" alt="Screenshot 2025-03-16 at 1 15 48 PM" src="https://github.com/user-attachments/assets/e4d32dff-d052-47bf-9976-741fd8761254" />
<img width="826" alt="Screenshot 2025-03-16 at 1 15 23 PM" src="https://github.com/user-attachments/assets/5ac9b413-8f9c-4054-9acb-75bb0d984e03" />
<img width="851" alt="Screenshot 2025-03-16 at 1 14 56 PM" src="https://github.com/user-attachments/assets/c11c9485-4926-449f-ab77-2432f0d47e9d" />
<img width="813" alt="Screenshot 2025-03-16 at 1 14 46 PM" src="https://github.com/user-attachments/assets/39e60f00-f78b-4c78-af7a-036900bb1ce6" />
<img width="848" alt="Screenshot 2025-03-16 at 1 12 33 PM" src="https://github.com/user-attachments/assets/430abb8e-ad5a-487a-8313-e26b22b99934" />



