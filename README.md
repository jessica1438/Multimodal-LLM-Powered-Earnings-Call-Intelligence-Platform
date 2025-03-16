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

<img width="848" alt="Screenshot 2025-03-16 at 1 12 33 PM" src="https://github.com/user-attachments/assets/26b4115b-b203-47ba-b3ec-2cf239e66a56" />
<img width="813" alt="Screenshot 2025-03-16 at 1 14 46 PM" src="https://github.com/user-attachments/assets/fd4667f2-7d94-4464-9c58-f65ed82ab8e5" />
<img width="851" alt="Screenshot 2025-03-16 at 1 14 56 PM" src="https://github.com/user-attachments/assets/e551cf87-b172-4d69-96af-aacc8b9416a6" />
<img width="826" alt="Screenshot 2025-03-16 at 1 15 23 PM" src="https://github.com/user-attachments/assets/d37f329e-6424-4f4d-a25a-c26f667931e7" />
<img width="811" alt="Screenshot 2025-03-16 at 1 15 48 PM" src="https://github.com/user-attachments/assets/4a5b73fb-303a-4761-8d2b-09f5bce097a0" />
<img width="807" alt="Screenshot 2025-03-16 at 1 16 03 PM" src="https://github.com/user-attachments/assets/67d3c1e9-a9a2-43f3-be7b-2a5b582a384a" />
<img width="834" alt="Screenshot 2025-03-16 at 1 16 15 PM" src="https://github.com/user-attachments/assets/a26c016f-b3c9-4e17-9db5-babf9e853d5f" />
<img width="830" alt="Screenshot 2025-03-16 at 1 16 46 PM" src="https://github.com/user-attachments/assets/6d22e684-eb51-4a85-a799-ebaa44961a2e" />
<img width="860" alt="Screenshot 2025-03-16 at 1 17 29 PM" src="https://github.com/user-attachments/assets/46afd8dc-e2f5-4374-93c7-862ff0308548" />
<img width="870" alt="Screenshot 2025-03-16 at 1 17 58 PM" src="https://github.com/user-attachments/assets/523d95f7-ec48-4fb9-8151-c549a992dfff" />
<img width="859" alt="Screenshot 2025-03-16 at 1 18 46 PM" src="https://github.com/user-attachments/assets/a58f6ec3-4fe0-4bca-9461-f926dae11f29" />
<img width="830" alt="Screenshot 2025-03-16 at 1 19 00 PM" src="https://github.com/user-attachments/assets/11ce7657-029d-497e-be92-0b15d0fbb5ea" />
<img width="885" alt="Screenshot 2025-03-16 at 1 19 15 PM" src="https://github.com/user-attachments/assets/b7300508-818a-4598-bf90-ae79eb79e509" />
<img width="851" alt="Screenshot 2025-03-16 at 1 19 31 PM" src="https://github.com/user-attachments/assets/a1cb0c5a-45cc-4573-9090-6bd22fcc6a59" />
<img width="848" alt="Screenshot 2025-03-16 at 1 19 39 PM" src="https://github.com/user-attachments/assets/e1f04213-3d19-4518-9a06-652d42d9139c" />
<img width="841" alt="Screenshot 2025-03-16 at 1 19 50 PM" src="https://github.com/user-attachments/assets/104bdd01-8be8-4595-b54d-9875830863f9" />


















