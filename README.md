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
