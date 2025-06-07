# talent-scout-chatbot 
Overview
TalentScout is an AI-powered hiring assistant designed to streamline the technical candidate screening process for recruitment agencies. This application leverages the Groq API with LLaMA 3.3 70B model to conduct interactive interviews, collect candidate information, and assess technical skills through tailored questions.

Key Features
🌍 Multilingual Support
Conduct interviews in English, Hindi, or Spanish

Dynamic language switching without losing context

📝 Candidate Information Collection
Structured collection of:

Personal details (name, email, phone)

Professional information (experience, desired position)

Technical specifications (tech stack, skills)

💻 Technical Assessment
Generates tailored technical questions based on:

Candidate's specified tech stack

Years of experience

Desired position

Multiple question types:

Conceptual questions

Coding challenges

Multiple-choice questions (MCQs)

🧠 AI-Powered Interaction
Conversation memory and context retention

Sentiment analysis of candidate responses

Adaptive questioning based on candidate inputs

🎨 Professional UI
Clean, responsive interface

Custom styling for different message types

Clear section formatting for technical questions

Technical Stack
Core Technologies
Streamlit: Web application framework

LangChain: LLM orchestration and prompt management

Groq API: High-performance LLM inference

TextBlob: Sentiment analysis

Supporting Libraries
Python-dotenv: Environment variable management

Regex: Text processing and cleaning

Logging: Robust error handling and debugging

Installation
Prerequisites
Python 3.8+

Groq API key (register at groq.com)

.env file with your API key:

text
GROQ_API_KEY=your_api_key_here
Setup
Clone the repository:

bash
git clone https://github.com/your-repo/talentscout.git
cd talentscout
Create and activate a virtual environment:

bash
python -m venv venv
source venv/bin/activate  # Linux/MacOS
venv\Scripts\activate    # Windows
Install dependencies:

bash
pip install -r requirements.txt
Run the application:

bash
streamlit run app.py
Usage Guide
Starting the Interview
Select your preferred language from the dropdown

The AI will greet you and begin the screening process

Interview Flow
Information Collection:

Provide your personal and professional details when prompted

Be specific about your tech stack (e.g., "Python, TensorFlow, PostgreSQL")

Technical Assessment:

Answer the generated technical questions

For coding questions, provide pseudocode or describe your approach

For MCQs, simply state the letter of your answer

Completion:

The interview concludes with a summary of your information

You can type "exit" or "quit" at any time to end early

Tips for Best Experience
Be specific about your skills and experience

Provide complete answers for better assessment

The system understands natural language - you don't need perfectly formatted responses

Architecture
Main Components
Conversation Manager:

Handles the interview state machine

Manages transitions between information collection and technical assessment

Prompt Templates:

Greeting and information collection prompts

Technical question generation templates

Error handling and fallback prompts

Memory System:

ConversationBufferMemory maintains context

Session state preserves candidate data

Question Generator:

Creates tailored questions based on tech stack

Formats questions with appropriate difficulty

Error Handling
Automatic retries for API rate limits

Comprehensive logging to debug.log

Fallback question system if generation fails

Customization
Modifying Questions
Edit the tech_prompt template in the code to:

Change question styles

Add new question types

Adjust difficulty levels

Adding Languages
Add new language options to the dropdown

Create translated versions of the prompt templates

Ensure the LLM supports the target language

Styling Changes
Modify the custom CSS in the st.markdown() section to:

Change colors and layouts

Add new UI elements

Adjust responsive behavior

Troubleshooting
Common Issues
API Errors:

Verify your GROQ_API_KEY is set correctly

Check the Groq status page for outages

Question Generation Problems:

Be specific about your tech stack

The system works best with well-known technologies

Language Issues:

Some languages may have limited support

English typically provides the most reliable results

Debugging
Check debug.log for detailed error information

Enable verbose logging by changing logging.basicConfig(level=logging.INFO)
