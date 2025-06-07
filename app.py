import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from textblob import TextBlob
import os
from dotenv import load_dotenv
from groq import RateLimitError
import time
import logging
import traceback
import re

# Setup logging
logging.basicConfig(
    level=logging.WARNING,  # Production-level logging
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('debug.log'),  # Log to file
        logging.StreamHandler()  # Keep console output for critical errors
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize LangChain components
llm = ChatGroq(model_name="llama-3.3-70b-versatile", api_key=os.environ.get("GROQ_API_KEY"))
memory = ConversationBufferMemory(return_messages=True)

# Define prompts with language support
greeting_prompt = PromptTemplate(
    input_variables=["history", "language"],
    template="""
    You are TalentScout, an AI-powered Hiring Assistant for a tech recruitment agency. Greet the candidate warmly in {language}, introduce yourself, and explain that you will collect their details and ask technical questions based on their tech stack to assess their fit for tech roles. Keep the tone professional and friendly. Start by asking for their full name.
    Conversation history: {history}
    """
)

info_prompt = PromptTemplate(
    input_variables=["history", "current_field", "language"],
    template="""
    You are collecting candidate details for a tech recruitment agency in {language}. Ask for one field at a time: Full Name, Email Address, Phone Number, Years of Experience, Desired Position(s), Current Location, Tech Stack. Current field to collect: {current_field}. Be conversational and polite. If the input is unclear, ask for clarification. Use the conversation history: {history}
    """
)

tech_prompt = PromptTemplate(
    input_variables=["history", "tech_stack", "experience", "language"],
    template="""
    As a senior technical interviewer, generate specific technical questions in {language} 
    to assess a candidate who knows: {tech_stack} (with {experience} years experience).
    
    Format the questions by technology using emojis and clear sections. For example:

    🤖 Deep Learning Questions:
    1. [Concept] What is the difference between Batch Normalization and Dropout? When should you use each?
    
    2. [Coding] Implement a simple feedforward neural network in PyTorch to classify MNIST digits.
    
    3. [MCQ] Which activation function is most suitable for binary classification?
    A) ReLU
    B) Sigmoid
    C) Tanh
    D) Softmax
    Correct: B
    
    🐍 Python Questions:
    1. [Coding] Write a Python function that returns the frequency count of all characters in a string.
    
    2. [Concept] Explain the difference between deepcopy() and copy() with a practical example.

    Generate questions for: {tech_stack}
    Keep questions clear, practical, and appropriate for their {experience} years of experience.
    Conversation history: {history}
    """
)

fallback_prompt = PromptTemplate(
    input_variables=["history", "user_input", "language"],
    template="""
    The candidate provided an unclear or irrelevant response: {user_input}. Politely ask for clarification in {language} while staying on topic. Do not deviate from the hiring assistant's purpose. Use the conversation history: {history}
    """
)

end_prompt = PromptTemplate(
    input_variables=["history", "candidate_data", "language"],
    template="""
    Thank the candidate for their time in {language}, summarize their details: {candidate_data}, and inform them that their information will be reviewed by the TalentScout team. Provide a professional farewell and wish them luck.
    Conversation history: {history}
    """
)

# Custom CSS for UI enhancement
st.markdown("""
    <style>
    .stChatMessage { 
        background-color: #f0f2f6; 
        padding: 15px; 
        border-radius: 10px; 
        margin-bottom: 10px; 
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stChatMessage.user { 
        background-color: #d1e7dd; 
        border: 1px solid #badbcc;
    }
    .stChatMessage.assistant { 
        background-color: #e9ecef; 
        border: 1px solid #dee2e6;
    }
    .stTitle { 
        color: #1a3c6e; 
        font-weight: bold; 
    }
    .stMarkdown { 
        font-family: 'Arial', sans-serif; 
    }
    .sentiment { 
        font-style: italic; 
        color: #6b7280; 
        font-size: 0.9em; 
    }
    .mcq-option { 
        margin-left: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Streamlit UI
st.title("TalentScout Hiring Assistant")
st.write("Welcome to the TalentScout AI Chatbot! Select your preferred language and begin the screening process.")

# Language selection
language = st.selectbox("Select Language / भाषा चुनें / Seleccione el idioma", ["English", "Hindi", "Spanish"])

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "candidate_data" not in st.session_state:
    st.session_state.candidate_data = {
        "name": None, "email": None, "phone": None, "experience": None,
        "position": None, "location": None, "tech_stack": None
    }
if "current_step" not in st.session_state:
    st.session_state.current_step = "greeting"
if "technical_questions" not in st.session_state:
    st.session_state.technical_questions = []
if "current_question_index" not in st.session_state:
    st.session_state.current_question_index = 0
if "language" not in st.session_state:
    st.session_state.language = language

# Update language if changed
if st.session_state.language != language:
    st.session_state.language = language
    st.session_state.messages = []  # Reset conversation for new language
    st.session_state.current_step = "greeting"
    memory.clear()

# Display conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # Format MCQ options if detected
        content = message["content"]
        if "[MCQ]" in content:
            parts = content.split("\n")
            question_part = parts[0]
            options_part = "\n".join([f'<p class="mcq-option">{opt}</p>' for opt in parts[1:] if opt.strip()])
            content = f"{question_part}\n{options_part}"
        st.markdown(content, unsafe_allow_html=True)
        if message["role"] == "user" and "sentiment" in message:
            st.markdown(f'<p class="sentiment">{message["sentiment"]}</p>', unsafe_allow_html=True)

def invoke_llm(prompt):
    """
    Invoke the LLM with retry logic and debug output cleaning.
    """
    max_retries = 3
    retry_delay = 1
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Attempt {attempt + 1}: Invoking LLM")
            response = llm.invoke(prompt).content
            
            # Clean debug output and normalize formatting
            cleaned_lines = []
            for line in response.split('\n'):
                # Skip debug/log lines and empty lines
                if not any(line.startswith(prefix) for prefix in ['DEBUG:', 'INFO:', 'WARNING:', 'ERROR:']):
                    cleaned_line = line.strip()
                    if cleaned_line:
                        cleaned_lines.append(cleaned_line)
            
            cleaned_response = '\n'.join(cleaned_lines)
            logger.info("LLM response successfully cleaned")
            return cleaned_response
            
        except RateLimitError:
            wait_time = retry_delay * (attempt + 1)
            logger.warning(f"RateLimitError occurred, retrying in {wait_time} seconds...")
            time.sleep(wait_time)
        except Exception as e:
            logger.error(f"Error invoking LLM: {str(e)}")
            if attempt == max_retries - 1:
                return ""
            time.sleep(retry_delay)
    return ""

def generate_tech_questions(tech_input, experience, language):
    """
    Generate technical questions based on the candidate's tech stack and experience.
    Returns a list of properly formatted questions.
    """
    try:
        # Clean and extract technologies
        tech_list = [t.strip() for t in re.split(r'[,/&]', tech_input) if t.strip()][:3]
        primary_tech = ', '.join(tech_list)
        
        # Get raw response from LLM
        response = invoke_llm(tech_prompt.format(
            history=memory.load_memory_variables({})["history"],
            tech_stack=primary_tech,
            experience=experience,
            language=language
        ))
        
        if not response.strip():
            logger.warning("Empty response from LLM, falling back to default questions")
            return get_fallback_questions(tech_list, experience, language)
            
        # Parse and format questions by section
        sections = []
        current_section = []
        current_section_title = None
        
        for line in response.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            # Check for new section (emoji + title)
            if re.match(r'^[^\w\s].*Questions:', line):
                if current_section_title and current_section:
                    sections.append(f"{current_section_title}\n" + "\n".join(current_section))
                current_section_title = line
                current_section = []
            else:
                current_section.append(line)
        
        # Add the last section
        if current_section_title and current_section:
            sections.append(f"{current_section_title}\n" + "\n".join(current_section))
        
        # If no proper sections were found, format as a single section
        if not sections:
            if current_section:
                sections = ["\n".join(current_section)]
            else:
                return get_fallback_questions(tech_list, experience, language)
        
        return sections
        
    except Exception as e:
        logger.error(f"Question generation failed: {str(e)}")
        return get_fallback_questions(tech_list, experience, language)

def get_fallback_questions(tech_list, experience, language):
    """Generate fallback questions with proper formatting."""
    questions = []
    exp_level = "junior" if int(experience or 0) < 3 else "senior" if int(experience or 0) < 7 else "expert"
    
    for tech in tech_list[:3]:
        emoji = "🤖" if tech.lower() in ["deep learning", "machine learning", "ai"] else \
               "🐍" if tech.lower() == "python" else \
               "☕" if tech.lower() == "java" else \
               "🌐" if tech.lower() in ["javascript", "typescript"] else \
               "🗄️" if tech.lower() in ["postgresql", "mysql", "mongodb"] else \
               "⚙️"
        
        section = [f"{emoji} {tech.title()} Questions:"]
        
        if tech.lower() in ['python', 'java', 'javascript', 'typescript']:
            section.extend([
                f"1. [Concept] Explain the event loop in {tech}",
                f"2. [Coding] Write a {tech} function that {'demonstrates polymorphism' if exp_level != 'junior' else 'reverses a string'}",
                f"3. [MCQ] Which feature is NOT built into {tech}?\nA) Exception handling\nB) Memory management\nC) Automatic garbage collection\nD) Manual memory allocation\nCorrect: D"
            ])
        elif tech.lower() in ['deep learning', 'machine learning', 'ai']:
            section.extend([
                "1. [Concept] Explain the difference between supervised and unsupervised learning",
                "2. [Coding] Implement a simple neural network using any deep learning framework",
                "3. [MCQ] Which activation function is best for hidden layers?\nA) Sigmoid\nB) ReLU\nC) Linear\nD) Softmax\nCorrect: B"
            ])
        else:
            section.extend([
                f"1. [Concept] What are the core features of {tech}?",
                f"2. [Coding] Implement a basic {tech} {'microservice' if exp_level != 'junior' else 'component'}",
                f"3. [MCQ] Which is a best practice in {tech}?\nA) Regular testing\nB) No documentation\nC) Monolithic design\nD) Tight coupling\nCorrect: A"
            ])
        
        questions.append("\n".join(section))
    
    return questions

def format_question(index: int, question: str) -> str:
    """
    Format a question with proper numbering and cleaning.
    Ensures consistent formatting for all question types.
    """
    # Remove any debug output and normalize whitespace
    clean_q = re.sub(r'DEBUG:.*?\n', '', question, flags=re.MULTILINE).strip()
    
    # Extract question type if present
    question_type = None
    for marker in ['[MCQ]', '[Coding]', '[Concept]', '[Scenario]']:
        if marker in clean_q:
            question_type = marker
            break
    
    # If no type marker found, add [Concept] as default
    if not question_type:
        clean_q = f"[Concept] {clean_q}"
    
    # Ensure proper numbering
    if not clean_q.startswith(f"{index}."):
        # Remove existing number if present
        clean_q = re.sub(r'^\d+\.\s*', '', clean_q)
        clean_q = f"{index}. {clean_q}"
    
    # Special formatting for MCQ questions
    if '[MCQ]' in clean_q:
        parts = clean_q.split('\n')
        question_part = parts[0]
        options_part = []
        correct_answer = None
        
        for part in parts[1:]:
            part = part.strip()
            if part.startswith(('A)', 'B)', 'C)', 'D)')):
                options_part.append(f"  {part}")
            elif part.lower().startswith('correct:'):
                correct_answer = part
        
        # Ensure all parts are present
        if options_part and correct_answer:
            clean_q = f"{question_part}\n" + '\n'.join(options_part) + f"\n{correct_answer}"
        
    return clean_q

# Handle user input
if user_input := st.chat_input("Your response / आपका जवाब / Su respuesta:"):
    # Sentiment analysis
    sentiment = TextBlob(user_input).sentiment.polarity
    sentiment_text = (
        "You seem confident!" if sentiment > 0.2 else
        "Let's dive deeper into this!" if sentiment < -0.2 else
        "Thanks for your response!"
    )

    # Save user input with sentiment
    st.session_state.messages.append({"role": "user", "content": user_input, "sentiment": sentiment_text})
    memory.save_context({"input": user_input}, {"output": ""})

    # Get conversation history
    history = memory.load_memory_variables({})["history"]

    # Check for conversation-ending keywords
    if any(keyword in user_input.lower() for keyword in ["exit", "quit", "done"]):
        response = invoke_llm(end_prompt.format(history=history, candidate_data=st.session_state.candidate_data, language=language))
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.current_step = "ended"
    else:
        # Handle conversation steps
        if st.session_state.current_step == "greeting":
            response = invoke_llm(greeting_prompt.format(history=history, language=language))
            st.session_state.current_step = "name"
        elif st.session_state.current_step == "name":
            st.session_state.candidate_data["name"] = user_input
            response = invoke_llm(info_prompt.format(history=history, current_field="Email Address", language=language))
            st.session_state.current_step = "email"
        elif st.session_state.current_step == "email":
            st.session_state.candidate_data["email"] = user_input
            response = invoke_llm(info_prompt.format(history=history, current_field="Phone Number", language=language))
            st.session_state.current_step = "phone"
        elif st.session_state.current_step == "phone":
            st.session_state.candidate_data["phone"] = user_input
            response = invoke_llm(info_prompt.format(history=history, current_field="Years of Experience", language=language))
            st.session_state.current_step = "experience"
        elif st.session_state.current_step == "experience":
            st.session_state.candidate_data["experience"] = user_input
            response = invoke_llm(info_prompt.format(history=history, current_field="Desired Position(s)", language=language))
            st.session_state.current_step = "position"
        elif st.session_state.current_step == "position":
            st.session_state.candidate_data["position"] = user_input
            response = invoke_llm(info_prompt.format(history=history, current_field="Current Location", language=language))
            st.session_state.current_step = "location"
        elif st.session_state.current_step == "location":
            st.session_state.candidate_data["location"] = user_input
            response = invoke_llm(info_prompt.format(history=history, current_field="Tech Stack (programming languages, frameworks, databases, tools)", language=language))
            st.session_state.current_step = "tech_stack"
        elif st.session_state.current_step == "tech_stack":
            user_input = user_input.strip()
            if not user_input:
                response = "Please specify at least one technology (e.g., 'Python, Deep Learning')"
            else:
                st.session_state.candidate_data["tech_stack"] = user_input
                sections = generate_tech_questions(
                    user_input,
                    st.session_state.candidate_data["experience"],
                    language
                )
                
                if sections:
                    st.session_state.technical_questions = sections
                    st.session_state.current_step = "questions"
                    st.session_state.current_question_index = 0
                    
                    # Show the first section initially
                    response = sections[0]
                else:
                    # Fallback to default questions
                    fallback_sections = get_fallback_questions(
                        user_input.split(',')[:3],
                        st.session_state.candidate_data["experience"],
                        language
                    )
                    st.session_state.technical_questions = fallback_sections
                    st.session_state.current_step = "questions"
                    st.session_state.current_question_index = 0
                    
                    response = fallback_sections[0] if fallback_sections else "Let's proceed with some general questions."
        elif st.session_state.current_step == "questions":
            if (st.session_state.technical_questions and 
                st.session_state.current_question_index < len(st.session_state.technical_questions) - 1):
                st.session_state.current_question_index += 1
                response = st.session_state.technical_questions[st.session_state.current_question_index]
            else:
                response = invoke_llm(end_prompt.format(
                    history=history,
                    candidate_data=st.session_state.candidate_data,
                    language=language
                ))
                st.session_state.current_step = "ended"
        else:
            response = invoke_llm(fallback_prompt.format(history=history, user_input=user_input, language=language))

        st.session_state.messages.append({"role": "assistant", "content": response})
        memory.save_context({"input": user_input}, {"output": response})

    # Refresh UI
    st.rerun()