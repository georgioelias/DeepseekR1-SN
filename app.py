import streamlit as st
import os
import openai
import time
import re
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="DS-R1",
    page_icon="💬",
    layout="wide"
)

# Get API key from environment variable
api_key = os.environ.get("SAMBANOVA_API_KEY")
if not api_key:
    st.error("SambaNova API key not found in environment variables. Please set the SAMBANOVA_API_KEY environment variable.")
    st.stop()

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []

if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = "You are a helpful assistant"

if "chat_started" not in st.session_state:
    st.session_state.chat_started = False

# App title and description
st.title("Deepseek R1")
st.subheader("Powered by SambaNova Cloud API")

# Sidebar for model parameters
with st.sidebar:
    st.header("Configuration")
    
    # Model parameters
    st.subheader("Model Parameters")
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.1, step=0.1,
                            help="Controls randomness: 0 is deterministic, 1 is very creative")
    top_p = st.slider("Top P", min_value=0.0, max_value=1.0, value=0.1, step=0.1,
                      help="Controls diversity via nucleus sampling")
    
    # Add a clear chat button
    if st.button("Clear Chat", type="primary"):
        st.session_state.messages = []
        st.session_state.chat_started = False
        st.rerun()

# System prompt section (only visible before chat starts)
if not st.session_state.chat_started:
    
    # Text area for system prompt
    system_prompt = st.text_area(
        "System Prompt",
        value=st.session_state.system_prompt,
        height=150,
        help="This defines the behavior and capabilities of the AI."
    )
    
    # Start chat button
    if st.button("Start Chat", type="primary"):
        st.session_state.system_prompt = system_prompt
        st.session_state.chat_started = True
        
        # Add system message to messages
        st.session_state.messages.append({"role": "system", "content": system_prompt})
        st.rerun()

# Function to process and format AI response
def process_response(text):
    # Regex pattern to find <think> content </think>
    think_pattern = r'<think>(.*?)</think>'
    
    # Search for thinking section
    thinking_match = re.search(think_pattern, text, re.DOTALL)
    
    if thinking_match:
        thinking_content = thinking_match.group(1).strip()
        # Remove thinking section from the original text to get final response
        final_response = re.sub(think_pattern, '', text, flags=re.DOTALL).strip()
        return thinking_content, final_response
    else:
        # If no thinking pattern found, return empty thinking and the original text
        return "", text

# Chat interface (only visible after chat starts)
if st.session_state.chat_started:
    # Show current system prompt
    with st.expander("Current System Prompt"):
        st.write(st.session_state.system_prompt)
    
    # Display chat messages
    for message in st.session_state.messages:
        if message["role"] != "system":  # Don't display system messages in the chat UI
            with st.chat_message(message["role"]):
                if message["role"] == "assistant" and "thinking" in message:
                    # Display thinking in an expander
                    with st.expander("Thinking Process"):
                        st.write(message['thinking'])
                    # Display final response
                    st.write(message["content"])
                else:
                    st.write(message["content"])
    
    # User input
    if user_input := st.chat_input("Type your message here..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Display user message
        with st.chat_message("user"):
            st.write(user_input)
        
        # Display assistant response with a spinner
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            thinking_expander = st.expander("Thinking Process", expanded=False)
            thinking_placeholder = thinking_expander.empty()
            
            full_response = ""
            
            try:
                # Initialize the OpenAI client
                client = openai.OpenAI(
                    api_key=api_key,
                    base_url="https://api.sambanova.ai/v1",
                )
                
                # Create messages list for API call
                messages_for_api = [msg for msg in st.session_state.messages]
                
                # Make API call
                with st.spinner("Thinking..."):
                    response = client.chat.completions.create(
                        model="DeepSeek-R1",
                        messages=messages_for_api,
                        temperature=temperature,
                        top_p=top_p,
                        stream=True  # Enable streaming
                    )
                    
                    # Process streaming response
                    for chunk in response:
                        if chunk.choices[0].delta.content:
                            full_response += chunk.choices[0].delta.content
                            
                            # Process the response to separate thinking and final response
                            thinking, final = process_response(full_response)
                            
                            # Update the thinking section if it exists
                            if thinking:
                                thinking_placeholder.write(thinking)
                                thinking_expander.expanded = True
                            
                            # Update the final response
                            response_placeholder.write(final)
                            
                            time.sleep(0.01)  # Small delay to make streaming visible
                
                # Process the complete response
                thinking_content, final_response = process_response(full_response)
                
                # Add the assistant's message to the history with both thinking and final response
                assistant_message = {
                    "role": "assistant", 
                    "content": final_response
                }
                
                if thinking_content:
                    assistant_message["thinking"] = thinking_content
                
                st.session_state.messages.append(assistant_message)
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
                if "api_key" in str(e).lower():
                    st.warning("API key error. Please check your SambaNova API key in the environment variables.")
                else:
                    st.warning("There was an error connecting to the model. Please try again.")