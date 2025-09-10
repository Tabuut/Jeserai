# Replit Configuration File

## Overview

This is an Iraqi Dialect Telegram Bot powered by advanced AI models (currently GPT-4o via OpenRouter for text/vision and DALL-E 3 for image generation). The bot provides intelligent responses in Iraqi Arabic dialect, AI-powered image generation, imaginative creative descriptions, and educational poll/quiz functionality. Its primary purpose is to serve as a conversational AI assistant that understands and responds in Iraqi dialect, generates images, creates educational content, and provides interactive learning experiences. Key capabilities include a streamlined command structure (/start, /help, /chat, /image, /imagine_prompt, /translate, /translate_ar, /create_poll, /quiz, /help_poll), processing all text messages through AI, creating custom images and creative descriptions, generating interactive polls and educational quizzes. It features a comprehensive memory system, admin controls, user management, and educational content creation.

## User Preferences

Preferred communication style: Simple, everyday language.
Bot Creator Attribution: Always mention that the bot was designed and programmed by Thabet (@tht_txt) in welcome and help messages.

## System Architecture

### Bot Framework Architecture
- **Telegram Bot Framework**: Built using `python-telegram-bot` for Telegram API interactions.
- **Modular Design**: Handlers are organized into dedicated classes.
- **Async/Await**: Utilizes asynchronous programming for concurrent interactions.
- **Error Handling**: Centralized system for graceful failure management.

### AI Integration Layer
- **Dual Client Architecture**:
  - OpenRouter Client (GPT-4o): For text generation, analysis, and vision, optimized for cost-efficiency.
  - Direct OpenAI Client (DALL-E 3): For image generation, if `OPENAI_API_KEY` is available.
- **Iraqi Dialect Specialization**: System instructions are configured for Iraqi Arabic dialect responses.
- **Response Generation**: Advanced prompt engineering for consistent, high-quality Iraqi dialect output.
- **Image Generation**: AI-powered image creation using DALL-E 3.
- **Image Analysis**: GPT-4o Vision integration for photo analysis and description.
- **Translation Services**: Bidirectional Arabic-English translation with improved accuracy.
- **Creative Descriptions**: Dual-language content generation (English prompts, Arabic descriptions).
- **Poll Creation System**: Interactive Telegram polls with custom questions and multiple choice options.
- **Educational Quiz System**: Pre-built 6th grade science curriculum with 10 questions, correct answers, and explanations.
- **Interactive Learning**: Quiz-type polls with instant feedback and educational explanations.
- **Graceful Degradation**: Intelligent fallback mechanisms when services are unavailable.
- **Advanced Memory System**: Comprehensive conversation memory with short-term (10 messages context, 50 messages storage) and long-term memory (user pattern recognition, preferences tracking). Includes context understanding with smart question linking and response correlation.
- **Group Control System**: Activation ("سولف") and silence ("انجب") keywords.

### Message Processing Pipeline
- **Command Routing**: Handles `/start`, `/help`, `/chat`, `/image`, `/imagine`, `/imagine_prompt`, `/translate`, `/translate_ar`, `/clear_memory`, `/create_poll`, `/quiz`, `/help_poll`, and admin commands (`/admin`, `/stats`, `/logs`, `/broadcast`, `/broadcast_to`, `/list_users`, `/add_users`, `/admin_commands`, `/promo_messages`).
- **Text Message Processing**: All non-command text messages are routed to AI.
- **Image Request Detection**: Automatic detection of image generation requests using keyword matching.
- **Response Formatting**: Structured formatting with Markdown and copyable text.
- **Media Handling**: Image generation, file management, and cleanup processes.
- **Timeout Management**: Advanced timeout handling for API requests.
- **Logging System**: Comprehensive logging for debugging and monitoring.

### Configuration Management
- **Environment Variables**: Secure configuration using environment variables for API keys.
- **Validation Layer**: Input validation for required environment variables.
- **Admin System**: Configured with a main admin user ID (7297257627), providing comprehensive control and user management.

## External Dependencies

### AI Services
- **OpenAI GPT-4o**: Primary AI service for generating Iraqi dialect responses, natural language processing, and multimodal analysis. Accessed via OpenRouter API (model: `openai/gpt-4o`).
- **OpenAI DALL-E 3**: AI-powered image creation service for high-quality custom image generation (model: `openai/dall-e-3`). Accessed via OpenRouter API, or direct OpenAI API if `OPENAI_API_KEY` is available.
- **OpenAI GPT-4o Vision**: Advanced image analysis service for photo description. Accessed via OpenRouter API (model: `openai/gpt-4o` with vision capabilities).

### Communication Platform
- **Telegram Bot API**: Core messaging platform integration for message handling, user interaction, and bot deployment.

### Python Libraries
- `python-telegram-bot`: Official Telegram Bot API wrapper.
- `openai`: OpenAI-compatible client for accessing services via OpenRouter.
- `requests`: HTTP library for image downloading.
- Standard Libraries: `logging`, `os`, `json`, `base64`, `uuid` for core functionality.

### Environment Configuration
- `TELEGRAM_BOT_TOKEN`: Authentication for Telegram Bot API.
- `OPENROUTER_API_KEY`: OpenRouter API key for accessing GPT-4o and DALL-E 3 services.
- `OPENAI_API_KEY`: Optional, for direct DALL-E 3 access if preferred over OpenRouter for image generation.