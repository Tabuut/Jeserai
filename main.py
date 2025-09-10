#!/usr/bin/env python3
"""
Arabic Telegram Bot with OpenAI Integration
Main entry point for the bot application.
"""

import logging
import os
from telegram.ext import Application, MessageHandler, CommandHandler, filters
from bot_handlers import BotHandlers
from keep_alive import keep_alive

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def main():
    """Main function to start the Telegram bot."""
    
    # Get environment variables
    telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
    openrouter_api_key = os.getenv('OPENROUTER_API_KEY')
    
    if not telegram_token:
        logger.error("TELEGRAM_BOT_TOKEN environment variable is not set")
        return
    
    if not openrouter_api_key:
        logger.error("OPENROUTER_API_KEY environment variable is not set")
        return
    
    # Create the Application
    application = Application.builder().token(telegram_token).build()
    
    # Initialize bot handlers
    handlers = BotHandlers()
    
    # Add handlers
    application.add_handler(CommandHandler("start", handlers.start_command))
    application.add_handler(CommandHandler("help", handlers.help_command))
    application.add_handler(CommandHandler("clear_memory", handlers.clear_memory_command))
    application.add_handler(CommandHandler("image", handlers.image_mode_command))
    application.add_handler(CommandHandler("chat", handlers.chat_mode_command))
    application.add_handler(CommandHandler("imagine", handlers.imagine_command))
    application.add_handler(CommandHandler("imagine_prompt", handlers.imagine_prompt_command))
    application.add_handler(CommandHandler("broadcast", handlers.broadcast_command))
    application.add_handler(CommandHandler("translate", handlers.translate_command))
    application.add_handler(CommandHandler("translate_ar", handlers.translate_ar_command))
    # GPT-4.1 Enhanced Commands
    application.add_handler(CommandHandler("gpt41_analyze", handlers.gpt41_analyze_command))
    application.add_handler(CommandHandler("gpt41_structured", handlers.gpt41_structured_command))
    application.add_handler(CommandHandler("gpt41_features", handlers.gpt41_features_command))
    # Poll and Quiz Commands  
    application.add_handler(CommandHandler("create_poll", handlers.create_poll_command))
    application.add_handler(CommandHandler("quiz", handlers.quiz_command))
    application.add_handler(CommandHandler("help_poll", handlers.help_poll_command))
    # Admin commands
    application.add_handler(CommandHandler("admin", handlers.admin_command))
    application.add_handler(CommandHandler("admin_commands", handlers.admin_commands_command))
    application.add_handler(CommandHandler("promo_messages", handlers.promo_messages_command))
    application.add_handler(CommandHandler("stats", handlers.stats_command))
    application.add_handler(CommandHandler("logs", handlers.logs_command))
    application.add_handler(CommandHandler("broadcast_to", handlers.broadcast_to_command))
    application.add_handler(CommandHandler("list_users", handlers.list_users_command))
    application.add_handler(CommandHandler("add_users", handlers.add_users_command))
    application.add_handler(MessageHandler(filters.PHOTO, handlers.handle_photo))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handlers.handle_message))
    
    # Error handler
    application.add_error_handler(handlers.error_handler)
    
    # Start keep-alive server
    keep_alive()
    
    logger.info("Starting Telegram bot...")
    
    # Run the bot with conflict resolution
    try:
        application.run_polling(
            allowed_updates=['message'],
            drop_pending_updates=True,
            poll_interval=2.0,
            timeout=10
        )
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Bot stopped due to error: {e}")

if __name__ == '__main__':
    main()
