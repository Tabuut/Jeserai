#!/usr/bin/env python3

import asyncio
import os
import sys
from telegram import Bot
from telegram.constants import ParseMode

# Bot configuration
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
ADMIN_ID = 7297257627  # Bot administrator ID

async def send_admin_commands_list():
    """Send the comprehensive admin commands list to the bot administrator."""
    
    if not TELEGRAM_BOT_TOKEN:
        print("❌ TELEGRAM_BOT_TOKEN not found in environment variables")
        return
    
    # Read the admin commands list
    try:
        with open('admin_commands_list.txt', 'r', encoding='utf-8') as f:
            admin_message = f.read()
    except FileNotFoundError:
        print("❌ admin_commands_list.txt file not found")
        return
    
    # Initialize bot
    bot = Bot(token=TELEGRAM_BOT_TOKEN)
    
    try:
        # Send the admin commands list to the administrator
        message = await bot.send_message(
            chat_id=ADMIN_ID,
            text=admin_message,
            parse_mode=None  # Send as plain text to avoid parsing issues
        )
        
        # Try to pin the message in the chat
        try:
            await bot.pin_chat_message(
                chat_id=ADMIN_ID,
                message_id=message.message_id,
                disable_notification=True
            )
            print("✅ Admin commands list sent and pinned successfully!")
        except Exception as pin_error:
            print(f"✅ Admin commands list sent successfully!")
            print(f"⚠️ Could not pin message (normal in private chats): {pin_error}")
        
        print(f"📤 Message sent to admin ID: {ADMIN_ID}")
        print(f"📝 Message length: {len(admin_message)} characters")
        
    except Exception as e:
        print(f"❌ Error sending admin commands list: {e}")

if __name__ == "__main__":
    asyncio.run(send_admin_commands_list())