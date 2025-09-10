#!/usr/bin/env python3
"""
Script to send broadcast message to all bot users
"""

import os
import json
import asyncio
from telegram import Bot

async def send_reactivation_broadcast():
    """Send reactivation broadcast to all users."""
    
    # Get bot token
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    if not bot_token:
        print("❌ TELEGRAM_BOT_TOKEN not found")
        return
    
    # Load users
    users_file = "bot_users.json"
    try:
        if os.path.exists(users_file):
            with open(users_file, 'r') as f:
                users = set(json.load(f))
        else:
            users = set()
            print("❌ No users file found")
            return
    except Exception as e:
        print(f"❌ Error loading users: {e}")
        return
    
    print(f"📊 Found {len(users)} users to notify")
    
    # Create bot instance
    bot = Bot(token=bot_token)
    
    # Broadcast message
    message = """🎉 تم إعادة تفعيل البوت!

السلام عليكم ورحمة الله وبركاته

نعلمكم بأنه تم إعادة تفعيل البوت بنجاح وهو الآن يعمل بشكل طبيعي ومتاح لجميع المستخدمين.

✅ جميع الميزات متاحة الآن:
• المحادثة باللهجة العراقية 💬
• إنشاء الصور بالذكاء الاصطناعي 🎨  
• الوصف الإبداعي للصور 🎭

شكراً لصبركم وثقتكم 🙏

- المطور ثابت (@tht_txt)"""
    
    sent_count = 0
    failed_count = 0
    
    print("🔄 بدء إرسال البث...")
    
    for user_id in users.copy():
        try:
            await bot.send_message(
                chat_id=user_id,
                text=f"📢 إعلان مهم:\n\n{message}"
            )
            sent_count += 1
            print(f"✅ تم إرسال الرسالة للمستخدم {user_id}")
            
            # Small delay to avoid rate limits
            await asyncio.sleep(0.1)
            
        except Exception as e:
            failed_count += 1
            print(f"❌ فشل إرسال الرسالة للمستخدم {user_id}: {e}")
            
            # Remove inactive users
            if "bot was blocked" in str(e).lower() or "chat not found" in str(e).lower():
                users.discard(user_id)
                print(f"🗑️ تمت إزالة المستخدم غير النشط {user_id}")
    
    # Save updated users list
    try:
        with open(users_file, 'w') as f:
            json.dump(list(users), f)
    except Exception as e:
        print(f"❌ خطأ في حفظ قائمة المستخدمين: {e}")
    
    print(f"\n✅ تم إكمال البث!")
    print(f"📊 الإحصائيات:")
    print(f"   • تم الإرسال: {sent_count}")
    print(f"   • فشل الإرسال: {failed_count}")
    print(f"   • إجمالي المستخدمين الحاليين: {len(users)}")

if __name__ == "__main__":
    asyncio.run(send_reactivation_broadcast())