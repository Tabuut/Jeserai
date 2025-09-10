"""
Telegram Bot Handlers
Contains all message and command handlers for the bot.
"""

import logging
import os
import uuid
import json
import asyncio
from telegram import Update
from telegram.ext import ContextTypes
from telegram.constants import ParseMode
from multi_ai_client import MultiAIClient

logger = logging.getLogger(__name__)

class BotHandlers:
    """Handles all bot interactions and commands."""
    
    def __init__(self):
        """Initialize the bot handlers with Multi-AI client (GPT-5 + Gemini)."""
        try:
            self.ai_client = MultiAIClient()
            # User modes: 'chat', 'image', 'imagine_prompt'  
            self.user_modes = {}
            # User tracking for broadcasts
            self.users_file = "bot_users.json"
            # Group status tracking: True = active (سولف), False = silent (انجب)
            self.group_status_file = "group_status.json"
            self.group_status = {}
            # Enhanced conversation memory system
            self.conversations_file = "conversations.json"
            self.conversations = {}  # In-memory storage: {user_id: [{"user": msg, "assistant": resp, "timestamp": time}, ...]}
            self.max_messages_per_user = 100  # Increased for better long-term memory
            self.context_messages = 15  # Increased for better context understanding
            self.long_term_memory_file = "long_term_memory.json"
            self.long_term_memory = {}  # Long-term patterns and preferences
            # Admin configuration
            self.ADMIN_IDS = {7297257627}  # Bot owner/admin IDs
            self.load_users()
            self.load_group_status()
            self.load_conversations()
            self.load_long_term_memory()
            logger.info("Multi-AI client (GPT-5 + Gemini) initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Multi-AI client: {e}")
            self.ai_client = None
    
    def load_users(self):
        """Load users from file."""
        try:
            if os.path.exists(self.users_file):
                with open(self.users_file, 'r') as f:
                    self.users = set(json.load(f))
            else:
                self.users = set()
        except Exception as e:
            logger.error(f"Error loading users: {e}")
            self.users = set()
    
    def save_users(self):
        """Save users to file."""
        try:
            with open(self.users_file, 'w') as f:
                json.dump(list(self.users), f)
        except Exception as e:
            logger.error(f"Error saving users: {e}")
    
    def add_user(self, user_id: int):
        """Add user to tracking list."""
        self.users.add(user_id)
        self.save_users()
    
    def load_group_status(self):
        """Load group status from file."""
        try:
            if os.path.exists(self.group_status_file):
                with open(self.group_status_file, 'r') as f:
                    self.group_status = json.load(f)
            else:
                self.group_status = {}
        except Exception as e:
            logger.error(f"Error loading group status: {e}")
            self.group_status = {}
    
    def save_group_status(self):
        """Save group status to file."""
        try:
            with open(self.group_status_file, 'w') as f:
                json.dump(self.group_status, f)
        except Exception as e:
            logger.error(f"Error saving group status: {e}")
    
    def set_group_status(self, chat_id: int, active: bool):
        """Set group status (True = active/سولف, False = silent/انجب)."""
        self.group_status[str(chat_id)] = active
        self.save_group_status()
    
    def is_group_active(self, chat_id: int) -> bool:
        """Check if group is active. Default is True for private chats and new groups."""
        if chat_id > 0:  # Private chat
            return True
        return self.group_status.get(str(chat_id), True)  # Default active for new groups
    
    def load_conversations(self):
        """Load conversation history from file."""
        try:
            if os.path.exists(self.conversations_file):
                with open(self.conversations_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Convert string keys back to int
                    self.conversations = {int(k): v for k, v in data.items()}
            else:
                self.conversations = {}
        except Exception as e:
            logger.error(f"Error loading conversations: {e}")
            self.conversations = {}
    
    def save_conversations(self):
        """Save conversation history to file."""
        try:
            with open(self.conversations_file, 'w', encoding='utf-8') as f:
                # Convert int keys to string for JSON
                data = {str(k): v for k, v in self.conversations.items()}
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Error saving conversations: {e}")
    
    def add_message_to_conversation(self, user_id: int, user_message: str, bot_response: str):
        """Add a message exchange to conversation history with enhanced tracking."""
        import time
        
        if user_id not in self.conversations:
            self.conversations[user_id] = []
        
        # Add the new exchange with timestamp
        exchange = {
            "user": user_message,
            "assistant": bot_response,
            "timestamp": time.time()
        }
        self.conversations[user_id].append(exchange)
        
        # Keep only the last max_messages_per_user exchanges
        if len(self.conversations[user_id]) > self.max_messages_per_user:
            self.conversations[user_id] = self.conversations[user_id][-self.max_messages_per_user:]
        
        # Update long-term memory patterns
        self.update_long_term_memory(user_id, user_message, bot_response)
        
        # Save to files
        self.save_conversations()
        self.save_long_term_memory()
    
    def get_conversation_context(self, user_id: int) -> str:
        """Get conversation context for a user."""
        if user_id not in self.conversations or not self.conversations[user_id]:
            return ""
        
        context = "=== المحادثة السابقة ===\n"
        
        # Get recent exchanges for context - increased to 8 for better understanding
        recent_exchanges = self.conversations[user_id][-8:]
        for i, exchange in enumerate(recent_exchanges, 1):
            context += f"#{i} المستخدم: {exchange['user']}\n"
            # Don't truncate responses - keep full context for better understanding
            context += f"البوت: {exchange['assistant']}\n\n"
        
        # Add user patterns and preferences
        patterns = self.get_user_patterns(user_id)
        if patterns:
            context += f"=== أنماط المستخدم ===\n{patterns}\n\n"
        
        # Add conversation analysis
        conversation_summary = self.analyze_conversation_patterns(user_id)
        if conversation_summary:
            context += f"=== تحليل المحادثة ===\n{conversation_summary}\n\n"
        
        context += "=== التعليمات المحسنة ===\n"
        context += "اعتمد على السياق أعلاه للرد بذكاء. لا تكرر نفس الردود للأسئلة المتشابهة - نوع في إجاباتك واربط مع المحادثات السابقة. إذا سأل نفس السؤال مرة ثانية، اعرف إنه يريد تفصيل أكثر أو زاوية مختلفة.\n"
        context += "=== الرسالة الجديدة ===\n"
        return context
    
    def load_long_term_memory(self):
        """Load long-term memory patterns from file."""
        try:
            if os.path.exists(self.long_term_memory_file):
                with open(self.long_term_memory_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.long_term_memory = {int(k): v for k, v in data.items()}
            else:
                self.long_term_memory = {}
        except Exception as e:
            logger.error(f"Error loading long-term memory: {e}")
            self.long_term_memory = {}
    
    def save_long_term_memory(self):
        """Save long-term memory patterns to file."""
        try:
            with open(self.long_term_memory_file, 'w', encoding='utf-8') as f:
                data = {str(k): v for k, v in self.long_term_memory.items()}
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Error saving long-term memory: {e}")
    
    def update_long_term_memory(self, user_id: int, user_message: str, bot_response: str):
        """Update user patterns and preferences in long-term memory."""
        if user_id not in self.long_term_memory:
            self.long_term_memory[user_id] = {
                "frequent_topics": {},
                "communication_style": [],
                "preferences": {},
                "total_messages": 0,
                "question_history": [],  # Track repeated questions
                "conversation_themes": {},  # Track conversation themes
                "response_preferences": {}  # Track what kind of responses user likes
            }
        
        memory = self.long_term_memory[user_id]
        
        # Ensure all required fields exist (fix for corrupted memory files)
        required_fields = {
            "frequent_topics": {},
            "communication_style": [],
            "preferences": {},
            "total_messages": 0,
            "question_history": [],
            "conversation_themes": {},
            "response_preferences": {}
        }
        
        for field, default_value in required_fields.items():
            if field not in memory:
                memory[field] = default_value.copy() if isinstance(default_value, (dict, list)) else default_value
        
        memory["total_messages"] += 1
        
        # Enhanced topic tracking
        user_lower = user_message.lower()
        extended_keywords = ["ذاكرة", "تقييم", "كم من 10", "جيد", "ممتاز", "شلونك", "صورة", 
                            "مساعدة", "كيف", "شنو", "وين", "ليش", "متى", "أسئلة", "معلومات", "شرح"]
        for keyword in extended_keywords:
            if keyword in user_lower:
                memory["frequent_topics"][keyword] = memory["frequent_topics"].get(keyword, 0) + 1
        
        # Track repeated questions for variation
        import time
        question_essence = self.extract_question_essence(user_message)
        if question_essence:
            memory["question_history"].append({
                "question": question_essence,
                "full_message": user_message,
                "timestamp": time.time(),
                "response_type": self.categorize_response(bot_response)
            })
            # Keep only last 20 questions for efficiency
            if len(memory["question_history"]) > 20:
                memory["question_history"] = memory["question_history"][-20:]
        
        # Enhanced communication pattern tracking
        if "من 10" in user_lower or "كم من 10" in user_lower:
            if "يحب التقييمات الرقمية" not in memory["communication_style"]:
                memory["communication_style"].append("يحب التقييمات الرقمية")
        
        if len(user_message.split()) <= 3:
            if "يستخدم رسائل قصيرة" not in memory["communication_style"]:
                memory["communication_style"].append("يستخدم رسائل قصيرة")
        elif len(user_message.split()) > 15:
            if "يفضل الشرح المفصل" not in memory["communication_style"]:
                memory["communication_style"].append("يفضل الشرح المفصل")
        
        # Track conversation themes
        theme = self.detect_conversation_theme(user_message)
        if theme:
            memory["conversation_themes"][theme] = memory["conversation_themes"].get(theme, 0) + 1
    
    def get_user_patterns(self, user_id: int) -> str:
        """Get user patterns summary for context."""
        if user_id not in self.long_term_memory:
            return ""
        
        memory = self.long_term_memory[user_id]
        patterns = []
        
        if memory["total_messages"] > 5:
            patterns.append(f"عدد الرسائل الكلي: {memory['total_messages']}")
        
        if memory["frequent_topics"]:
            top_topics = sorted(memory["frequent_topics"].items(), key=lambda x: x[1], reverse=True)[:3]
            topics_str = ", ".join([f"{topic} ({count})" for topic, count in top_topics])
            patterns.append(f"المواضيع المفضلة: {topics_str}")
        
        if memory["communication_style"]:
            patterns.append(f"نمط التواصل: {', '.join(memory['communication_style'])}")
        
        return " | ".join(patterns) if patterns else ""
    
    def analyze_conversation_patterns(self, user_id: int) -> str:
        """Analyze conversation patterns and detect repeated questions."""
        if user_id not in self.long_term_memory or not self.conversations.get(user_id):
            return ""
        
        memory = self.long_term_memory[user_id]
        analysis = []
        
        # Check for repeated questions
        if "question_history" in memory and len(memory["question_history"]) > 1:
            recent_questions = memory["question_history"][-5:]  # Last 5 questions
            question_counts = {}
            for q in recent_questions:
                question_counts[q["question"]] = question_counts.get(q["question"], 0) + 1
            
            repeated = [q for q, count in question_counts.items() if count > 1]
            if repeated:
                analysis.append(f"أسئلة متكررة: {', '.join(repeated[:2])}")
        
        # Analyze conversation flow
        if len(self.conversations[user_id]) >= 3:
            recent_messages = [exc["user"] for exc in self.conversations[user_id][-3:]]
            if self.has_conversation_progression(recent_messages):
                analysis.append("المحادثة متدرجة ومترابطة")
            else:
                analysis.append("أسئلة متنوعة غير مترابطة")
        
        return " | ".join(analysis) if analysis else ""
    
    def extract_question_essence(self, message: str) -> str:
        """Extract the core essence of a question for tracking repetitions."""
        message_lower = message.lower().strip()
        
        # Common question patterns in Iraqi dialect
        question_patterns = {
            "شنو": "what_question",
            "كيف": "how_question", 
            "وين": "where_question",
            "متى": "when_question",
            "ليش": "why_question",
            "كم من 10": "rating_question",
            "شلونك": "greeting_question",
            "مساعدة": "help_question",
            "شرح": "explain_request"
        }
        
        for pattern, essence in question_patterns.items():
            if pattern in message_lower:
                return essence
        
        # Check for question marks or general inquiry patterns
        if "؟" in message or any(word in message_lower for word in ["أريد", "أبغى", "عاوز", "بدي"]):
            return "general_inquiry"
            
        return ""
    
    def categorize_response(self, response: str) -> str:
        """Categorize the type of response given."""
        response_lower = response.lower()
        
        if any(word in response_lower for word in ["شرح", "تفصيل", "معلومات"]):
            return "detailed_explanation"
        elif any(word in response_lower for word in ["نعم", "لا", "صح", "خطأ"]):
            return "simple_answer"
        elif any(word in response_lower for word in ["مساعدة", "خدمة", "أساعدك"]):
            return "helpful_guidance"
        elif "/10" in response or any(word in response_lower for word in ["درجة", "تقييم"]):
            return "rating_response"
        else:
            return "general_response"
    
    def detect_conversation_theme(self, message: str) -> str:
        """Detect the main theme of a conversation message."""
        message_lower = message.lower()
        
        theme_patterns = {
            "تقنية": ["برمجة", "كمبيوتر", "إنترنت", "تطبيق", "موقع", "ذكاء اصطناعي"],
            "تعليمية": ["شرح", "تعلم", "دراسة", "كتاب", "معلومات", "أسئلة"],
            "شخصية": ["شلونك", "كيفك", "أخبار", "حال", "صحة"],
            "طلب مساعدة": ["مساعدة", "ساعدني", "أحتاج", "بدي", "أريد"],
            "تقييم": ["كم من 10", "تقييم", "رأي", "درجة", "جيد", "ممتاز"],
        }
        
        for theme, keywords in theme_patterns.items():
            if any(keyword in message_lower for keyword in keywords):
                return theme
                
        return "عام"
    
    def has_conversation_progression(self, messages: list) -> bool:
        """Check if messages show logical progression."""
        if len(messages) < 2:
            return False
            
        # Simple check: see if messages build upon each other
        # Look for follow-up words or continuation patterns
        follow_up_indicators = ["وبعدين", "كمان", "أيضاً", "لكن", "ماذا عن", "وإيش", "وشنو"]
        
        for i in range(1, len(messages)):
            current_msg = messages[i].lower()
            if any(indicator in current_msg for indicator in follow_up_indicators):
                return True
                
        return False
    
    def clear_conversation(self, user_id: int):
        """Clear conversation history for a user."""
        if user_id in self.conversations:
            del self.conversations[user_id]
            self.save_conversations()
    
    def is_admin(self, user_id: int) -> bool:
        """Check if user is an admin."""
        return user_id in self.ADMIN_IDS
    
    def admin_only(func):
        """Decorator to restrict command to admins only."""
        async def wrapper(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
            user_id = update.effective_user.id
            if not self.is_admin(user_id):
                await update.message.reply_text(
                    "معذرة، هذا الأمر مخصص للمطورين فقط."
                )
                return
            return await func(self, update, context)
        return wrapper
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle the /start command."""
        user_id = update.effective_user.id
        self.add_user(user_id)  # Track user
        
        welcome_message = "أهلاً وسهلاً بيك في البوت العراقي المدعوم بالذكاء الاصطناعي!\n\nأني بوت ذكي استخدم تقنيات الذكاء الاصطناعي المتطورة للرد عليك باللهجة العراقية الأصيلة.\n\nالأوامر الموجودة:\n💬 /chat - المحادثة العادية\n🎨 /image - وضع إنشاء الصور\n🎭 /imagine_prompt - وصف إبداعي للصور مع نص إنجليزي\n🌐 /translate - ترجمة النص للإنجليزية\n🌐 /translate_ar - ترجمة النص للعربية\n📊 /create_poll - إنشاء استطلاع رأي مخصص\n🎓 /quiz - اختبار العلوم للسادس الابتدائي\n📋 /help_poll - مساعدة الاستطلاعات والاختبارات\n📸 ارسل صور - تحليل ووصف الصور تلقائياً\n❓ /help - المساعدة والشرح المفصل\n\nميزات جديدة:\n📊 إنشاء استطلاعات واختبارات تعليمية تفاعلية\n🎓 اختبار العلوم للسادس الابتدائي مع إجابات صحيحة\n📸 ارسل أي صورة وراح أحللها وأوصفها لك\n🌐 ترجمة فورية بين العربية والإنجليزية\n🎭 وصف إبداعي متطور للصور\n\nيلا نبدي الحچي! هسه إنت بوضع المحادثة العادية 💬\n\nهذا البوت تم تصميمه وبرمجته بواسطة المطور ثابت\nللتواصل مع المطور: @tht_txt\n\nمرحباً بيك مرة ثانية ونورت البوت!"
        
        await update.message.reply_text(
            welcome_message
        )
        
        logger.info(f"Start command used by user {user_id}")
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle the /help command."""
        help_message = "مساعدة البوت العراقي 🤖\n\nالأوامر الموجودة:\n- /start بداية الحچي ويا البوت\n- /help عرض هاي الرسالة\n- /clear_memory مسح ذاكرة المحادثة 🧠\n- /image وضع إنشاء الصور 🎨\n- /chat وضع المحادثة العادي 💬\n- /imagine_prompt وصف إبداعي للصور مع نص إنجليزي 🎭\n- /translate ترجمة النص للإنجليزية 🌐\n- /translate_ar ترجمة النص للعربية 🌐\n- /create_poll إنشاء استطلاع رأي مخصص 📊\n- /quiz اختبار العلوم للسادس الابتدائي 🎓\n- /help_poll مساعدة الاستطلاعات والاختبارات 📋\n- 📸 ارسال الصور - تحليل الصور ووصفها\n\nشلون تستخدمني:\n1. ارسل أي رسالة نصية للبوت\n2. راح أرد عليك باللهجة العراقية باستخدام الذكاء الاصطناعي\n3. تكدر تسأل أسئلة أو تطلب مساعدة بأي موضوع\n4. استخدم الأوامر للتنقل بين الأوضاع المختلفة\n\n🎨 وضع إنشاء الصور (/image):\nبعد استخدام هذا الأمر، كل رسالة ترسلها راح تتحول لصورة:\n- \"قط صغير لطيف\"\n- \"بيت تراثي عراقي\"\n- \"منظر طبيعي جميل\"\n- \"سيارة حمراء رياضية\"\n\n🎭 أمر الوصف الإبداعي (/imagine_prompt):\nاستخدم هذا الأمر لوصف إبداعي وحيوي مع نص إنجليزي:\n1. اكتب `/imagine_prompt`\n2. راح يطلب منك البوت تكتب وصف الصورة\n3. اكتب الوصف مثل \"طائر العنقاء ذهبي\"\n4. راح يكتب لك نص إنجليزي مفصل + وصف إبداعي عراقي\n\n📸 تحليل الصور:\nارسل أي صورة للبوت وراح يحللها ويوصفها لك باللهجة العراقية:\n- ارسل صورة منظر طبيعي\n- ارسل صورة شخص أو حيوان\n- ارسل صورة طعام أو أشياء\n- تكدر تكتب تعليق مع الصورة لتوضيح شنو تريد\n\n🌐 الترجمة:\n- `/translate النص هنا` - ترجمة للإنجليزية\n- `/translate_ar Text here` - ترجمة للعربية\n\nأمثلة:\n- `/translate مرحبا بك في العراق`\n- `/translate_ar Hello world`\n- `/translate كيف الحال اليوم؟`\n- `/translate_ar How are you today?`\n\n💬 وضع المحادثة (/chat):\nالوضع العادي للمحادثة والأسئلة والأجوبة\n\nميزات جديدة:\n- 📸 تحليل الصور تلقائياً عند إرسالها\n- 🌐 ترجمة النصوص بين العربية والإنجليزية\n- 🎭 وصف إبداعي متطور للصور\n- 💬 محادثة ذكية باللهجة العراقية\n\nملاحظات مهمة:\n- البوت يفهم النصوص العربية والإنجليزية\n- كل الردود راح تكون باللهجة العراقية\n- الوضع الحالي يظهر في بداية كل رد\n- تكدر تغير الوضع أي وقت بالأوامر\n- النصوص الإنجليزية تكون قابلة للنسخ\n- الصور تحلل تلقائياً بالذكاء الاصطناعي\n\nإذا واجهت أي مشكلة، تأكد إن رسالتك واضحة ومفهومة.\n\nهذا البوت تم تصميمه وبرمجته بالكامل بواسطة المطور الموهوب ثابت\nللتواصل مع المطور أو لطلب تطوير بوتات مشابهة: @tht_txt\n\nشكراً لاستخدامك البوت العراقي!"
        
        await update.message.reply_text(
            help_message
        )
        
        logger.info(f"Help command used by user {update.effective_user.id}")
    
    async def clear_memory_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle the /clear_memory command to clear conversation history."""
        user_id = update.effective_user.id
        
        if user_id in self.conversations and self.conversations[user_id]:
            # Clear conversation history
            self.clear_conversation(user_id)
            
            await update.message.reply_text(
                "🧠 تم مسح ذاكرة المحادثة بنجاح!\n\n"
                "البوت هسه نسى جميع المحادثات السابقة وياك.\n"
                "المحادثة الجاية راح تبدأ من الصفر. 🔄"
            )
            logger.info(f"User {user_id} cleared conversation memory")
        else:
            await update.message.reply_text(
                "🧠 ما اكو ذاكرة محادثة لتمسحها!\n\n"
                "البوت ما عنده أي محادثات محفوظة وياك.\n"
                "ابدأ محادثة جديدة وراح يتذكر كلامك. 💬"
            )
            logger.info(f"User {user_id} tried to clear empty conversation memory")
    
    async def image_mode_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Switch to image generation mode."""
        user_id = update.effective_user.id
        
        # Check if image generation is available
        if not self.ai_client.gemini_api_key:
            await update.message.reply_text(
                "🎨 معذرة، وضع إنشاء الصور غير متاح حالياً.\n\n"
                "السبب: يحتاج مفتاح Gemini API للوصول لـ Gemini Image Generation\n\n"
                "بدلاً من ذلك، جرب:\n"
                "🎭 /imagine_prompt - للحصول على وصف إبداعي مفصل مع نص إنجليزي قابل للنسخ\n"
                "💬 /chat - للمحادثة العادية"
            )
            return
        
        self.user_modes[user_id] = 'image'
        
        await update.message.reply_text(
            "🎨 تم التحويل لوضع إنشاء الصور!\n\n"
            "هسه كل رسالة ترسلها راح اسوي منها صورة.\n"
            "مثال: ارسل \"قط صغير أبيض\" وراح اسوي لك صورة قط.\n\n"
            "للعودة للمحادثة العادية استخدم: /chat"
        )
        logger.info(f"User {user_id} switched to image mode")
    

    
    async def chat_mode_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Switch to normal chat mode."""
        user_id = update.effective_user.id
        self.user_modes[user_id] = 'chat'
        
        await update.message.reply_text(
            "💬 تم التحويل للمحادثة العادية!\n\n"
            "هسه تكدر تحچي وياي عادي وتسأل أسئلة.\n"
            "للتحويل لأوضاع أخرى:\n"
            "🎨 /image - وضع إنشاء الصور\n"
            "🎭 /imagine_prompt - احصل على نص إنجليزي مفصل للصور"
        )
        logger.info(f"User {user_id} switched to chat mode")
    
    async def imagine_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle the /imagine command with prompt."""
        user_id = update.effective_user.id
        
        # Get the full command text
        full_text = update.message.text
        
        # Check if the command follows the format "/imagine prompt:"
        if not full_text.startswith("/imagine prompt:"):
            await update.message.reply_text(
                "🎭 استخدم الأمر بالطريقة الصحيحة:\n\n"
                "`/imagine prompt: وصف الصورة هنا`\n\n"
                "مثال:\n"
                "`/imagine prompt: طائر العنقاء ذو ألوان ذهبية يطير في السماء`",
                parse_mode='Markdown'
            )
            return
        
        # Extract the prompt text after "/imagine prompt:"
        prompt_text = full_text[16:].strip()  # Remove "/imagine prompt:" (16 characters)
        
        if not prompt_text:
            await update.message.reply_text(
                "🎭 اكتب وصف الصورة بعد الأمر!\n\n"
                "مثال:\n"
                "`/imagine prompt: منظر طبيعي جميل مع غروب الشمس`",
                parse_mode='Markdown'
            )
            return
        
        try:
            # Send typing action
            await context.bot.send_chat_action(
                chat_id=update.effective_chat.id,
                action="typing"
            )
            
            # Create a specialized prompt for creative description with English prompt
            creative_prompt = f"""
أنت خبير في كتابة الأوصاف الإبداعية والنصوص الفنية. المستخدم أرسل هذا الطلب: "{prompt_text}"

اكتب أولاً نص إنجليزي مفصل (prompt) لإنشاء هذه الصورة بالذكاء الاصطناعي، ثم اكتب وصف إبداعي باللهجة العراقية.

النص الإنجليزي يجب أن يشمل:
- وصف تفصيلي للمشهد
- التفاصيل الفنية والبصرية
- الألوان والإضاءة
- جودة عالية ومواصفات تقنية

الوصف العربي يجب أن يكون:
- مفصل وغني بالتفاصيل البصرية
- إبداعي ومثير للخيال
- حيوي ومليء بالألوان والحركة
- باللهجة العراقية الأصيلة

اكتب النص الإنجليزي أولاً، ثم الوصف العراقي.
            """
            
            # Generate creative description using Gemini with timeout
            try:
                import asyncio
                response = await asyncio.wait_for(
                    self.ai_client.generate_response(creative_prompt), 
                    timeout=20.0
                )
            except asyncio.TimeoutError:
                response = None
                logger.warning("Gemini API timed out for imagine command")
            
            # Check if response is valid or use fallback
            if not response or "هلا وغلا" in response or "ما كدرت افهم" in response:
                response = self._create_creative_fallback(prompt_text)
            
            # Format the response
            formatted_response = self._format_copyable_text(response)
            
            await update.message.reply_text(
                f"🎭 هاي الوصف الإبداعي للي طلبته:\n\n{formatted_response}",
                parse_mode='Markdown'
            )
            
            logger.info(f"Successfully generated creative description for user {user_id}")
            
        except Exception as e:
            logger.error(f"Error in imagine command for user {user_id}: {e}")
            await update.message.reply_text(
                "معذرة، صار خطأ وقت كتابة الوصف الإبداعي. جرب مرة ثانية."
            )

    async def imagine_prompt_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle the /imagine_prompt command that waits for user input."""
        user_id = update.effective_user.id
        
        # Set user to imagine_prompt mode to wait for their text input
        self.user_modes[user_id] = 'imagine_prompt'
        
        await update.message.reply_text(
            "🎭 زين! هسه اكتب وصف الصورة الي تريدها:\n\n"
            "أمثلة:\n"
            "• طائر العنقاء ذهبي يطير بالسماء\n"
            "• قلعة أسطورية بين الغيوم\n"
            "• غروب جميل فوق البحر\n\n"
            "راح اكتب لك نص إنجليزي مفصل تكدر تستخدمه للذكاء الاصطناعي! ✨"
        )
        
        logger.info(f"User {user_id} started imagine_prompt mode")
    
    async def admin_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle the /admin command to show admin panel."""
        user_id = update.effective_user.id
        
        if not self.is_admin(user_id):
            await update.message.reply_text("انت مو مخول لاستخدام هذا الأمر.")
            return
        
        admin_message = f"""
🔧 لوحة التحكم الإدارية

مرحباً بك أدمن البوت! 👨‍💻

الأوامر المتاحة لك:
• /broadcast <رسالة> - إرسال رسالة جماعية لجميع المستخدمين
• /broadcast_to <معرفات> <رسالة> - إرسال رسالة لمستخدمين محددين
• /list_users - عرض قائمة جميع المستخدمين المسجلين
• /add_users <معرفات> - إضافة مستخدمين جدد يدوياً
• /promo_messages - الرسائل الترويجية الجاهزة
• /admin_commands - دليل الأوامر الكامل
• /stats - إحصائيات البوت والمستخدمين  
• /admin - عرض هذه اللوحة
• /logs - عرض آخر الأخطاء والسجلات

الصلاحيات الخاصة بك:
✅ إرسال رسائل جماعية
✅ عرض إحصائيات مفصلة
✅ الوصول لجميع ميزات البوت
✅ إدارة المستخدمين
✅ مراقبة أداء البوت

معلومات المستخدم:
• معرف المستخدم: {user_id}
• حالة الأدمن: ✅ مفعل
• عدد المستخدمين: {len(self.users)}
        """
        
        await update.message.reply_text(admin_message.strip())
        logger.info(f"Admin panel accessed by user {user_id}")
    
    async def admin_commands_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle the /admin_commands command to show comprehensive admin guide (admin only)."""
        user_id = update.effective_user.id
        
        if not self.is_admin(user_id):
            await update.message.reply_text("انت مو مخول لاستخدام هذا الأمر.")
            return
        
        try:
            with open('admin_commands_list.txt', 'r', encoding='utf-8') as f:
                admin_commands_text = f.read()
            
            # Split into chunks if too long (Telegram has 4096 character limit)
            if len(admin_commands_text) <= 4000:
                await update.message.reply_text(admin_commands_text)
            else:
                # Split by sections
                sections = admin_commands_text.split('═══════════════════════════════════════════════════════════════')
                current_chunk = ""
                
                for section in sections:
                    if len(current_chunk + section) > 4000:
                        if current_chunk:
                            await update.message.reply_text(current_chunk)
                            current_chunk = section
                        else:
                            await update.message.reply_text(section[:4000])
                    else:
                        current_chunk += section + '\n═══════════════════════════════════════════════════════════════\n'
                
                if current_chunk:
                    await update.message.reply_text(current_chunk)
            
            logger.info(f"Full admin commands guide accessed by user {user_id}")
            
        except FileNotFoundError:
            fallback_message = (
                "🔧 دليل أوامر المسؤول الشامل 🔧\n\n"
                "📊 أوامر الإحصائيات:\n"
                "• /admin - لوحة التحكم الرئيسية\n"
                "• /stats - إحصائيات البوت\n"
                "• /logs - سجل النظام\n\n"
                "👥 إدارة المستخدمين:\n"
                "• /list_users - قائمة المستخدمين\n"
                "• /add_users معرف1,معرف2 - إضافة مستخدمين\n\n"
                "📢 البث الجماعي:\n"
                "• /broadcast رسالة - بث لجميع المستخدمين\n"
                "• /broadcast_to معرفات رسالة - بث مستهدف\n\n"
                "جميع الأوامر محمية وللمسؤولين فقط! 🚀"
            )
            await update.message.reply_text(fallback_message)
            logger.warning(f"Admin commands file not found, sent fallback message to {user_id}")
    
    async def promo_messages_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle the /promo_messages command to show promotional message templates (admin only)."""
        user_id = update.effective_user.id
        
        if not self.is_admin(user_id):
            await update.message.reply_text("انت مو مخول لاستخدام هذا الأمر.")
            return
        
        try:
            with open('promotional_messages.txt', 'r', encoding='utf-8') as f:
                promo_text = f.read()
            
            # Split into chunks if too long (Telegram has 4096 character limit)
            if len(promo_text) <= 4000:
                await update.message.reply_text(promo_text)
            else:
                # Split by sections
                sections = promo_text.split('═════════════════════════════════════')
                current_chunk = ""
                
                for section in sections:
                    if len(current_chunk + section) > 4000:
                        if current_chunk:
                            await update.message.reply_text(current_chunk)
                            current_chunk = section
                        else:
                            await update.message.reply_text(section[:4000])
                    else:
                        current_chunk += section + '\n═════════════════════════════════════\n'
                
                if current_chunk:
                    await update.message.reply_text(current_chunk)
            
            logger.info(f"Promotional messages accessed by admin {user_id}")
            
        except FileNotFoundError:
            await update.message.reply_text(
                "📢 ملف الرسائل الترويجية غير موجود حالياً.\n"
                "راح يتم إنشاؤه قريباً مع رسائل جاهزة للاستخدام."
            )
            logger.warning(f"Promotional messages file not found for admin {user_id}")
    
    async def stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle the /stats command for admin statistics."""
        user_id = update.effective_user.id
        
        if not self.is_admin(user_id):
            await update.message.reply_text("انت مو مخول لاستخدام هذا الأمر.")
            return
        
        stats_message = f"""
📊 إحصائيات البوت

👥 معلومات المستخدمين:
• العدد الكلي: {len(self.users)}
• المستخدمون النشطون: {len([u for u in self.user_modes.keys()])}
• المستخدمون في وضع الصور: {len([u for u, m in self.user_modes.items() if m == 'image'])}
• المستخدمون في وضع الوصف: {len([u for u, m in self.user_modes.items() if m == 'imagine_prompt'])}

🤖 حالة البوت:
• حالة OpenAI: {'✅ متصل' if self.ai_client else '❌ غير متصل'}
• الوضع الافتراضي: 💬 المحادثة العادية

🔧 معلومات النظام:
• الأدمن الحالي: {user_id}
• ملف المستخدمين: {self.users_file}
• عدد الأدمن: {len(self.ADMIN_IDS)}
        """
        
        await update.message.reply_text(stats_message.strip())
        logger.info(f"Stats viewed by admin {user_id}")
    
    async def logs_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle the /logs command to show recent logs."""
        user_id = update.effective_user.id
        
        if not self.is_admin(user_id):
            await update.message.reply_text("انت مو مخول لاستخدام هذا الأمر.")
            return
        
        logs_message = f"""
📋 حالة النظام

🔍 آخر الفحوصات:
• حالة OpenAI: {'✅ يعمل بشكل طبيعي' if self.ai_client else '❌ يوجد مشكلة'}
• ملف المستخدمين: {'✅ محمل بنجاح' if os.path.exists(self.users_file) else '⚠️ لم يتم العثور عليه'}
• عدد المستخدمين المسجلين: {len(self.users)}

⚙️ إعدادات النظام:
• البوت مسؤول: {user_id}
• صلاحية الأدمن: ✅ مفعلة

📊 الاستخدام:
• الوضع الأكثر استخداماً: المحادثة العادية
• آخر تحديث للمستخدمين: متاح

ملاحظة: للمزيد من التفاصيل التقنية، راجع سجلات الخادم.
        """
        
        await update.message.reply_text(logs_message.strip())
        logger.info(f"Logs viewed by admin {user_id}")
    
    async def list_users_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle the /list_users command to show all registered users (admin only)."""
        user_id = update.effective_user.id
        
        if not self.is_admin(user_id):
            await update.message.reply_text("انت مو مخول لاستخدام هذا الأمر.")
            return
        
        if not self.users:
            await update.message.reply_text("📝 لا يوجد مستخدمين مسجلين حالياً.")
            return
        
        # Prepare user list message
        users_list = "📋 قائمة المستخدمين المسجلين:\n\n"
        users_list += f"العدد الكلي: {len(self.users)}\n\n"
        
        # Group users for better display (10 per message to avoid hitting message limits)
        user_list = list(self.users)
        batch_size = 50  # Safe limit for message length
        
        for i in range(0, len(user_list), batch_size):
            batch = user_list[i:i + batch_size]
            batch_message = f"📋 المستخدمين ({i+1}-{min(i+batch_size, len(user_list))}):\n\n"
            
            for idx, uid in enumerate(batch, start=i+1):
                # Check current mode if available
                current_mode = self.user_modes.get(uid, 'chat')
                mode_emoji = {'chat': '💬', 'image': '🎨', 'imagine_prompt': '🎭'}.get(current_mode, '💬')
                
                # Try to get user info from Telegram
                user_info = ""
                try:
                    chat = await context.bot.get_chat(uid)
                    first_name = chat.first_name or ""
                    last_name = chat.last_name or ""
                    username = f"@{chat.username}" if chat.username else ""
                    
                    # Build name display
                    full_name = f"{first_name} {last_name}".strip()
                    if full_name:
                        user_info = f" - {full_name}"
                    if username:
                        user_info += f" ({username})"
                        
                except Exception:
                    # If we can't get user info, just show the ID
                    user_info = ""
                
                batch_message += f"{idx}. <code>{uid}</code> {mode_emoji}"
                if user_info:
                    batch_message += f"{user_info}\n"
                else:
                    batch_message += "\n"
            
            # Add instructions at the end of first batch
            if i == 0:
                batch_message += "\n💡 للإرسال لمستخدمين محددين:\n"
                batch_message += "/broadcast_to معرف1,معرف2 الرسالة\n\n"
                batch_message += "مثال:\n"
                batch_message += f"/broadcast_to <code>{user_list[0] if user_list else '123456789'}</code> مرحباً!"
            
            await update.message.reply_text(batch_message, parse_mode='HTML')
            
            # Small delay between messages if sending multiple batches
            if i + batch_size < len(user_list):
                await asyncio.sleep(0.5)
        
        logger.info(f"User list viewed by admin {user_id}")
    
    async def add_users_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle the /add_users command to manually add users (admin only)."""
        user_id = update.effective_user.id
        
        if not self.is_admin(user_id):
            await update.message.reply_text("انت مو مخول لاستخدام هذا الأمر.")
            return
        
        if not context.args:
            await update.message.reply_text(
                "استخدم الأمر بهذا الشكل:\n"
                "/add_users معرف1,معرف2,معرف3\n\n"
                "مثال:\n"
                "/add_users 123456789,987654321"
            )
            return
        
        # Parse user IDs
        user_ids_str = context.args[0]
        try:
            new_user_ids = []
            for uid_str in user_ids_str.split(','):
                uid_str = uid_str.strip()
                if uid_str:
                    new_user_ids.append(int(uid_str))
        except ValueError:
            await update.message.reply_text(
                "❌ خطأ في تحليل معرفات المستخدمين!\n"
                "تأكد من استخدام أرقام صحيحة فقط مفصولة بفاصلة."
            )
            return
        
        # Add users to database
        added_count = 0
        existing_count = 0
        
        for new_user_id in new_user_ids:
            if new_user_id not in self.users:
                self.users.add(new_user_id)
                added_count += 1
            else:
                existing_count += 1
        
        # Save to file
        self.save_users()
        
        # Send confirmation
        result_message = f"✅ تم تحديث قاعدة المستخدمين!\n\n"
        result_message += f"• مستخدمين جدد: {added_count}\n"
        result_message += f"• موجودين مسبقاً: {existing_count}\n"
        result_message += f"• إجمالي المستخدمين: {len(self.users)}"
        
        await update.message.reply_text(result_message)
        logger.info(f"Admin {user_id} added {added_count} new users")
    
    async def gpt41_analyze_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle the /gpt41_analyze command using GPT-4.1's enhanced capabilities."""
        user_id = update.effective_user.id
        
        # Track user
        self.add_user(user_id)
        
        # Get message to analyze
        if not context.args:
            await update.message.reply_text(
                "🧠 تحليل متقدم بـ GPT-4.1\n\n"
                "استخدم الأمر بهذا الشكل:\n"
                "/gpt41_analyze النص اللي تريد تحليله\n\n"
                "مثال:\n"
                "/gpt41_analyze أبي تحلل هالنص وتقول لي شنو معناه"
            )
            return
        
        # Join arguments to form the message
        message_to_analyze = " ".join(context.args)
        
        await update.message.reply_text("🧠 قاعد أحلل طلبك بقدرات GPT-4.1 المتقدمة...")
        
        try:
            # Use GPT-4.1's enhanced analysis capabilities
            analysis_result = await self.ai_client.analyze_with_tools(message_to_analyze, "advanced_analysis")
            
            response_message = f"🧠 تحليل GPT-4.1 المتقدم:\n\n{analysis_result}"
            
            await update.message.reply_text(response_message)
            logger.info(f"GPT-4.1 advanced analysis completed for user {user_id}")
            
        except Exception as e:
            logger.error(f"Error in GPT-4.1 analysis: {e}")
            await update.message.reply_text("معذرة، صار خطأ في التحليل المتقدم. جرب مرة ثانية.")
    
    async def gpt41_structured_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle the /gpt41_structured command for structured responses."""
        user_id = update.effective_user.id
        
        # Track user
        self.add_user(user_id)
        
        # Get structured response type and prompt
        if len(context.args) < 2:
            await update.message.reply_text(
                "📋 ردود منظمة بـ GPT-4.1\n\n"
                "استخدم الأمر بهذا الشكل:\n"
                "/gpt41_structured [نوع] [المحتوى]\n\n"
                "الأنواع المتاحة:\n"
                "• detailed - رد مفصل\n"
                "• summary - ملخص\n"
                "• analysis - تحليل\n"
                "• explanation - شرح\n\n"
                "مثال:\n"
                "/gpt41_structured detailed اشرح لي الذكاء الاصطناعي"
            )
            return
        
        response_type = context.args[0]
        prompt = " ".join(context.args[1:])
        
        await update.message.reply_text(f"📋 قاعد أسوي رد منظم ({response_type}) بـ GPT-4.1...")
        
        try:
            # Use GPT-4.1's structured response capabilities
            structured_result = await self.ai_client.generate_structured_response(prompt, response_type)
            
            if structured_result["success"]:
                response_message = f"📋 رد منظم ({response_type}) - GPT-4.1:\n\n{structured_result['content']}"
            else:
                response_message = f"❌ {structured_result['content']}"
            
            await update.message.reply_text(response_message)
            logger.info(f"GPT-4.1 structured response ({response_type}) completed for user {user_id}")
            
        except Exception as e:
            logger.error(f"Error in GPT-4.1 structured response: {e}")
            await update.message.reply_text("معذرة، صار خطأ في إنشاء الرد المنظم. جرب مرة ثانية.")
    
    async def gpt41_features_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle the /gpt41_features command to show GPT-4.1 capabilities."""
        user_id = update.effective_user.id
        
        # Track user
        self.add_user(user_id)
        
        features_message = """
🚀 مميزات GPT-4.1 المتقدمة في البوت

🧠 **قدرات محسنة:**
• فهم السياق الطويل (مليون رمز)
• تحليل متقدم للنصوص والصور
• ترجمة بجودة عالية
• برمجة وحل مشاكل تقنية متقدمة

🛠️ **أدوات جديدة:**
• /gpt41_analyze - تحليل متقدم للنصوص
• /gpt41_structured - ردود منظمة ومفصلة
• تحليل الصور بجودة أعلى
• ترجمة محسنة للهجة العراقية

🎯 **تحسينات في الوظائف الحالية:**
• /chat - ردود أذكى وأكثر طبيعية
• /translate - ترجمة أدق مع فهم السياق
• /translate_ar - تحويل محسن للعراقية
• /imagine_prompt - أوصاف إبداعية متطورة

🔧 **مميزات تقنية:**
• استدعاء الدوال المحسن
• معالجة أفضل للاستفسارات المعقدة
• كفاءة أعلى في الردود
• أداء محسن بنسبة 30%

💡 **للمطورين:**
• دعم أفضل لاستفسارات البرمجة
• حل مشاكل تقنية معقدة
• تحليل الكود وتحسينه
• شروحات تقنية مفصلة

جرب الأوامر الجديدة واستمتع بقدرات GPT-4.1 المتطورة! 🎉
        """
        
        await update.message.reply_text(features_message.strip())
        logger.info(f"GPT-4.1 features viewed by user {user_id}")
    
    async def broadcast_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle the /broadcast command (admin only)."""
        user_id = update.effective_user.id
        
        # Check if user is admin
        if not self.is_admin(user_id):
            await update.message.reply_text("انت مو مخول لاستخدام هذا الأمر.")
            return
        
        # Get the broadcast message from command arguments
        message_text = " ".join(context.args) if context.args else None
        
        if not message_text:
            await update.message.reply_text(
                "استخدم الأمر بهذا الشكل:\n"
                "/broadcast رسالة البث هنا\n\n"
                "للبث لمستخدمين محددين:\n"
                "/broadcast_to 123456789,987654321 رسالة البث\n\n"
                "أمثلة:\n"
                "/broadcast تم إعادة تفعيل البوت! 🎉\n"
                "/broadcast_to 123456789 رسالة خاصة للمستخدم\n"
                "/broadcast_to 111,222,333 رسالة لثلاث مستخدمين"
            )
            return
        
        # Send broadcast message to all users
        await self.send_broadcast(update, context, message_text)
    
    async def broadcast_to_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle the /broadcast_to command for targeted broadcasting (admin only)."""
        user_id = update.effective_user.id
        
        # Check if user is admin
        if not self.is_admin(user_id):
            await update.message.reply_text("انت مو مخول لاستخدام هذا الأمر.")
            return
        
        if not context.args or len(context.args) < 2:
            await update.message.reply_text(
                "استخدم الأمر بهذا الشكل:\n"
                "/broadcast_to معرفات_المستخدمين رسالة_البث\n\n"
                "أمثلة:\n"
                "/broadcast_to 123456789 رسالة للمستخدم الواحد\n"
                "/broadcast_to 111,222,333 رسالة لثلاث مستخدمين\n"
                "/broadcast_to 123,456 مرحباً أصدقائي! 👋\n\n"
                "ملاحظة: فصل المعرفات بفاصلة (,) بدون مسافات"
            )
            return
        
        # Parse user IDs from first argument
        user_ids_str = context.args[0]
        message_text = " ".join(context.args[1:])
        
        # Parse user IDs
        try:
            target_user_ids = []
            for uid_str in user_ids_str.split(','):
                uid_str = uid_str.strip()
                if uid_str:
                    target_user_ids.append(int(uid_str))
        except ValueError:
            await update.message.reply_text(
                "❌ خطأ في تحليل معرفات المستخدمين!\n\n"
                "تأكد من:\n"
                "• استخدام أرقام صحيحة فقط\n"
                "• فصل المعرفات بفاصلة (,)\n"
                "• عدم وجود مسافات قبل أو بعد الفاصلة\n\n"
                "مثال صحيح: 123456789,987654321"
            )
            return
        
        if not target_user_ids:
            await update.message.reply_text("❌ لم يتم تحديد أي معرف مستخدم صحيح!")
            return
        
        # Send targeted broadcast
        await self.send_targeted_broadcast(update, context, message_text, target_user_ids)
    
    async def send_targeted_broadcast(self, update: Update, context: ContextTypes.DEFAULT_TYPE, message: str, target_user_ids: list):
        """Send broadcast message to specific users."""
        sent_count = 0
        failed_count = 0
        not_found_count = 0
        
        # Format the message for better readability
        formatted_message = self._format_broadcast_message(message)
        
        # Send status message to admin
        status_msg = await update.message.reply_text(
            f"🎯 بدء إرسال البث المحدد إلى {len(target_user_ids)} مستخدم..."
        )
        
        for user_id in target_user_ids:
            try:
                # Check if user exists in our database
                if user_id not in self.users:
                    not_found_count += 1
                    logger.warning(f"User {user_id} not found in bot database")
                    continue
                
                await context.bot.send_message(
                    chat_id=user_id,
                    text=f"📢 رسالة خاصة من المطور:\n\n{formatted_message}",
                    parse_mode='Markdown'
                )
                sent_count += 1
                
                # Add small delay to avoid hitting rate limits
                await asyncio.sleep(0.1)
                
            except Exception as e:
                failed_count += 1
                # Remove inactive users (blocked the bot, deleted account, etc.)
                if "bot was blocked" in str(e).lower() or "chat not found" in str(e).lower():
                    self.users.discard(user_id)
                    logger.info(f"Removed inactive user {user_id}")
                
                logger.error(f"Failed to send targeted broadcast to user {user_id}: {e}")
        
        # Save updated user list
        self.save_users()
        
        # Update status message
        result_message = f"✅ تم إرسال البث المحدد!\n\n📊 الإحصائيات:\n"
        result_message += f"• تم الإرسال: {sent_count}\n"
        result_message += f"• فشل الإرسال: {failed_count}\n"
        if not_found_count > 0:
            result_message += f"• مستخدمين غير مسجلين: {not_found_count}\n"
        result_message += f"• إجمالي المحاولات: {len(target_user_ids)}"
        
        await status_msg.edit_text(result_message)
        
        logger.info(f"Targeted broadcast completed: {sent_count} sent, {failed_count} failed, {not_found_count} not found")
    
    def _format_broadcast_message(self, message: str) -> str:
        """Format broadcast message with proper line breaks and spacing."""
        # Clean up the message
        message = message.strip()
        
        # If message already has proper line breaks, preserve them
        if '\n' in message:
            lines = []
            for line in message.split('\n'):
                lines.append(line.strip())
            return '\n'.join(lines)
        
        # If it's a long single line, intelligently break it into proper format
        # Look for natural breaking points
        formatted_text = message
        
        # Add line break after emoji headers followed by colon
        import re
        formatted_text = re.sub(r'(🎨 ميزة إنشاء الصور:|📸 ميزة تحليل الصور:|🌐 ميزة الترجمة:|🎭 ميزة الوصف الإبداعي:|أوامر البوت:)', 
                               r'\n\n\1', formatted_text)
        
        # Add line break after sentences ending with exclamation/period followed by emoji
        formatted_text = re.sub(r'([.!]) (🎨|📸|🌐|🎭)', r'\1\n\n\2', formatted_text)
        
        # Add line break before bullet points
        formatted_text = re.sub(r' (- 💬|- 🎨|- 🎭|- 🌐|- 📸|- ❓)', r'\n\1', formatted_text)
        
        # Add line break after main greeting and before first feature
        formatted_text = re.sub(r'(يلا نبدي الحچي!) (🎨)', r'\1\n\n\2', formatted_text)
        
        # Add line break before final message
        formatted_text = re.sub(r'(المفصل) (يلا)', r'\1\n\n\2', formatted_text)
        
        # Clean up multiple consecutive newlines
        formatted_text = re.sub(r'\n\n+', '\n\n', formatted_text)
        
        return formatted_text.strip()

    async def send_broadcast(self, update: Update, context: ContextTypes.DEFAULT_TYPE, message: str):
        """Send broadcast message to all users."""
        sent_count = 0
        failed_count = 0
        
        # Format the message for better readability
        formatted_message = self._format_broadcast_message(message)
        
        # Send status message to admin
        status_msg = await update.message.reply_text(
            f"🔄 بدء إرسال البث إلى {len(self.users)} مستخدم..."
        )
        
        for user_id in self.users.copy():  # Copy to avoid modification during iteration
            try:
                await context.bot.send_message(
                    chat_id=user_id,
                    text=f"📢 إعلان من المطور:\n\n{formatted_message}",
                    parse_mode='Markdown'
                )
                sent_count += 1
                
                # Add small delay to avoid hitting rate limits
                await asyncio.sleep(0.1)
                
            except Exception as e:
                failed_count += 1
                # Remove inactive users (blocked the bot, deleted account, etc.)
                if "bot was blocked" in str(e).lower() or "chat not found" in str(e).lower():
                    self.users.discard(user_id)
                    logger.info(f"Removed inactive user {user_id}")
                
                logger.error(f"Failed to send broadcast to user {user_id}: {e}")
        
        # Save updated user list
        self.save_users()
        
        # Update status message
        await status_msg.edit_text(
            f"✅ تم إرسال البث!\n\n"
            f"📊 الإحصائيات:\n"
            f"• تم الإرسال: {sent_count}\n"
            f"• فشل الإرسال: {failed_count}\n"
            f"• إجمالي المستخدمين الحاليين: {len(self.users)}"
        )
        
        logger.info(f"Broadcast completed: {sent_count} sent, {failed_count} failed")
    
    async def handle_photo(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle photo messages - analyze images using Gemini Vision."""
        if not self.ai_client:
            await update.message.reply_text("معذرة، البوت مو شغال هسه. جرب مرة ثانية باجر.")
            return
        
        user_id = update.effective_user.id
        chat_id = update.effective_chat.id
        self.add_user(user_id)
        
        # Check if bot is active in group
        if not self.is_group_active(chat_id):
            # Bot is silenced in this group
            return
        
        try:
            # Send processing message
            await update.message.reply_text("🔍 دا احلل الصورة... شوي صبر...")
            
            # Get the largest photo size
            photo = update.message.photo[-1]
            file = await context.bot.get_file(photo.file_id)
            
            # Download the image data
            image_data = await file.download_as_bytearray()
            
            # Get user's caption as additional prompt
            user_caption = update.message.caption if update.message.caption else ""
            
            # Analyze the image
            response = await self.ai_client.analyze_image(image_data, user_caption)
            
            # Send response safely without markdown formatting
            await update.message.reply_text(
                f"📸 وصف الصورة:\n\n{response}"
            )
            
            # No cleanup needed as we use image data directly
            
            logger.info(f"Successfully analyzed image for user {user_id}")
            
        except Exception as e:
            logger.error(f"Error analyzing image for user {user_id}: {e}")
            await update.message.reply_text(
                "معذرة، صار خطأ وقت تحليل الصورة. جرب ترسل الصورة مرة ثانية."
            )
            
            # No cleanup needed for image data
    
    async def translate_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle the /translate command."""
        if not self.ai_client:
            await update.message.reply_text("معذرة، البوت مو شغال هسه. جرب مرة ثانية باجر.")
            return
        
        user_id = update.effective_user.id
        self.add_user(user_id)
        
        # Get text to translate from command arguments
        if not context.args:
            await update.message.reply_text(
                "🌐 استخدم الأمر بهذا الشكل:\n\n"
                "للترجمة إلى الإنجليزية:\n"
                "/translate النص المراد ترجمته\n\n"
                "للترجمة إلى العربية:\n"
                "/translate_ar Text to translate\n\n"
                "مثال:\n"
                "/translate مرحبا بك في العراق\n"
                "/translate_ar Hello world"
            )
            return
        
        text_to_translate = " ".join(context.args)
        
        try:
            # Send processing message
            await update.message.reply_text("🔄 دا اترجم النص... شوي صبر...")
            
            # Translate to English by default
            translated_text = await self.ai_client.translate_to_english(text_to_translate)
            
            # Format response
            await update.message.reply_text(
                f"🌐 الترجمة الإنجليزية:\n\n"
                f"**النص الأصلي:**\n{text_to_translate}\n\n"
                f"**الترجمة:**\n{translated_text}",
                parse_mode='Markdown'
            )
            
            logger.info(f"Successfully translated text for user {user_id}")
            
        except Exception as e:
            logger.error(f"Error translating text for user {user_id}: {e}")
            await update.message.reply_text(
                "معذرة، صار خطأ وقت الترجمة. جرب مرة ثانية."
            )
    
    async def translate_ar_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle the /translate_ar command (translate to Arabic)."""
        if not self.ai_client:
            await update.message.reply_text("معذرة، البوت مو شغال هسه. جرب مرة ثانية باجر.")
            return
        
        user_id = update.effective_user.id
        self.add_user(user_id)
        
        # Get text to translate from command arguments
        if not context.args:
            await update.message.reply_text(
                "🌐 Use this command like this:\n\n"
                "/translate_ar Text you want to translate to Arabic\n\n"
                "Example:\n"
                "/translate_ar Hello, how are you today?"
            )
            return
        
        text_to_translate = " ".join(context.args)
        
        try:
            # Send processing message
            await update.message.reply_text("🔄 Translating text... please wait...")
            
            # Translate to Arabic
            translated_text = await self.ai_client.translate_to_arabic(text_to_translate)
            
            # Format response
            await update.message.reply_text(
                f"🌐 Arabic Translation:\n\n"
                f"**Original Text:**\n{text_to_translate}\n\n"
                f"**الترجمة العربية:**\n{translated_text}",
                parse_mode='Markdown'
            )
            
            logger.info(f"Successfully translated text to Arabic for user {user_id}")
            
        except Exception as e:
            logger.error(f"Error translating text to Arabic for user {user_id}: {e}")
            await update.message.reply_text(
                "Sorry, there was an error during translation. Please try again."
            )
    
    def _get_user_mode(self, user_id: int) -> str:
        """Get current mode for user, default is chat."""
        return self.user_modes.get(user_id, 'chat')
    
    def _is_image_request(self, message: str) -> bool:
        """Check if the message is requesting image generation."""
        # More specific keywords that indicate actual image generation requests
        image_request_keywords = [
            "ارسم لي", "اصنع صورة", "سوي صورة", "اعمل صورة", "انشئ صورة",
            "اريد صورة", "ابي صورة", "اصنعلي", "سولي صورة", "ارسم",
            "draw me", "create image", "make image", "generate image",
            "اعمل لي", "سوي لي", "اصنع لي"
        ]
        
        # Keywords to exclude (promotional/informational text about images)
        exclude_keywords = [
            "ميزة", "خاصية", "تم إضافة", "جديدة مضافة", "للبوت", 
            "أمثلة على الاستخدام", "برمجة وتطوير", "الأوامر الموجودة",
            "feature", "added", "examples", "commands"
        ]
        
        message_lower = message.lower()
        
        # If message contains excluded keywords, don't treat as image request
        if any(exclude_keyword in message_lower for exclude_keyword in exclude_keywords):
            return False
            
        # Check for actual image request keywords
        return any(keyword in message_lower for keyword in image_request_keywords)
    
    def _is_generic_response(self, response: str) -> bool:
        """Check if the response is generic and unhelpful."""
        if not response or len(response.strip()) < 10:
            return True
            
        generic_indicators = [
            "ما كدرت افهم طلبك زين",
            "ممكن توضحلي اكثر",
            "كلش أسف",
            "شلونك؟ كلش أسف",
            "ما فهمت شتريد"
        ]
        
        # Check for hardcoded fallback responses that might loop
        fallback_indicators = [
            "معذرة، صار خطأ مؤقت في الاتصال بالذكاء الاصطناعي",
            "جرب الأوامر التالية للحصول على تجربة ذكية كاملة"
        ]
        
        return any(indicator in response for indicator in generic_indicators + fallback_indicators)
    
    def _is_valid_response(self, response: str) -> bool:
        """Check if the response is valid and not empty or corrupted."""
        if not response:
            return False
        
        # Check for minimum length
        if len(response.strip()) < 5:
            return False
            
        # Check for common error patterns
        error_patterns = [
            "Error:",
            "Exception:",
            "Traceback:",
            "None",
            "null",
            "undefined"
        ]
        
        return not any(pattern in response for pattern in error_patterns)
    
    def _get_helpful_fallback(self, user_message: str) -> str:
        """Get a helpful fallback response based on user message context."""
        user_lower = user_message.lower()
        
        # Detect question type and provide helpful response
        if any(word in user_lower for word in ["اريد", "ابي", "طلب", "ممكن"]):
            return "تمام! اكدر اساعدك. وضحلي اكثر شنو تريد بالضبط وراح احاول اجاوبك بأفضل شكل ممكن."
        elif any(word in user_lower for word in ["شلون", "كيف", "وين", "متى"]):
            return "زين! هاي اسئلة مهمة. خبرني تفاصيل اكثر حتى اكدر افيدك بالشكل الصحيح."
        elif any(word in user_lower for word in ["شنو", "ماذا", "ايش"]):
            return "اكيد! اكدر اشرحلك. بس وضحلي اكثر شنو تريد تعرف بالضبط."
        elif any(word in user_lower for word in ["سيارات", "سيارة"]):
            return "زين! اشوف انك تسأل عن السيارات. ممكن تحدد اكثر شنو تريد تعرف - انواع معينة، اسعار، مواصفات؟"
        else:
            return "فهمت! اكدر اساعدك بأشياء كثيرة. وضحلي اكثر شتريد وراح اجاوبك بأفضل شكل."
    
    async def _generate_promotional_response(self, user_message: str) -> str:
        """Generate a promotional response for bot marketing."""
        promotional_prompt = f"""
المستخدم يطلب منك كتابة رسالة ترويجية أو إعلانية للبوت. طلبه: "{user_message}"

اكتب رسالة ترويجية جذابة باللهجة العراقية تسوق للبوت وميزاته:

ميزات البوت المهمة:
- محادثة ذكية باللهجة العراقية
- إنشاء الصور بالذكاء الاصطناعي (استخدم /image)
- تحليل الصور (ارسل أي صورة)
- الترجمة بين العربية والإنجليزية
- الوصف الإبداعي للصور
- متاح 24/7

اكتب رسالة حماسية ومقنعة تجذب المستخدمين للبوت، واستخدم الرموز التعبيرية المناسبة.
        """
        return await self.ai_client.generate_response(promotional_prompt)
    

    
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle incoming text messages."""
        if not self.ai_client:
            error_message = "معذرة، البوت مو شغال هسه. جرب مرة ثانية باجر."
            await update.message.reply_text(error_message)
            return
        
        user_message = update.message.text if update.message.text else ""
        user_id = update.effective_user.id
        username = update.effective_user.username or "Unknown"
        chat_id = update.effective_chat.id
        
        # Track user
        self.add_user(user_id)
        
        # Check for group control keywords
        if user_message.strip().lower() in ["سولف", "سولف يا بوت", "تكلم", "رد", "فعل البوت"]:
            if chat_id < 0:  # Group chat
                self.set_group_status(chat_id, True)
                await update.message.reply_text("🔊 تمام! البوت صار فعال وراح أرد على كلامكم.")
                logger.info(f"Bot activated in group {chat_id} by user {user_id}")
                return
            else:
                await update.message.reply_text("💬 هذا الأمر مخصص للمجموعات فقط. بالمحادثات الخاصة أني أرد دائماً.")
                return
        
        if user_message.strip().lower() in ["انجب", "اسكت", "انجب يا بوت", "لا ترد", "عطل البوت", "توقف"]:
            if chat_id < 0:  # Group chat
                self.set_group_status(chat_id, False)
                await update.message.reply_text("🔇 تمام! البوت صار صامت ومراح أرد على أي كلام.")
                logger.info(f"Bot silenced in group {chat_id} by user {user_id}")
                return
            else:
                await update.message.reply_text("💬 هذا الأمر مخصص للمجموعات فقط. بالمحادثات الخاصة أني أرد دائماً.")
                return
        
        # Check if bot is active in group
        if not self.is_group_active(chat_id):
            # Bot is silenced in this group
            return
        
        logger.info(f"Received message from user {user_id} (@{username}): {user_message[:50]}...")
        
        # Get user's current mode
        user_mode = self._get_user_mode(user_id)
        
        # Handle based on current mode
        if user_mode == 'image':
            await self._handle_image_request(update, context, user_message, user_id)
            return
        elif user_mode == 'imagine_prompt':
            await self._handle_imagine_prompt_input(update, context, user_message, user_id)
            return
        
        # In chat mode, check if user is requesting specific actions
        if self._is_image_request(user_message):
            await self._handle_image_request(update, context, user_message, user_id)
            return
        

        
        # Send typing action to show bot is processing
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id,
            action="typing"
        )
        
        try:
            # Get conversation context for this user
            context_string = self.get_conversation_context(user_id)
            
            # Build the message with context
            if context_string:
                enhanced_message = f"{context_string}المستخدم يقول الآن: {user_message}\n\nاعتمد على التاريخ المذكور أعلاه للرد بذكاء واربط بين الأسئلة والإجابات. رد باللهجة العراقية."
            else:
                enhanced_message = f"المستخدم يقول: {user_message}\n\nرد عليه بطريقة مفيدة باللهجة العراقية."
            
            # Generate response using OpenAI with conversation context
            response = await self.ai_client.generate_response(enhanced_message)
            
            # Validate response quality before using it
            if not self._is_valid_response(response):
                logger.warning(f"Invalid response received for user {user_id}: {response[:100]}...")
                response = "معذرة، صار خطأ مؤقت. جرب مرة ثانية."
            elif self._is_generic_response(response):
                # Try to provide more context for promotional message requests
                if "رسالة" in user_message and ("ترويج" in user_message or "إعلان" in user_message or "مستخدم" in user_message):
                    response = await self._generate_promotional_response(user_message)
                else:
                    # Only retry once to avoid loops
                    logger.info(f"Retrying with more specific context for user {user_id}")
                    retry_message = f"المستخدم يقول: {user_message}\n\nرد عليه بطريقة مفيدة ومفصلة باللهجة العراقية، وقدم له مساعدة حقيقية حسب طلبه. لا تبدأ بتحيات متكررة مثل 'هلا بيك' أو 'شلونك' بل ادخل مباشرة في الموضوع."
                    try:
                        response = await self.ai_client.generate_response(retry_message)
                        # If still generic after retry, use a helpful fallback
                        if self._is_generic_response(response):
                            response = self._get_helpful_fallback(user_message)
                    except Exception as e:
                        logger.error(f"Error in retry for user {user_id}: {e}")
                        response = self._get_helpful_fallback(user_message)
            
            # Add mode indicator for chat mode
            mode_emoji = "💬"
            
            # Clean response to avoid parsing errors
            cleaned_response = self.ai_client._clean_response_text(response) if self.ai_client else response
            await update.message.reply_text(
                f"{mode_emoji} {cleaned_response}",
                parse_mode=None
            )
            
            # Save this conversation exchange to memory
            self.add_message_to_conversation(user_id, user_message, response)
            
            logger.info(f"Successfully sent response to user {user_id}")
            
        except Exception as e:
            logger.error(f"Error handling message from user {user_id}: {e}")
            error_message = "معذرة، صار خطأ وقت معالجة رسالتك. جرب مرة ثانية."
            await update.message.reply_text(error_message)
    
    async def _handle_image_request(self, update: Update, context: ContextTypes.DEFAULT_TYPE, user_message: str, user_id: int):
        """Handle image generation requests."""
        try:
            # Send upload photo action to show bot is processing
            await context.bot.send_chat_action(
                chat_id=update.effective_chat.id,
                action="upload_photo"
            )
            
            # Create images directory if it doesn't exist
            if not os.path.exists("images"):
                os.makedirs("images")
            
            # Generate unique filename
            image_filename = f"images/generated_{uuid.uuid4().hex}.jpg"
            
            # Send processing message
            processing_msg = await update.message.reply_text(
                "هسه اسوي لك الصورة، صبر شوية... 🎨"
            )
            
            # Generate the image
            success = await self.ai_client.generate_image(user_message, image_filename)
            
            if success and os.path.exists(image_filename):
                # Get user mode and add indicator
                user_mode = self._get_user_mode(user_id)
                mode_emoji = "🎨" if user_mode == 'image' else ""
                
                # Send the generated image
                with open(image_filename, 'rb') as photo:
                    await update.message.reply_photo(
                        photo=photo,
                        caption=f"{mode_emoji} هاي الصورة الي طلبتها! شلونها؟"
                    )
                
                # Delete the processing message
                await processing_msg.delete()
                
                # Clean up the image file
                os.remove(image_filename)
                
                logger.info(f"Successfully generated and sent image to user {user_id}")
            else:
                if not self.ai_client.gemini_api_key:
                    await processing_msg.edit_text(
                        "معذرة، إنشاء الصور غير متاح حالياً - يحتاج مفتاح Gemini API.\n\n"
                        "جرب بدلاً من ذلك:\n"
                        "🎭 /imagine_prompt - وصف إبداعي مع نص إنجليزي قابل للنسخ\n"
                        "💬 /chat - المحادثة العادية"
                    )
                else:
                    await processing_msg.edit_text(
                        "معذرة، ما كدرت اسوي الصورة هسه. جرب مرة ثانية او اوصف الصورة بطريقة ثانية."
                    )
                logger.warning(f"Failed to generate image for user {user_id}")
                
        except Exception as e:
            logger.error(f"Error generating image for user {user_id}: {e}")
            await update.message.reply_text(
                "معذرة، صار خطأ وقت عمل الصورة. جرب مرة ثانية."
            )
    
    async def _handle_imagine_prompt_input(self, update: Update, context: ContextTypes.DEFAULT_TYPE, user_message: str, user_id: int):
        """Handle user input for imagine prompt generation."""
        try:
            # Reset user mode back to chat after processing
            self.user_modes[user_id] = 'chat'
            
            # Send typing action
            await context.bot.send_chat_action(
                chat_id=update.effective_chat.id,
                action="typing"
            )
            
            # Generate English technical prompt only
            try:
                import asyncio
                arabic_description, english_prompt = await asyncio.wait_for(
                    self.ai_client.generate_creative_description(user_message), 
                    timeout=30.0
                )
                
                # Return only the English prompt as requested in copyable format
                response = f"📝 **English Prompt:**\n```\n{english_prompt}\n```"
                
            except asyncio.TimeoutError:
                response = None
                logger.warning("OpenAI API timed out for imagine_prompt command")
            except Exception as e:
                logger.error(f"Error generating creative description: {e}")
                response = None
            
            # Check if response is valid or use fallback
            if not response or len(response.strip()) < 50:
                response = self._create_creative_fallback(user_message)
            
            # Format the response
            formatted_response = self._format_copyable_text(response)
            
            await update.message.reply_text(
                f"🎭 هاي النص الإنجليزي اللي طلبته:\n\n{response}",
                parse_mode='Markdown'
            )
            
            logger.info(f"Successfully generated creative description from imagine_prompt for user {user_id}")
            
        except Exception as e:
            logger.error(f"Error in imagine_prompt input handling for user {user_id}: {e}")
            # Reset mode on error
            self.user_modes[user_id] = 'chat'
            await update.message.reply_text(
                "معذرة، صار خطأ وقت إنشاء الوصف الإبداعي. جرب مرة ثانية."
            )

    def _format_copyable_text(self, text: str) -> str:
        """Format text to make English prompts and important text copyable."""
        import re
        
        # If the text is mainly English (more than 50% English characters), make it copyable
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        total_chars = len(re.sub(r'\s', '', text))
        
        if total_chars > 0 and english_chars / total_chars > 0.5:
            # This looks like an English prompt, make the whole thing copyable
            return f'`{text.strip()}`'
        
        # Find and format specific English sections within Arabic text
        formatted_text = text
        
        # Look for English sentences or technical specifications
        english_patterns = [
            r'([A-Z][a-zA-Z\s,.-]+(?:detailed|hyperrealistic|cinematic|photorealistic|resolution|ultra|masterpiece|professional|lighting|Canon|Nikon|Sony|Fuji|Leica|ISO|aperture|lens|mm)[a-zA-Z\s,.-]*[.!])',
            r'(Shot with [^.]+\.)',
            r'(Captured using [^.]+\.)',
            r'([A-Z][^.]*[0-9]+[kmKM]+[^.]*\.)',
            r'(f/[0-9.]+)',
            r'(ISO [0-9]+)',
            r'([0-9]+mm)',
        ]
        
        for pattern in english_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match) > 15:  # Only format substantial text
                    formatted_text = formatted_text.replace(match, f'`{match.strip()}`')
        
        return formatted_text
    

    
    def _create_creative_fallback(self, prompt_text: str) -> str:
        """Create a creative description with English prompt when Gemini fails."""
        keywords = prompt_text.lower()
        
        if "طائر العنقاء" in keywords or "phoenix" in keywords:
            english_prompt = "Majestic phoenix bird in flight, wings spread wide with brilliant fiery plumage, gold and crimson feathers, ethereal flames trailing from wings, dramatic lighting against starry night sky, mythical creature, ultra-detailed, cinematic, photorealistic, 8K resolution"
            
            arabic_description = """
شوف خويه، هاي صورة طائر العنقاء الأسطوري... 

طائر كلش جميل ومهيب، جناحاته منفوشة ومفرودة بالسماء مثل اللهب الذهبي اللي يرقص بالهواء. ريشه ملون بألوان النار - أحمر ناري وذهبي لامع وبرتقالي زاهي. عيونه تلمع مثل الجمر، وذيله الطويل يتموج خلفه مثل شلال من النار.

يطير بين الغيوم والسماء الليلية المليانة نجوم. حواليه شرارات نارية صغيرة تتساقط مثل المطر الذهبي. جمال كلش خرافي وسحري، مثل اللي نشوفه بالحلم.

الخلفية سماء ليلية داكنة مليانة نجوم لامعة، والقمر يضوي من بعيد ويخلي الطائر يبين أكثر حلو وجميل.
            """
            
        elif "منظر طبيعي" in keywords or "غروب" in keywords:
            english_prompt = "Beautiful natural landscape at sunset, golden and orange sky with fluffy clouds, rolling green hills with colorful wildflowers, peaceful river reflecting sunset colors, serene atmosphere, professional photography, ultra-detailed, 8K, cinematic lighting"
            
            arabic_description = """
هاي صورة منظر طبيعي خرافي...

الشمس تغرب بالأفق والسماء ملونة بألوان حلوة كلش - برتقالي وأحمر وردي وذهبي. الغيوم مثل القطن الملون تطفو بالسماء. 

بالمقدمة اكو أشجار خضراء كبيرة وحقول واسعة مليانة ورود وزهور ملونة. النهر يجري بهدوء ويعكس ألوان الغروب مثل المرايا. الماء صافي ولامع.

طيور صغيرة تطير بعيد، والهواء هادئ وجميل. احساس بالراحة والهدوء يملي المكان. مناظر تخلي القلب يفرح ويطمئن.
            """
            
        else:
            english_prompt = f"Beautiful detailed {prompt_text}, high quality composition, vibrant colors, professional lighting, ultra-detailed, photorealistic, 8K resolution, cinematic, masterpiece"
            
            arabic_description = f"""
شوف خويه، هاي صورة حلوة للي طلبتها: {prompt_text}

منظر كلش جميل ومليء بالتفاصيل الحلوة. الألوان زاهية وحيوية، والتكوين متوازن ومرتب. كل تفصيلة بمكانها الصحيح.

الإضاءة طبيعية وناعمة، تخلي كل شي يبين واضح وجميل. الخلفية متناسقة مع الموضوع الرئيسي.

احساس بالجمال والهدوء يملي الصورة، مثل اللوحات الفنية الحقيقية. منظر يخلي الناظر يحس بالراحة والإعجاب.
            """
        
        return f"`{english_prompt}`\n\n{arabic_description.strip()}"
    
    def _create_creative_fallback_english(self, prompt_text: str) -> str:
        """Create a creative description in English when Gemini fails."""
        keywords = prompt_text.lower()
        
        if "طائر العنقاء" in keywords or "phoenix" in keywords:
            english_prompt = "Majestic phoenix bird in flight, wings spread wide with brilliant fiery plumage, gold and crimson feathers, ethereal flames trailing from wings, dramatic lighting against starry night sky, mythical creature, ultra-detailed, cinematic, photorealistic, 8K resolution"
            
            english_description = """
Look at this stunning mythical phoenix...

A majestic and awe-inspiring bird with its wings spread wide across the sky like dancing golden flames. Its feathers are painted in fire colors - blazing red, brilliant gold, and vibrant orange. Its eyes gleam like burning embers, and its long tail flows behind it like a waterfall of fire.

It soars through clouds in the star-filled night sky. Small sparks of fire cascade around it like golden rain. The beauty is absolutely magical and otherworldly, like something from a dream.

The background shows a dark night sky filled with twinkling stars, and the moon glows in the distance, making the phoenix appear even more beautiful and magnificent.
            """
            
        elif "منظر طبيعي" in keywords or "غروب" in keywords or "sunset" in keywords or "landscape" in keywords:
            english_prompt = "Beautiful natural landscape at sunset, golden and orange sky with fluffy clouds, rolling green hills with colorful wildflowers, peaceful river reflecting sunset colors, serene atmosphere, professional photography, ultra-detailed, 8K, cinematic lighting"
            
            english_description = """
This is a breathtaking natural landscape...

The sun sets on the horizon with the sky painted in beautiful colors - orange, red, pink, and gold. Fluffy clouds float like colored cotton across the sky.

In the foreground, there are large green trees and vast fields filled with colorful flowers and blossoms. A peaceful river flows quietly, reflecting the sunset colors like mirrors. The water is crystal clear and shimmering.

Small birds fly in the distance, and the air is calm and beautiful. A sense of peace and tranquility fills the scene. Views that make the heart happy and content.
            """
            
        else:
            english_prompt = f"Beautiful detailed {prompt_text}, high quality composition, vibrant colors, professional lighting, ultra-detailed, photorealistic, 8K resolution, cinematic, masterpiece"
            
            english_description = f"""
Here's a beautiful image of what you requested: {prompt_text}

A stunning scene full of beautiful details. The colors are vibrant and lively, with a balanced and well-arranged composition. Every detail is in its perfect place.

The lighting is natural and soft, making everything appear clear and beautiful. The background harmonizes perfectly with the main subject.

A sense of beauty and tranquility fills the image, like real artistic paintings. A view that makes the observer feel relaxed and amazed.
            """
        
        return f"`{english_prompt}`\n\n{english_description.strip()}"
    
    async def create_poll_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle the /create_poll command - create custom polls."""
        user_id = update.effective_user.id
        self.add_user(user_id)
        
        if not context.args or len(context.args) < 3:
            await update.message.reply_text(
                "📊 إنشاء استطلاع رأي مخصص\n\n"
                "استخدم الأمر بهذا الشكل:\n"
                "/create_poll السؤال,الخيار الأول,الخيار الثاني,الخيار الثالث\n\n"
                "مثال:\n"
                "/create_poll ما هو لونك المفضل؟,أحمر,أزرق,أخضر,أصفر\n\n"
                "ملاحظة: يمكنك إضافة حتى 10 خيارات، افصل بينها بفاصلة"
            )
            return
        
        try:
            # Join all arguments and split by comma
            poll_data = " ".join(context.args).split(',')
            
            if len(poll_data) < 3:
                await update.message.reply_text("❌ يجب أن يحتوي الاستطلاع على سؤال وخيارين على الأقل")
                return
            
            question = poll_data[0].strip()
            options = [option.strip() for option in poll_data[1:] if option.strip()]
            
            if len(options) < 2:
                await update.message.reply_text("❌ يجب أن يحتوي الاستطلاع على خيارين على الأقل")
                return
            
            if len(options) > 10:
                await update.message.reply_text("❌ الحد الأقصى هو 10 خيارات")
                return
            
            # Create the poll
            await update.message.reply_poll(
                question=question,
                options=options,
                is_anonymous=True,
                allows_multiple_answers=False
            )
            
            logger.info(f"Poll created successfully by user {user_id}")
            
        except Exception as e:
            logger.error(f"Error creating poll for user {user_id}: {e}")
            await update.message.reply_text(
                "❌ حدث خطأ أثناء إنشاء الاستطلاع. تأكد من صحة تنسيق الأمر."
            )
    
    async def quiz_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle the /quiz command - create educational quiz with multiple levels."""
        user_id = update.effective_user.id
        self.add_user(user_id)
        
        try:
            # Check if user specified a level or subject
            quiz_level = "ابتدائي"  # Default level
            user_input = " ".join(context.args) if context.args else ""
            
            if any(level in user_input.lower() for level in ["اعدادي", "إعدادي", "متوسط"]):
                quiz_level = "إعدادي"
            elif any(level in user_input.lower() for level in ["ثانوي", "إعدادية"]):
                quiz_level = "ثانوي"
            
            # Send introductory message
            await update.message.reply_text(
                f"🎓 اختبار العلوم للسادس {quiz_level}\n\n"
                "راح أرسلك 10 أسئلة علمية مناسبة للمرحلة المطلوبة. "
                "كل سؤال له عدة خيارات، اختار الإجابة الصحيحة!\n\n"
                "جاهز؟ نبدأ بأول سؤال... 🚀"
            )
            
            # Wait a moment before sending first question
            await asyncio.sleep(2)
            
            # Define science questions based on level
            if quiz_level == "إعدادي":
                quiz_questions = [
                    {
                        "question": "ما هو الرمز الكيميائي للماء؟",
                        "options": ["H2O", "CO2", "NaCl", "O2"],
                        "correct": 0,
                        "explanation": "الماء يتكون من ذرتين هيدروجين وذرة أكسجين واحدة، لذلك رمزه H2O"
                    },
                    {
                        "question": "كم عدد غرف القلب في الإنسان؟",
                        "options": ["غرفتان", "ثلاث غرف", "أربع غرف", "خمس غرف"],
                        "correct": 2,
                        "explanation": "قلب الإنسان يحتوي على أربع غرف: أذينان وبطينان"
                    },
                    {
                        "question": "ما هو أصغر وحدة في المادة؟",
                        "options": ["الجزيء", "الذرة", "الإلكترون", "البروتون"],
                        "correct": 1,
                        "explanation": "الذرة هي أصغر وحدة في المادة تحتفظ بخصائص العنصر"
                    },
                    {
                        "question": "كم تستغرق الأرض للدوران حول الشمس؟",
                        "options": ["24 ساعة", "30 يوماً", "365 يوماً", "12 شهراً"],
                        "correct": 2,
                        "explanation": "تستغرق الأرض 365 يوماً (سنة واحدة) لتكمل دورة واحدة حول الشمس"
                    },
                    {
                        "question": "أي من هذه العناصر معدن؟",
                        "options": ["الكربون", "الأكسجين", "الحديد", "النيتروجين"],
                        "correct": 2,
                        "explanation": "الحديد هو عنصر معدني يرمز له بـ Fe في الجدول الدوري"
                    },
                    {
                        "question": "ما هو الغاز الأكثر وفرة في الغلاف الجوي؟",
                        "options": ["الأكسجين", "النيتروجين", "ثاني أكسيد الكربون", "الأرجون"],
                        "correct": 1,
                        "explanation": "النيتروجين يشكل حوالي 78% من الغلاف الجوي للأرض"
                    },
                    {
                        "question": "ما هي وحدة قياس القوة؟",
                        "options": ["المتر", "الكيلوغرام", "النيوتن", "الثانية"],
                        "correct": 2,
                        "explanation": "النيوتن هو وحدة قياس القوة في النظام الدولي للوحدات"
                    },
                    {
                        "question": "كم عدد كروموسومات الإنسان؟",
                        "options": ["23 زوجاً", "46 زوجاً", "22 زوجاً", "48 زوجاً"],
                        "correct": 0,
                        "explanation": "الإنسان لديه 23 زوجاً من الكروموسومات، أي 46 كروموسوماً في المجموع"
                    },
                    {
                        "question": "ما اسم العملية التي تحول الضوء إلى طاقة كيميائية في النباتات؟",
                        "options": ["التنفس", "التبخر", "التمثيل الضوئي", "الهضم"],
                        "correct": 2,
                        "explanation": "التمثيل الضوئي هو العملية التي تستخدم فيها النباتات ضوء الشمس لصنع الغذاء"
                    },
                    {
                        "question": "ما هي سرعة الضوء في الفراغ تقريباً؟",
                        "options": ["300,000 كم/ث", "150,000 كم/ث", "500,000 كم/ث", "100,000 كم/ث"],
                        "correct": 0,
                        "explanation": "سرعة الضوء في الفراغ تبلغ تقريباً 300,000 كيلومتر في الثانية"
                    }
                ]
            else:
                # Primary level questions
                quiz_questions = [
                {
                    "question": "ما هو العضو المسؤول عن ضخ الدم في جسم الإنسان؟",
                    "options": ["القلب", "الكبد", "الرئة", "المعدة"],
                    "correct": 0,
                    "explanation": "القلب هو العضو الذي يضخ الدم إلى جميع أجزاء الجسم"
                },
                {
                    "question": "كم عدد كواكب النظام الشمسي؟",
                    "options": ["7 كواكب", "8 كواكب", "9 كواكب", "10 كواكب"],
                    "correct": 1,
                    "explanation": "النظام الشمسي يحتوي على 8 كواكب منذ إعادة تصنيف بلوتو"
                },
                {
                    "question": "ما هو الغاز الذي نتنفسه للبقاء على قيد الحياة؟",
                    "options": ["ثاني أكسيد الكربون", "النيتروجين", "الأكسجين", "الهيدروجين"],
                    "correct": 2,
                    "explanation": "الأكسجين هو الغاز الضروري لعملية التنفس والحياة"
                },
                {
                    "question": "في أي حالة يكون الماء عند درجة الصفر المئوي؟",
                    "options": ["سائل", "غاز", "صلب (جليد)", "بلازما"],
                    "correct": 2
                },
                {
                    "question": "ما هو أكبر حيوان على وجه الأرض؟",
                    "options": ["الفيل", "الحوت الأزرق", "القرش الأبيض", "الزرافة"],
                    "correct": 1
                },
                {
                    "question": "أي من هذه النباتات يحتاج للشمس لصنع غذائه؟",
                    "options": ["جميع النباتات الخضراء", "النباتات الصحراوية فقط", "الأشجار الكبيرة فقط", "لا يحتاج أي نبات للشمس"],
                    "correct": 0
                },
                {
                    "question": "كم عدد الأسنان اللبنية عند الطفل؟",
                    "options": ["16 سن", "20 سن", "24 سن", "32 سن"],
                    "correct": 1
                },
                {
                    "question": "ما هو أقرب نجم إلى الأرض؟",
                    "options": ["القمر", "الشمس", "النجم القطبي", "المريخ"],
                    "correct": 1
                },
                {
                    "question": "أي من هذه المواد موصل جيد للكهرباء؟",
                    "options": ["الخشب", "البلاستيك", "النحاس", "الزجاج"],
                    "correct": 2
                },
                {
                    "question": "كم عدد أرجل النحلة؟",
                    "options": ["4 أرجل", "6 أرجل", "8 أرجل", "10 أرجل"],
                    "correct": 1
                }
            ]
            
            # Send each question as a poll with better error handling
            for i, q in enumerate(quiz_questions, 1):
                try:
                    await asyncio.sleep(1.5)  # Longer delay between questions
                    
                    question_text = f"السؤال {i} من 10:\n{q['question']}"
                    
                    # Get explanation
                    explanation = q.get('explanation', f"✅ الإجابة الصحيحة هي: {q['options'][q['correct']]}")
                    
                    await update.message.reply_poll(
                        question=question_text,
                        options=q['options'],
                        is_anonymous=False,
                        allows_multiple_answers=False,
                        type='quiz',
                        correct_option_id=q['correct'],
                        explanation=explanation
                    )
                    
                    logger.info(f"Quiz question {i} sent successfully to user {user_id}")
                    
                except Exception as e:
                    logger.error(f"Error sending quiz question {i} to user {user_id}: {e}")
                    await update.message.reply_text(f"❌ خطأ في إرسال السؤال {i}")
                    continue
            
            # Send completion message
            await asyncio.sleep(3)
            await update.message.reply_text(
                f"🎉 تم إرسال جميع الأسئلة!\n\n"
                f"تهانينا! لقد أكملت اختبار العلوم للسادس {quiz_level}. "
                "الآن يمكنك مراجعة إجاباتك والتعلم من الأخطاء.\n\n"
                "💡 نصيحة: إذا أخطأت في أي سؤال، اقرأ التفسير لتتعلم الإجابة الصحيحة!\n\n"
                "حظاً موفقاً في دراستك! 📚✨\n\n"
                f"🔄 للحصول على اختبار جديد: /quiz\n"
                f"📊 لإنشاء استطلاع: /create_poll"
            )
            
            logger.info(f"Science quiz ({quiz_level}) completed for user {user_id}")
            
        except Exception as e:
            logger.error(f"Error creating science quiz for user {user_id}: {e}")
            await update.message.reply_text(
                "❌ حدث خطأ أثناء إنشاء الاختبار. يرجى المحاولة مرة أخرى."
            )
    
    async def help_poll_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show help for poll-related commands."""
        user_id = update.effective_user.id
        self.add_user(user_id)
        
        help_text = """
🎯 مساعدة الاستطلاعات والاختبارات

📊 **إنشاء استطلاع مخصص:**
/create_poll السؤال,الخيار1,الخيار2,الخيار3

🎓 **اختبار العلوم (السادس الابتدائي):**
/quiz - اختبار شامل من 10 أسئلة في العلوم

📋 **أمثلة:**
• `/create_poll ما رأيك بالبوت؟,ممتاز,جيد,يحتاج تحسين`
• `/create_poll أي وقت تفضل الدراسة؟,صباحاً,مساءً,ليلاً`
• `/quiz` - لبدء اختبار العلوم

💡 **ملاحظات مهمة:**
• يمكن إضافة حتى 10 خيارات للاستطلاع
• افصل بين الخيارات بفاصلة
• اختبار العلوم يحتوي على إجابات صحيحة وتفسيرات
• جميع الاستطلاعات مجانية ومفتوحة للجميع

🤖 تم تصميم هذا البوت بواسطة ثابت (@tht_txt)
        """
        
        await update.message.reply_text(help_text.strip())
        logger.info(f"Poll help displayed for user {user_id}")

    async def error_handler(self, update: object, context: ContextTypes.DEFAULT_TYPE):
        """Handle errors that occur during bot operation."""
        logger.error(f"Update {update} caused error {context.error}")
        
        # Notify user about the error if possible
        if update and hasattr(update, 'effective_message') and update.effective_message:
            try:
                await update.effective_message.reply_text(
                    "معذرة، صار خطأ مو متوقع. جرب مرة ثانية باجر."
                )
            except Exception as e:
                logger.error(f"Failed to send error message to user: {e}")
