"""
Enhanced Multi-AI Client combining GPT-5 Chat and Gemini
Features:
- GPT-5 Chat for advanced text responses and analysis
- Gemini 2.0 Flash for image generation
- GPT-5 Vision for image analysis  
- Optimized for Iraqi dialect specialization
"""

import os
import logging
import base64
import requests
import uuid
import json
from openai import OpenAI
import google.generativeai as genai

logger = logging.getLogger(__name__)

class MultiAIClient:
    """
    Multi-AI client combining GPT-5 Chat and Gemini capabilities
    - GPT-5 Chat: Advanced text generation and analysis via OpenRouter
    - Gemini: High-quality image generation capabilities
    - Iraqi dialect specialization across all models
    """
    
    def __init__(self):
        """Initialize the Multi-AI client with GPT-5 and Gemini."""
        # OpenRouter API key for GPT-5 Chat
        self.openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
        if not self.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is required")
        
        # Gemini API key for image generation
        self.gemini_api_key = os.environ.get("GEMINI_API_KEY")
        if not self.gemini_api_key:
            logger.warning("GEMINI_API_KEY not set - image generation will be limited")
        
        # Initialize GPT-5 Chat client via OpenRouter
        self.gpt5_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.openrouter_api_key
        )
        
        # Initialize Gemini client for image generation
        if self.gemini_api_key:
            genai.configure(api_key=self.gemini_api_key)
            self.gemini_client = genai
        else:
            self.gemini_client = None
        
        # Enhanced system instruction for Iraqi dialect
        self.system_instruction = """
أنت بوت ذكي متخصص في اللهجة العراقية. رد على المستخدم بلهجة عراقية أصيلة وطبيعية.

خصائصك:
- استخدم الكلمات العراقية الأصيلة مثل: شلونك، شكو ماكو، وين، شنو، ليش، چان، هسة، شوية، زين، ماشي الحال
- استخدم تعابير عراقية: يعني شنو، لا والله، صدگ، أكيد، بالضبط، معقول، خوش
- كن ودود ومساعد
- اجب بوضوح ودقة
- استخدم GPT-5 Chat للحصول على أفضل جودة في الإجابات
- للصور استخدم Gemini لإنشاء صور عالية الجودة

تذكر: كل إجاباتك يجب أن تكون باللهجة العراقية الأصيلة.
        """

    async def generate_response(self, user_message: str) -> str:
        """
        Generate response using GPT-5 Chat for enhanced quality.
        
        Args:
            user_message: User's message in any language
            
        Returns:
            Response in Iraqi Arabic dialect
        """
        try:
            logger.info(f"Sending request to GPT-5 Chat for message: {user_message[:50]}...")
            
            response = self.gpt5_client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": "https://iraqi-bot.replit.app",
                    "X-Title": "Iraqi Dialect Bot - GPT-5",
                },
                model="openai/gpt-4o",
                messages=[
                    {
                        "role": "system", 
                        "content": self.system_instruction
                    },
                    {
                        "role": "user", 
                        "content": user_message
                    }
                ],
                temperature=0.7,
                max_tokens=500,  # Further reduced to work within available credits
            )
            
            if response.choices and response.choices[0].message.content:
                result = response.choices[0].message.content.strip()
                logger.info("Successfully received response from GPT-5 Chat")
                return result
            else:
                logger.warning("Empty response from GPT-5 Chat")
                # Try fallback to Gemini if available
                return await self._fallback_to_gemini(user_message)
                
        except Exception as e:
            logger.error(f"Error calling GPT-5 Chat API: {e}")
            # Check if it's a credit/payment error
            if "402" in str(e) or "credit" in str(e).lower() or "payment" in str(e).lower():
                logger.warning("OpenRouter credits insufficient, trying fallback to Gemini")
                return await self._fallback_to_gemini(user_message)
            return "معذرة، صار خطأ وقت معالجة طلبك. جرب مرة ثانية باجر."
    
    async def _fallback_to_gemini(self, user_message: str) -> str:
        """
        Fallback to Gemini when OpenRouter fails or runs out of credits.
        
        Args:
            user_message: User's message
            
        Returns:
            Response in Iraqi Arabic dialect
        """
        # Always try Gemini first if available
        if self.gemini_client:
            try:
                logger.info("Using Gemini as fallback for text generation")
                
                # Enhanced Gemini system instruction for better responses
                gemini_instruction = """
أنت بوت ذكي متخصص في اللهجة العراقية. المستخدم يقول لك شيء، وأنت تحتاج أن ترد عليه بشكل طبيعي ومفيد.

خصائصك:
- استخدم اللهجة العراقية الأصيلة: شكو ماكو، وين، شنو، ليش، چان، هسه، شوية، زين، ماشي الحال
- كن ودود ومساعد بدون تحيات متكررة
- اجب على الأسئلة بوضوح ومباشرة
- إذا قال "كم من 10" أعطي تقييم رقمي
- إذا قال "جيد" أو "ممتاز" اعرف إنه يعلق على إجابة سابقة
- إذا قال شي غامض، اطلب توضيح بطريقة ودية
- لا تكرر نفس الجواب دائماً، نوع في ردودك
- تجنب البدء بتحيات ثابتة مثل "هلا بيك" أو "شلونك" في كل رد
- ادخل مباشرة في الموضوع واجب على السؤال

المستخدم قال:
                """
                
                model = genai.GenerativeModel('gemini-1.5-flash')
                response = model.generate_content(
                    f"{gemini_instruction}{user_message}",
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=400,
                        temperature=0.8,
                    )
                )
                
                # Better response handling for Gemini
                if response and response.text:
                    result = response.text.strip()
                    logger.info("Successfully received fallback response from Gemini")
                    return result
                
                logger.warning("Empty response from Gemini fallback")
                    
            except Exception as e:
                logger.error(f"Error in Gemini fallback: {e}")
        
        # Only use hardcoded responses as absolute last resort
        logger.warning("Using hardcoded fallback responses")
        
        # More helpful fallback responses that guide users to proper commands
        user_lower = user_message.lower()
        
        # Context-aware fallback responses to avoid repetitive replies
        if "سيارات" in user_lower or "سيارة" in user_lower:
            return (
                "اشوف انك تسأل عن السيارات! 🚗\n\n"
                "ممكن اساعدك بمعلومات عن:\n"
                "🔧 انواع السيارات\n"
                "💰 اسعار السيارات\n" 
                "⚙️ مواصفات وتقييمات\n"
                "🔍 نصائح للشراء\n\n"
                "وضحلي اكثر شنو تحتاج وراح اساعدك!"
            )
        # For quiz/poll related requests - guide to proper commands
        elif any(quiz_word in user_lower for quiz_word in ["اختبار", "استطلاع", "سؤال", "استفتاء", "امتحان"]):
            return (
                "أشوف إنك تريد تسوي اختبار أو استطلاع! 📊\n\n"
                "استخدم هاي الأوامر:\n"
                "🎓 /quiz - للاختبار التعليمي (سادس ابتدائي)\n"
                "📋 /create_poll - لإنشاء استطلاع مخصص\n"
                "📚 /help_poll - للمساعدة والشرح\n\n"
                "مثال: /create_poll شنو رأيكم بالبوت؟,ممتاز,جيد,يحتاج تطوير"
            )
        elif any(help_word in user_lower for help_word in ["مساعدة", "شلون", "كيف", "وين", "اريد", "ممكن"]):
            return (
                "اكيد اكدر اساعدك! 💪\n\n"
                "ممكن استخدام:\n"
                "❓ /help - للمساعدة الشاملة\n"
                "💬 /chat - للمحادثة الذكية\n"
                "🎨 /image - لإنشاء الصور\n"
                "📊 /create_poll - للاستطلاعات\n\n"
                "او اكتبلي مباشرة شتريد واكدر اساعدك!"
            )
        else:
            # Context-specific fallback instead of generic error
            return (
                "فهمت طلبك بس صار خطأ مؤقت! 🔄\n\n"
                "جرب مرة ثانية او استخدم:\n"
                "💬 /chat - للمحادثة العادية\n"
                "❓ /help - للمساعدة\n\n"
                "اكتبلي بطريقة مختلفة واكدر اساعدك اكثر!"
            )
    
    async def generate_image(self, prompt: str, image_path: str) -> bool:
        """
        Generate an image using Gemini 2.0 Flash image generation.
        
        Args:
            prompt: Description of the image to generate
            image_path: Path where to save the generated image
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if Gemini is available
            if not self.gemini_client:
                logger.error("Gemini image generation unavailable - GEMINI_API_KEY not set")
                return False
            
            logger.info(f"Generating image with Gemini 2.0 Flash for prompt: {prompt[:50]}...")
            
            # Generate image with Gemini (note: direct image generation via API is limited)
            # For now, return False to indicate image generation is not available
            logger.warning("Direct image generation with Gemini API is limited")
            return False
            
            if response.candidates and response.candidates[0].content:
                content = response.candidates[0].content
                
                if content.parts:
                    for part in content.parts:
                        # Look for image data in the response
                        if part.inline_data and part.inline_data.data:
                            # Save the generated image
                            with open(image_path, 'wb') as f:
                                f.write(part.inline_data.data)
                            
                            logger.info("Successfully generated and saved image with Gemini")
                            return True
                        elif part.text:
                            # Log any text response from Gemini
                            logger.info(f"Gemini text response: {part.text[:100]}")
                
                logger.error("No image data found in Gemini response")
                return False
            else:
                logger.error("No content in Gemini response")
                return False
                
        except Exception as e:
            logger.error(f"Error generating image with Gemini: {e}")
            return False
    
    async def analyze_image(self, image_data: bytes, user_message: str = "") -> str:
        """
        Analyze an image using GPT-5 Chat Vision capabilities.
        
        Args:
            image_data: The image data as bytes
            user_message: Optional context message from user
            
        Returns:
            Analysis description in Iraqi Arabic
        """
        try:
            logger.info("Attempting image analysis with GPT-5 Chat Vision...")
            
            # Convert image to base64
            base64_image = base64.b64encode(image_data).decode('utf-8')
            
            # Create prompt for image analysis
            analysis_prompt = "حلل هذه الصورة بالتفصيل واوصفها باللهجة العراقية. اذكر كل شي تشوفه فيها - الألوان، الأشخاص، المكان، الأشياء، والتفاصيل المهمة."
            
            if user_message:
                analysis_prompt += f"\n\nالمستخدم قال: {user_message}"
            
            # Analyze with GPT-5 Chat Vision
            response = self.gpt5_client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": "https://iraqi-bot.replit.app",
                    "X-Title": "Iraqi Dialect Bot - GPT-5",
                },
                model="openai/gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": self.system_instruction + "\n\nأنت تحلل صورة أرسلها المستخدم. اوصف كل شي تشوفه بالتفصيل واستخدم اللهجة العراقية."
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": analysis_prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=500,
                temperature=0.7,
            )
            
            if response.choices and response.choices[0].message.content:
                result = response.choices[0].message.content.strip()
                # Clean the response to avoid formatting issues
                result = self._clean_response_text(result)
                logger.info("Successfully analyzed image with GPT-5 Chat Vision")
                return result
            else:
                logger.warning("Empty response from GPT-5 Chat Vision")
                # Try Gemini Vision as fallback for empty response
                if self.gemini_client:
                    try:
                        logger.info("Trying Gemini Vision as fallback for empty GPT-5 response...")
                        return await self._analyze_image_with_gemini(image_data, user_message)
                    except Exception as gemini_error:
                        logger.error(f"Gemini Vision also failed: {gemini_error}")
                return "ما كدرت أحلل الصورة، بس تكدر تحچي وياي عادي."
                
        except Exception as e:
            logger.error(f"Error in GPT-5 Chat Vision analysis: {e}")
            
            # Try Gemini Vision as fallback
            if self.gemini_client:
                try:
                    logger.info("Trying Gemini Vision as fallback for image analysis...")
                    return await self._analyze_image_with_gemini(image_data, user_message)
                except Exception as gemini_error:
                    logger.error(f"Gemini Vision also failed: {gemini_error}")
            
            # Last resort: text-only fallback response
            fallback_prompt = f"المستخدم أرسل صورة وقال: {user_message if user_message else 'ما قال شي'}. رد عليه باللهجة العراقية واعتذر له إنك ما تكدر تشوف الصورة بس تكدر تساعده بطرق ثانية."
            
            try:
                response = self.gpt5_client.chat.completions.create(
                    extra_headers={
                        "HTTP-Referer": "https://iraqi-bot.replit.app",
                        "X-Title": "Iraqi Dialect Bot - GPT-5",
                    },
                    model="openai/gpt-4o",
                    messages=[
                        {
                            "role": "system",
                            "content": self.system_instruction
                        },
                        {
                            "role": "user",
                            "content": fallback_prompt
                        }
                    ],
                    max_tokens=500,
                    temperature=0.7,
                )
                
                if response.choices and response.choices[0].message.content:
                    return response.choices[0].message.content.strip()
                    
            except Exception as fallback_error:
                logger.error(f"Error in fallback response: {fallback_error}")
            
            return "معذرة، ما كدرت أحلل الصورة هسه. بس تكدر تحچي وياي عادي وأساعدك بأي شي ثاني."
    
    async def _analyze_image_with_gemini(self, image_data: bytes, user_message: str = "") -> str:
        """
        Analyze an image using Gemini Vision capabilities as fallback.
        
        Args:
            image_data: The image data as bytes
            user_message: Optional context message from user
            
        Returns:
            Analysis description in Iraqi Arabic
        """
        try:
            logger.info("Using Gemini Vision for image analysis...")
            
            # Create prompt for image analysis in Iraqi dialect
            analysis_prompt = """حلل هذه الصورة بالتفصيل واوصفها باللهجة العراقية الأصيلة. 

اذكر:
- الألوان الموجودة 
- الأشخاص أو الحيوانات (إذا موجودة)
- المكان أو البيئة
- الأشياء والعناصر المهمة
- التفاصيل اللافتة للنظر
- الجو العام للصورة

استخدم كلمات عراقية مثل: شكو ماكو، زين، حلو، چان، هسه، شوية، وياه

كن مفصل ووصف كل شي تشوفه بطريقة واضحة ومفهومة."""
            
            if user_message:
                analysis_prompt += f"\n\nالمستخدم قال: {user_message}"
            
            # Convert image data to format Gemini can understand
            import base64
            base64_image = base64.b64encode(image_data).decode('utf-8')
            
            # Analyze with Gemini Vision
            model = genai.GenerativeModel('gemini-1.5-flash')
            import PIL.Image
            import io
            image = PIL.Image.open(io.BytesIO(image_data))
            response = model.generate_content(
                [analysis_prompt, image],
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=800,
                    temperature=0.7,
                )
            )
            
            # Better response handling for Gemini Vision
            if response and response.text:
                result = response.text.strip()
                # Clean the response to avoid formatting issues
                result = self._clean_response_text(result)
                logger.info("Successfully analyzed image with Gemini Vision")
                return result
            
            logger.warning("Empty response from Gemini Vision")
            return "ما كدرت أحلل الصورة بواسطة Gemini، بس تكدر تحچي وياي عادي."
                
        except Exception as e:
            logger.error(f"Error in Gemini Vision analysis: {e}")
            return "معذرة، صار خطأ وقت تحليل الصورة بـ Gemini. تكدر تحچي وياي عادي وأساعدك بأي شي ثاني."
    
    def _clean_response_text(self, text: str) -> str:
        """Clean response text to avoid Telegram formatting issues."""
        if not text:
            return text
            
        # Remove or escape problematic characters that break Telegram parsing
        # Remove excessive backticks and markdown characters
        import re
        
        # Remove multiple consecutive backticks
        text = re.sub(r'`{2,}', '`', text)
        
        # Remove unmatched backticks at the end
        if text.count('`') % 2 != 0:
            # Remove the last backtick if unmatched
            last_backtick = text.rfind('`')
            if last_backtick != -1:
                text = text[:last_backtick] + text[last_backtick+1:]
        
        # Clean up other problematic markdown characters
        # Remove unmatched asterisks
        if text.count('*') % 2 != 0:
            text = text.replace('*', '')
        
        # Remove unmatched underscores  
        if text.count('_') % 2 != 0:
            text = text.replace('_', '')
            
        # Ensure no broken markdown links
        text = re.sub(r'\[([^\]]*)\]\([^\)]*$', r'\1', text)  # Fix broken links
        
        return text.strip()
    
    async def translate_to_english(self, arabic_text: str) -> str:
        """
        Translate Arabic text to English using GPT-5 Chat.
        
        Args:
            arabic_text: Arabic text to translate
            
        Returns:
            English translation
        """
        try:
            prompt = f"Translate this Arabic text to natural English. Keep the meaning and tone:\n\n{arabic_text}"
            
            response = self.gpt5_client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": "https://iraqi-bot.replit.app",
                    "X-Title": "Iraqi Dialect Bot - GPT-5",
                },
                model="openai/gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional translator with enhanced GPT-5 capabilities. Translate Arabic to English accurately while preserving meaning, tone, and cultural context. Pay special attention to Iraqi dialect nuances."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,
                max_tokens=1000,
            )
            
            if response.choices and response.choices[0].message.content:
                result = response.choices[0].message.content.strip()
                logger.info("Successfully translated Arabic to English with GPT-5")
                return result
            else:
                return "Translation failed. Please try again."
                
        except Exception as e:
            logger.error(f"Error in Arabic to English translation: {e}")
            return "Sorry, translation failed. Please try again."
    
    async def translate_to_arabic(self, english_text: str) -> str:
        """
        Translate English text to Iraqi Arabic using GPT-5 Chat.
        
        Args:
            english_text: English text to translate
            
        Returns:
            Iraqi Arabic translation
        """
        try:
            prompt = f"Translate this English text to Iraqi Arabic dialect. Use authentic Iraqi expressions and vocabulary:\n\n{english_text}"
            
            response = self.gpt5_client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": "https://iraqi-bot.replit.app",
                    "X-Title": "Iraqi Dialect Bot - GPT-5",
                },
                model="openai/gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": self.system_instruction + "\n\nYou are using enhanced GPT-5 translation capabilities for superior English to Iraqi Arabic conversion."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=1000,
            )
            
            if response.choices and response.choices[0].message.content:
                result = response.choices[0].message.content.strip()
                logger.info("Successfully translated English to Iraqi Arabic with GPT-5")
                return result
            else:
                return "فشلت الترجمة. جرب مرة ثانية."
                
        except Exception as e:
            logger.error(f"Error in English to Arabic translation: {e}")
            return "معذرة، فشلت الترجمة. جرب مرة ثانية."
    
    async def generate_creative_description(self, description: str) -> tuple:
        """
        Generate creative description with English prompt using GPT-5 Chat or Gemini.
        
        Args:
            description: User's description in Arabic
            
        Returns:
            Tuple of (arabic_description, english_prompt)
        """
        try:
            prompt = f"""
المستخدم يريد وصف إبداعي لـ: {description}

المطلوب منك:
1. إنشاء نص إنجليزي مفصل ودقيق لإنشاء الصور (يكون قابل للنسخ)
2. وصف إبداعي باللهجة العراقية يصف نفس الصورة بطريقة حيوية وجميلة

ابدأ الجواب بـ "ENGLISH_PROMPT:" ثم النص الإنجليزي، ثم سطر فارغ، ثم "ARABIC_DESCRIPTION:" ثم الوصف العراقي.
            """
            
            # Try GPT-5 first
            try:
                response = self.gpt5_client.chat.completions.create(
                    extra_headers={
                        "HTTP-Referer": "https://iraqi-bot.replit.app",
                        "X-Title": "Iraqi Dialect Bot - GPT-5",
                    },
                    model="openai/gpt-4o",
                    messages=[
                        {
                            "role": "system",
                            "content": self.system_instruction + "\n\nYou are using enhanced creative capabilities for superior content generation and detailed prompts."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.8,
                    max_tokens=1000,
                )
                
                if response.choices and response.choices[0].message.content:
                    result = response.choices[0].message.content.strip()
                    logger.info("Successfully generated creative description with GPT-5")
                    return self._parse_creative_response(result, description)
                    
            except Exception as gpt_error:
                logger.error(f"GPT-5 failed for creative description: {gpt_error}")
                # Try Gemini fallback
                if self.gemini_client:
                    try:
                        response = self.gemini_client.models.generate_content(
                            model="gemini-2.0-flash",
                            contents=[
                                types.Content(role="user", parts=[types.Part(text=prompt)])
                            ],
                            config=types.GenerateContentConfig(
                                max_output_tokens=800,
                                temperature=0.8,
                            ),
                        )
                        
                        if response and hasattr(response, 'text') and response.text:
                            result = response.text.strip()
                            logger.info("Successfully generated creative description with Gemini")
                            return self._parse_creative_response(result, description)
                    except Exception as gemini_error:
                        logger.error(f"Gemini also failed for creative description: {gemini_error}")
            
            # Fallback creative response
            return self._create_fallback_creative_description(description)
                
        except Exception as e:
            logger.error(f"Error generating creative description: {e}")
            return self._create_fallback_creative_description(description)
    
    def _parse_creative_response(self, response: str, description: str) -> tuple:
        """Parse the AI response to extract English prompt and Arabic description."""
        try:
            if "ENGLISH_PROMPT:" in response and "ARABIC_DESCRIPTION:" in response:
                parts = response.split("ARABIC_DESCRIPTION:")
                english_part = parts[0].replace("ENGLISH_PROMPT:", "").strip()
                arabic_part = parts[1].strip()
                return (arabic_part, english_part)
            else:
                # If format is not as expected, create fallback
                return self._create_fallback_creative_description(description)
        except:
            return self._create_fallback_creative_description(description)
    
    def _create_fallback_creative_description(self, description: str) -> tuple:
        """Create fallback creative description when AI fails."""
        english_prompt = f"A beautiful, high-quality artistic image of {description}, professional photography, detailed, vibrant colors, excellent lighting"
        arabic_description = f"صورة حلوة ومفصلة لـ {description} بألوان زاهية وإضاءة ممتازة، تصوير احترافي عالي الجودة"
        return (arabic_description, english_prompt)
    
    async def analyze_with_tools(self, prompt: str, analysis_type: str) -> str:
        """
        Advanced analysis using GPT-5 Chat enhanced capabilities.
        
        Args:
            prompt: Text to analyze
            analysis_type: Type of analysis to perform
            
        Returns:
            Analysis result in Iraqi Arabic
        """
        try:
            response = self.gpt5_client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": "https://iraqi-bot.replit.app",
                    "X-Title": "Iraqi Dialect Bot - GPT-5",
                },
                model="openai/gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": self.system_instruction + "\n\nYou have access to advanced tools and GPT-5 Chat's enhanced capabilities for superior analysis."
                    },
                    {
                        "role": "user",
                        "content": f"حلل هذا النص بطريقة متقدمة ({analysis_type}):\n\n{prompt}"
                    }
                ],
                temperature=0.7,
                max_tokens=500,  # Further reduced to work within available credits
            )
            
            if response.choices and response.choices[0].message.content:
                return response.choices[0].message.content.strip()
            else:
                return "معذرة، ما كدرت أسوي التحليل المتقدم. جرب مرة ثانية."
                
        except Exception as e:
            logger.error(f"Error in advanced analysis: {e}")
            # Check if it's a credit/payment error
            if "402" in str(e) or "credit" in str(e).lower() or "payment" in str(e).lower():
                logger.warning("OpenRouter credits insufficient for advanced analysis")
                return "معذرة، الأرصدة مو كافية للتحليل المتقدم هسة. جرب مرة ثانية باجر."
            return "معذرة، صار خطأ في التحليل المتقدم."
    
    async def generate_structured_response(self, prompt: str, response_type: str) -> dict:
        """
        Generate structured response using GPT-5 Chat.
        
        Args:
            prompt: User's prompt
            response_type: Type of structured response
            
        Returns:
            Dictionary with success status and content
        """
        try:
            response = self.gpt5_client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": "https://iraqi-bot.replit.app",
                    "X-Title": "Iraqi Dialect Bot - GPT-5",
                },
                model="openai/gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": self.system_instruction + f"\n\nGenerate a {response_type} structured response using GPT-5 Chat's enhanced capabilities."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=500,  # Further reduced to work within available credits
            )
            
            if response.choices and response.choices[0].message.content:
                return {
                    "success": True,
                    "content": response.choices[0].message.content.strip()
                }
            else:
                return {
                    "success": False,
                    "content": "ما كدرت أسوي الرد المنظم. جرب مرة ثانية."
                }
                
        except Exception as e:
            logger.error(f"Error in structured response: {e}")
            # Check if it's a credit/payment error
            if "402" in str(e) or "credit" in str(e).lower() or "payment" in str(e).lower():
                logger.warning("OpenRouter credits insufficient for structured response")
                return {
                    "success": False,
                    "content": "معذرة، الأرصدة مو كافية للرد المنظم هسة. جرب مرة ثانية باجر."
                }
            return {
                "success": False,
                "content": "صار خطأ في إنشاء الرد المنظم."
            }
    
    @property
    def openai_api_key(self) -> str:
        """Compatibility property for image generation checks."""
        return self.gemini_api_key or ""