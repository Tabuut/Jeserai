import logging
import os
import base64
import requests
from openai import OpenAI

logger = logging.getLogger(__name__)

class Gemini25Client:
    """
    Gemini 2.5 Flash client for Iraqi dialect responses via OpenRouter.
    Provides high-performance AI responses with multimodal capabilities.
    """
    
    def __init__(self):
        """Initialize the Gemini 2.5 Flash client."""
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        
        # Initialize OpenAI client with OpenRouter base URL for Gemini 2.5 Flash
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key
        )
        
        # System instruction for Iraqi dialect responses
        self.system_instruction = """
أنت مساعد ذكي متخصص باللهجة العراقية الأصيلة. 

تعليمات مهمة:
- رد دائماً باللهجة العراقية الطبيعية والأصيلة
- استخدم كلمات عراقية أصيلة مثل: شلونك، زين، ماكو، وياك، شنو، وين، كلش، هوايه
- كن مفيد ومساعد في كل الأسئلة
- إذا ما تعرف الجواب، قل "معذرة، ما أعرف هالشي بس تكدر تسأل مرة ثانية"
- لا تستخدم اللهجة المصرية أو السعودية أو أي لهجة ثانية
- كن طبيعي وودود بردودك
- استخدم الأمثلة العراقية والثقافة العراقية وقت الحاجة

أمثلة على الرد الصحيح:
- "هلا وغلا، شلونك؟"
- "زين هالسؤال، راح أساعدك"
- "ماكو مشكلة، هاي المعلومات..."
- "كلش زين، هذا الشي اللي تريده"
"""
        
        logger.info("Gemini 2.5 Flash client initialized successfully")
    
    async def generate_arabic_response(self, user_message: str) -> str:
        """
        Generate Arabic response using Gemini 2.5 Flash model.
        
        Args:
            user_message: The user's message in Arabic or English
            
        Returns:
            Response in Iraqi Arabic dialect
        """
        try:
            logger.info(f"Sending request to Gemini 2.5 Flash for message: {user_message[:50]}...")
            
            response = self.client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": "https://iraqi-bot.replit.app",
                    "X-Title": "Iraqi Dialect Bot",
                },
                model="google/gemini-2.5-flash",
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
                max_tokens=1000,
            )
            
            if response.choices and response.choices[0].message.content:
                result = response.choices[0].message.content.strip()
                logger.info("Successfully received response from Gemini 2.5 Flash")
                return result
            else:
                logger.warning("Empty response from Gemini 2.5 Flash")
                return "معذرة، ما كدرت أجاوب هسه. جرب مرة ثانية."
                
        except Exception as e:
            logger.error(f"Error calling Gemini 2.5 Flash API: {e}")
            return "معذرة، صار خطأ وقت معالجة طلبك. جرب مرة ثانية باجر."
    
    async def generate_image(self, prompt: str, image_path: str) -> bool:
        """
        Generate an image using DALL-E 3 via OpenAI API.
        Note: Gemini doesn't support image generation, so we use OpenAI for this.
        
        Args:
            prompt: Description of the image to generate
            image_path: Path where to save the generated image
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Generating image with DALL-E 3 for prompt: {prompt[:50]}...")
            
            # Use OpenAI client for image generation
            openai_client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=self.api_key
            )
            
            # Generate image with DALL-E 3
            response = openai_client.images.generate(
                model="openai/dall-e-3",
                prompt=prompt,
                size="1024x1024",
                quality="standard",
                n=1,
            )
            
            if response.data and len(response.data) > 0:
                image_url = response.data[0].url
                
                if image_url:
                    # Download the image
                    img_response = requests.get(image_url, timeout=30)
                    img_response.raise_for_status()
                    
                    # Save the image
                    with open(image_path, 'wb') as f:
                        f.write(img_response.content)
                    
                    logger.info("Successfully generated and saved image with DALL-E 3")
                    return True
                else:
                    logger.error("No image URL in DALL-E 3 response")
                    return False
            else:
                logger.error("No image data in DALL-E 3 response")
                return False
                
        except Exception as e:
            logger.error(f"Error generating image with DALL-E 3: {e}")
            return False
    
    async def analyze_image(self, image_data: bytes, user_message: str = "") -> str:
        """
        Analyze an image using Gemini 2.5 Flash vision capabilities.
        
        Args:
            image_data: The image data as bytes
            user_message: Optional context message from user
            
        Returns:
            Analysis description in Iraqi Arabic
        """
        try:
            logger.info("Attempting image analysis with Gemini 2.5 Flash Vision...")
            
            # Convert image to base64
            base64_image = base64.b64encode(image_data).decode('utf-8')
            
            # Create prompt for image analysis
            analysis_prompt = "حلل هذه الصورة بالتفصيل واوصفها باللهجة العراقية. اذكر كل شي تشوفه فيها - الألوان، الأشخاص، المكان، الأشياء، والتفاصيل المهمة."
            
            if user_message:
                analysis_prompt += f"\n\nالمستخدم قال: {user_message}"
            
            # Analyze with Gemini 2.5 Flash Vision
            response = self.client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": "https://iraqi-bot.replit.app",
                    "X-Title": "Iraqi Dialect Bot",
                },
                model="google/gemini-2.5-flash",
                messages=[
                    {
                        "role": "system", 
                        "content": self.system_instruction
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
                temperature=0.7,
                max_tokens=1000,
            )
            
            if response.choices and response.choices[0].message.content:
                result = response.choices[0].message.content.strip()
                logger.info("Successfully analyzed image with Gemini 2.5 Flash Vision")
                return result
            else:
                logger.warning("Empty response from Gemini 2.5 Flash Vision")
                return "معذرة، ما كدرت احلل الصورة هسه. جرب مرة ثانية."
                
        except Exception as e:
            logger.error(f"Error analyzing image with Gemini 2.5 Flash Vision: {e}")
            # Fallback to text-only analysis
            fallback_prompt = f"المستخدم أرسل صورة وقال: {user_message if user_message else 'ما قال شي'}. رد عليه باللهجة العراقية واعتذر له إنك ما تكدر تشوف الصورة بس تكدر تساعده بطرق ثانية."
            
            try:
                response = self.client.chat.completions.create(
                    extra_headers={
                        "HTTP-Referer": "https://iraqi-bot.replit.app",
                        "X-Title": "Iraqi Dialect Bot",
                    },
                    model="google/gemini-2.5-flash",
                    messages=[
                        {"role": "system", "content": self.system_instruction},
                        {"role": "user", "content": fallback_prompt}
                    ],
                    temperature=0.7,
                    max_tokens=500,
                )
                
                if response.choices and response.choices[0].message.content:
                    return response.choices[0].message.content.strip()
                    
            except Exception as fallback_error:
                logger.error(f"Fallback analysis also failed: {fallback_error}")
            
            return "معذرة، ما كدرت احلل الصورة هسه. جرب ترسل الصورة مرة ثانية."
    
    async def translate_to_english(self, arabic_text: str) -> str:
        """
        Translate Arabic text to English using Gemini 2.5 Flash.
        
        Args:
            arabic_text: The Arabic text to translate
            
        Returns:
            English translation
        """
        try:
            prompt = f"Translate this Arabic text to natural English. Keep the meaning and tone:\n\n{arabic_text}"
            
            response = self.client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": "https://iraqi-bot.replit.app",
                    "X-Title": "Iraqi Dialect Bot",
                },
                model="google/gemini-2.5-flash",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional translator. Translate Arabic to English accurately while preserving meaning and tone."
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
                return response.choices[0].message.content.strip()
            else:
                return "Translation failed. Please try again."
                
        except Exception as e:
            logger.error(f"Error translating to English: {e}")
            return "معذرة، ما كدرت أترجم النص. جرب مرة ثانية."
    
    async def translate_to_arabic(self, english_text: str) -> str:
        """
        Translate English text to Iraqi Arabic using Gemini 2.5 Flash.
        
        Args:
            english_text: The English text to translate
            
        Returns:
            Iraqi Arabic translation
        """
        try:
            prompt = f"Translate this English text to Iraqi Arabic dialect. Use authentic Iraqi expressions and vocabulary:\n\n{english_text}"
            
            response = self.client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": "https://iraqi-bot.replit.app",
                    "X-Title": "Iraqi Dialect Bot",
                },
                model="google/gemini-2.5-flash",
                messages=[
                    {
                        "role": "system",
                        "content": self.system_instruction
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.5,
                max_tokens=1000,
            )
            
            if response.choices and response.choices[0].message.content:
                return response.choices[0].message.content.strip()
            else:
                return "الترجمة ما اشتغلت. جرب مرة ثانية."
                
        except Exception as e:
            logger.error(f"Error translating to Arabic: {e}")
            return "معذرة، ما كدرت أترجم النص. جرب مرة ثانية."
    
    async def generate_creative_description(self, user_prompt: str) -> tuple[str, str]:
        """
        Generate creative description with both Arabic description and English prompt.
        
        Args:
            user_prompt: The user's creative prompt request
            
        Returns:
            Tuple of (arabic_description, english_prompt)
        """
        try:
            creative_prompt = f"""
المستخدم يريد إنشاء صورة وطلب: "{user_prompt}"

اكتب جوابين منفصلين:

1. أولاً، اكتب نص إنجليزي مفصل (prompt) لإنشاء هذه الصورة بالذكاء الاصطناعي. النص يجب أن يشمل:
- وصف تفصيلي للمشهد
- التفاصيل الفنية والبصرية 
- الألوان والإضاءة
- جودة عالية ومواصفات تقنية
- مصطلحات فنية مثل: ultra-detailed, cinematic, photorealistic, 8K resolution

2. ثانياً، اكتب وصف إبداعي قصير باللهجة العراقية يصف الصورة بطريقة جميلة ومثيرة للخيال.

ابدأ الجواب بـ "ENGLISH_PROMPT:" ثم النص الإنجليزي، ثم سطر فارغ، ثم "ARABIC_DESCRIPTION:" ثم الوصف العراقي.
            """
            
            response = self.client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": "https://iraqi-bot.replit.app",
                    "X-Title": "Iraqi Dialect Bot",
                },
                model="google/gemini-2.5-flash",
                messages=[
                    {
                        "role": "system",
                        "content": self.system_instruction
                    },
                    {
                        "role": "user",
                        "content": creative_prompt
                    }
                ],
                temperature=0.8,
                max_tokens=1500,
            )
            
            if response.choices and response.choices[0].message.content:
                result = response.choices[0].message.content.strip()
                
                # Parse the response to extract English prompt and Arabic description
                lines = result.split('\n')
                english_prompt = ""
                arabic_description = ""
                current_section = ""
                
                for line in lines:
                    line = line.strip()
                    if line.startswith("ENGLISH_PROMPT:"):
                        current_section = "english"
                        english_prompt += line.replace("ENGLISH_PROMPT:", "").strip()
                    elif line.startswith("ARABIC_DESCRIPTION:"):
                        current_section = "arabic"
                        arabic_description += line.replace("ARABIC_DESCRIPTION:", "").strip()
                    elif current_section == "english" and line:
                        english_prompt += " " + line
                    elif current_section == "arabic" and line:
                        arabic_description += " " + line
                
                # Fallback if parsing fails
                if not english_prompt or not arabic_description:
                    parts = result.split("ARABIC_DESCRIPTION:")
                    if len(parts) == 2:
                        english_part = parts[0].replace("ENGLISH_PROMPT:", "").strip()
                        arabic_part = parts[1].strip()
                        english_prompt = english_part
                        arabic_description = arabic_part
                    else:
                        # Last resort - use the whole response as English prompt
                        english_prompt = result
                        arabic_description = "وصف إبداعي جميل للصورة المطلوبة"
                
                logger.info("Successfully generated creative description with Gemini 2.5 Flash")
                return arabic_description.strip(), english_prompt.strip()
            else:
                logger.warning("Empty response from Gemini 2.5 Flash for creative description")
                return "وصف إبداعي للصورة", "A detailed artistic description"
                
        except Exception as e:
            logger.error(f"Error generating creative description: {e}")
            return "معذرة، ما كدرت أسوي الوصف الإبداعي", "Creative description generation failed"