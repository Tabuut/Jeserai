"""
Gemini AI Client for Arabic response generation.
Handles all interactions with the Gemini API.
"""

import json
import logging
import os
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

class GeminiClient:
    """Client for interacting with Gemini AI API."""
    
    def __init__(self):
        """Initialize the Gemini client."""
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        
        self.client = genai.Client(api_key=api_key)
        self.model = "gemini-2.5-flash"
        
        # Enhanced system instruction for Iraqi dialect responses
        self.system_instruction = (
            "أنت مساعد ذكي يتحدث باللهجة العراقية بطريقة متقنة وطبيعية وحيوية. "
            "يجب أن تجيب على جميع الأسئلة والرسائل باللهجة العراقية فقط بطريقة ودودة ومفيدة. "
            "استخدم كلمات مثل 'شلونك، شكو ماكو، وين، هاي، هاذا، هاذي، دا، هسه، ويا، اكو، ماكو، شوف، ترا، يلا، تعال، خوش، زين، تدلل، نورت، أهلين، ما شاء الله' وغيرها من المفردات العراقية الأصيلة. "
            "إذا طلب منك المستخدم كتابة رسالة ترويجية أو إعلانية للبوت، اكتب رسالة جذابة ومقنعة باللهجة العراقية تسوق لميزات البوت. "
            "إذا سأل عن ميزات البوت، اذكر: المحادثة الذكية، إنشاء الصور، تحليل الصور، الترجمة، الوصف الإبداعي. "
            "هذا البوت تم تصميمه وبرمجته بالكامل بواسطة المطور الموهوب ثابت (@tht_txt). "
            "إذا سأل أحد عن حساب المطور، اعطه: @tht_txt "
            "كن متفاعل وحماسي في ردودك، واستخدم الرموز التعبيرية المناسبة. "
            "إذا لم تفهم السؤال، اطلب التوضيح بطريقة لطيفة باللهجة العراقية."
        )
    
    async def generate_arabic_response(self, user_message: str, use_system_instruction: bool = True) -> str:
        """
        Generate an Arabic response to the user's message using Gemini AI.
        
        Args:
            user_message: The user's input message
            
        Returns:
            Arabic response from Gemini AI
        """
        try:
            logger.info(f"Sending request to Gemini for message: {user_message[:50]}...")
            
            # Use a more direct approach with timeout
            config = types.GenerateContentConfig(
                temperature=0.7,
                max_output_tokens=800,
                candidate_count=1,
            )
            
            if use_system_instruction:
                config.system_instruction = self.system_instruction
            
            response = self.client.models.generate_content(
                model=self.model,
                contents=[types.Part(text=user_message)],
                config=config
            )
            
            if response.text and response.text.strip():
                logger.info("Successfully received response from Gemini")
                return response.text.strip()
            else:
                logger.warning(f"Empty response from Gemini for prompt: {user_message[:50]}...")
                # Try with minimal configuration
                try:
                    simple_response = self.client.models.generate_content(
                        model=self.model,
                        contents=user_message,
                        config=types.GenerateContentConfig(
                            temperature=0.5,
                            max_output_tokens=500,
                        )
                    )
                    
                    if simple_response.text and simple_response.text.strip():
                        return simple_response.text.strip()
                except:
                    pass
                
                return "هلا وغلا! شلونك؟ كلش أسف، ما كدرت افهم طلبك زين. ممكن توضحلي اكثر شوي؟"
                
        except Exception as e:
            logger.error(f"Error generating response from Gemini: {e}")
            return "معذرة، صار خطأ وقت معالجة طلبك. جرب مرة ثانية باجر."
    
    async def generate_image(self, prompt: str, image_path: str) -> bool:
        """
        Generate an image using Gemini AI.
        
        Args:
            prompt: Description of the image to generate
            image_path: Path where to save the generated image
            
        Returns:
            True if image was generated successfully, False otherwise
        """
        try:
            logger.info(f"Generating image for prompt: {prompt[:50]}...")
            
            response = self.client.models.generate_content(
                model="gemini-2.0-flash-preview-image-generation",
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_modalities=['TEXT', 'IMAGE']
                )
            )

            if not response.candidates:
                logger.warning("No image candidates returned from Gemini")
                return False

            content = response.candidates[0].content
            if not content or not content.parts:
                logger.warning("No content parts in response")
                return False

            for part in content.parts:
                if part.inline_data and part.inline_data.data:
                    with open(image_path, 'wb') as f:
                        f.write(part.inline_data.data)
                    logger.info(f"Image saved successfully to {image_path}")
                    return True

            logger.warning("No image data found in response")
            return False
            
        except Exception as e:
            logger.error(f"Error generating image: {e}")
            return False
    
    async def check_connection(self) -> bool:
        """
        Check if the Gemini API connection is working.
        
        Returns:
            True if connection is successful, False otherwise
        """
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents="مرحبا"
            )
            return response.text is not None
        except Exception as e:
            logger.error(f"Gemini connection check failed: {e}")
            return False
    
    async def analyze_image(self, image_path: str, user_prompt: str = None) -> str:
        """
        Analyze an image using Gemini Vision and respond in Iraqi dialect.
        
        Args:
            image_path: Path to the image file
            user_prompt: Optional user prompt about the image
            
        Returns:
            Arabic description of the image
        """
        try:
            with open(image_path, "rb") as f:
                image_bytes = f.read()
            
            # Create prompt for image analysis in Iraqi dialect
            if user_prompt:
                prompt = f"{user_prompt}\n\nاوصف الصورة باللهجة العراقية بتفصيل."
            else:
                prompt = "اوصف هاي الصورة باللهجة العراقية بتفصيل. شنو شايف فيها؟ وين مكانها؟ شلون تبدو؟"
            
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[
                    types.Part.from_bytes(
                        data=image_bytes,
                        mime_type="image/jpeg",
                    ),
                    prompt
                ],
                config=types.GenerateContentConfig(
                    system_instruction=self.system_instruction,
                    temperature=0.7,
                    max_output_tokens=800,
                )
            )
            
            if response.text and response.text.strip():
                logger.info("Successfully analyzed image with Gemini")
                return response.text.strip()
            else:
                logger.warning("Empty response from Gemini for image analysis")
                return "ما كدرت اشوف الصورة زين، جرب ترسل صورة واضحة اكثر."
        
        except Exception as e:
            logger.error(f"Error in image analysis: {e}")
            return "صار خطأ وقت تحليل الصورة، جرب مرة ثانية."
    
    async def translate_text(self, text: str, target_language: str = "english") -> str:
        """
        Translate text to the specified language.
        
        Args:
            text: Text to translate
            target_language: Target language (english, arabic, etc.)
            
        Returns:
            Translated text
        """
        try:
            if target_language.lower() in ["english", "انجليزي", "انكليزي"]:
                prompt = f"ترجم هذا النص إلى الإنجليزية، اكتب الترجمة فقط بدون أي تعليق:\n\n{text}"
                system_prompt = "أنت مترجم محترف. اكتب الترجمة الإنجليزية فقط بدون أي كلام إضافي."
            else:
                prompt = f"ترجم هذا النص إلى العربية (اللهجة العراقية إذا أمكن)، اكتب الترجمة فقط بدون أي تعليق:\n\n{text}"
                system_prompt = "أنت مترجم محترف. اكتب الترجمة العربية فقط بدون أي كلام إضافي، ويفضل باللهجة العراقية إذا كان النص مناسب لذلك."
            
            response = self.client.models.generate_content(
                model=self.model,
                contents=[types.Part(text=prompt)],
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.3,
                    max_output_tokens=1000,
                )
            )
            
            if response.text and response.text.strip():
                logger.info(f"Successfully translated text to {target_language}")
                return response.text.strip()
            else:
                logger.warning("Empty response from Gemini for translation")
                return "ما كدرت اترجم النص، جرب مرة ثانية."
        
        except Exception as e:
            logger.error(f"Error in translation: {e}")
            return "صار خطأ وقت الترجمة، جرب مرة ثانية."
