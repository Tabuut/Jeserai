import json
import os
import logging
import asyncio
from openai import OpenAI
import base64
from io import BytesIO
import requests

# Setup logging
logger = logging.getLogger(__name__)

class OpenAIClient:
    def __init__(self):
        """Initialize OpenAI client with API key."""
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        self.client = OpenAI(api_key=self.api_key)
        
        # System instruction for Iraqi dialect responses
        self.system_instruction = """أنت مساعد ذكي يتحدث باللهجة العراقية. مهامك:

1. رد على المستخدمين باللهجة العراقية الأصيلة والطبيعية
2. استخدم كلمات وتعابير عراقية مثل: شلونك، هسه، هواية، كلش، زين، شنو، وين، كيفك
3. كن مفيد ومساعد في كل الاستفسارات
4. إذا طلب المستخدم رسالة ترويجية أو إعلانية، اكتب له رسالة جذابة باللهجة العراقية
5. استخدم الرموز التعبيرية المناسبة لجعل المحادثة أكثر حيوية
6. كن صديق ومتفهم واجعل المحادثة ممتعة

تذكر: الهدف هو التحدث مثل العراقي الأصيل، فاستخدم اللهجة بطريقة طبيعية ومفهومة."""

    async def generate_arabic_response(self, user_message: str) -> str:
        """
        Generate an Arabic response to the user's message using OpenAI.
        
        Args:
            user_message: The user's input message
            
        Returns:
            Arabic response from OpenAI
        """
        try:
            logger.info(f"Sending request to OpenAI for message: {user_message[:50]}...")
            
            # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
            # do not change this unless explicitly requested by the user
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": self.system_instruction},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.7,
                max_tokens=800,
            )
            
            if response.choices and response.choices[0].message.content:
                result = response.choices[0].message.content.strip()
                logger.info("Successfully received response from OpenAI")
                return result
            else:
                logger.warning(f"Empty response from OpenAI for prompt: {user_message[:50]}...")
                return "هلا وغلا! شلونك؟ كلش أسف، ما كدرت افهم طلبك زين. ممكن توضحلي اكثر شوي؟"
                
        except Exception as e:
            logger.error(f"Error generating response from OpenAI: {e}")
            return "معذرة، صار خطأ وقت معالجة طلبك. جرب مرة ثانية باجر."
    
    async def generate_image(self, prompt: str, image_path: str) -> bool:
        """
        Generate an image using OpenAI DALL-E.
        
        Args:
            prompt: Description of the image to generate
            image_path: Path where to save the generated image
            
        Returns:
            True if image was generated successfully, False otherwise
        """
        try:
            logger.info(f"Generating image with OpenAI DALL-E for prompt: {prompt[:50]}...")
            
            response = self.client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                n=1,
                size="1024x1024",
                quality="standard",
            )
            
            if response.data and len(response.data) > 0:
                image_url = response.data[0].url
                
                # Download the image (with null check)
                if image_url:
                    image_response = requests.get(image_url)
                    if image_response.status_code == 200:
                        with open(image_path, 'wb') as f:
                            f.write(image_response.content)
                        logger.info(f"Image saved successfully to {image_path}")
                        return True
                    else:
                        logger.warning(f"Failed to download image from URL: {image_url}")
                        return False
                else:
                    logger.warning("No image URL provided")
                    return False
            else:
                logger.warning("No image data returned from OpenAI DALL-E")
                return False
                
        except Exception as e:
            logger.error(f"Error generating image with OpenAI: {e}")
            return False
    
    async def analyze_image(self, image_data: bytes, user_message: str = "") -> str:
        """
        Analyze an image using OpenAI GPT-4 Vision.
        
        Args:
            image_data: The image data as bytes
            user_message: Optional context message from user
            
        Returns:
            Analysis description in Arabic
        """
        try:
            logger.info("Analyzing image with OpenAI GPT-4 Vision...")
            
            # Convert image to base64
            base64_image = base64.b64encode(image_data).decode('utf-8')
            
            # Create prompt for image analysis
            analysis_prompt = "حلل هذه الصورة بالتفصيل واوصفها باللهجة العراقية. اذكر كل شي تشوفه فيها - الألوان، الأشخاص، المكان، الأشياء، والتفاصيل المهمة."
            
            if user_message:
                analysis_prompt += f"\n\nالمستخدم قال: {user_message}"
            
            # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
            # do not change this unless explicitly requested by the user
            response = self.client.chat.completions.create(
                model="gpt-4o",
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
                logger.info("Successfully analyzed image with OpenAI")
                return result
            else:
                logger.warning("Empty response from OpenAI for image analysis")
                return "معذرة، ما كدرت احلل الصورة هسه. جرب مرة ثانية."
                
        except Exception as e:
            logger.error(f"Error analyzing image with OpenAI: {e}")
            return "معذرة، صار خطأ وقت تحليل الصورة. جرب مرة ثانية."
    
    async def generate_creative_description(self, user_prompt: str) -> tuple[str, str]:
        """
        Generate creative description with both Arabic response and English prompt.
        
        Args:
            user_prompt: User's description request
            
        Returns:
            Tuple of (arabic_response, english_prompt)
        """
        try:
            logger.info(f"Generating creative description for: {user_prompt[:50]}...")
            
            # First, generate the English technical prompt
            english_prompt_request = f"""
Based on this user request: "{user_prompt}"

Create a detailed, technical English prompt suitable for AI image generation. Include:
- Visual style and artistic elements
- Colors, lighting, and composition
- Technical photography/art terms
- Detailed scene description

Make it comprehensive and specific for best AI image generation results.
Return ONLY the English prompt, nothing else.
"""
            
            # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
            # do not change this unless explicitly requested by the user
            english_response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "user", "content": english_prompt_request}
                ],
                temperature=0.8,
                max_tokens=300,
            )
            
            english_prompt = ""
            if english_response.choices and english_response.choices[0].message.content:
                english_prompt = english_response.choices[0].message.content.strip()
            
            # Then generate the Arabic creative description
            arabic_prompt_request = f"""
المستخدم طلب منك: "{user_prompt}"

اكتب وصف إبداعي جميل وخيالي باللهجة العراقية يصف الصورة أو المشهد الي طلبه المستخدم.
استخدم كلمات شاعرية وتعابير عراقية جميلة واجعل الوصف حيوي ومليء بالتفاصيل الرائعة.
استخدم الرموز التعبيرية المناسبة لجعل الوصف أكثر جمالاً.
"""
            
            arabic_response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": self.system_instruction},
                    {"role": "user", "content": arabic_prompt_request}
                ],
                temperature=0.8,
                max_tokens=500,
            )
            
            arabic_description = ""
            if arabic_response.choices and arabic_response.choices[0].message.content:
                arabic_description = arabic_response.choices[0].message.content.strip()
            else:
                arabic_description = "وصف إبداعي رائع للصورة الي طلبتها! 🎨✨"
            
            logger.info("Successfully generated creative description with OpenAI")
            return arabic_description, english_prompt
            
        except Exception as e:
            logger.error(f"Error generating creative description with OpenAI: {e}")
            fallback_arabic = "معذرة، ما كدرت اسوي الوصف الإبداعي هسه. جرب مرة ثانية! 🎨"
            fallback_english = f"Create an image of: {user_prompt}"
            return fallback_arabic, fallback_english
    
    async def translate_text(self, text: str, target_language: str) -> str:
        """
        Translate text using OpenAI.
        
        Args:
            text: Text to translate
            target_language: 'english' or 'arabic'
            
        Returns:
            Translated text
        """
        try:
            logger.info(f"Translating text to {target_language}: {text[:50]}...")
            
            if target_language.lower() == 'english':
                prompt = f"Translate this Arabic text to clear, natural English:\n\n{text}"
            else:
                prompt = f"Translate this English text to natural Iraqi Arabic dialect:\n\n{text}"
            
            # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
            # do not change this unless explicitly requested by the user
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500,
            )
            
            if response.choices and response.choices[0].message.content:
                result = response.choices[0].message.content.strip()
                logger.info(f"Successfully translated text to {target_language}")
                return result
            else:
                logger.warning(f"Empty response from OpenAI for translation to {target_language}")
                return "معذرة، ما كدرت اترجم النص هسه. جرب مرة ثانية."
                
        except Exception as e:
            logger.error(f"Error translating text with OpenAI: {e}")
            return "معذرة، صار خطأ بالترجمة. جرب مرة ثانية."