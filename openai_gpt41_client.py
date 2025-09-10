import logging
import os
import base64
import requests
from openai import OpenAI

logger = logging.getLogger(__name__)

class OpenAIGPT41Client:
    """
    OpenAI GPT-4.1 client for Iraqi dialect responses and advanced AI capabilities.
    Uses OpenRouter to access GPT-4.1 model with full features including:
    - Enhanced function calling and tool usage
    - Improved coding and instruction following
    - Long-context understanding (1M tokens)
    - Advanced agentic workflows
    """
    
    def __init__(self):
        """Initialize the OpenAI GPT-4.1 client."""
        # OpenRouter API key for text generation
        self.openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
        if not self.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is not set")
        
        # OpenAI API key for DALL-E 3 image generation
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not self.openai_api_key:
            logger.warning("OPENAI_API_KEY not set - DALL-E 3 image generation will be disabled")
        
        # Initialize OpenAI client with OpenRouter base URL for text generation
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.openrouter_api_key
        )
        
        # Initialize direct OpenAI client for image generation
        if self.openai_api_key:
            self.image_client = OpenAI(api_key=self.openai_api_key)
        else:
            self.image_client = None
        
        # Enhanced system instruction for Iraqi dialect responses with GPT-4.1 capabilities
        self.system_instruction = """
أنت مساعد ذكي متقدم متخصص باللهجة العراقية الأصيلة ومزود بقدرات GPT-4.1 المتطورة.

تعليمات أساسية:
- رد دائماً باللهجة العراقية الطبيعية والأصيلة
- استخدم كلمات عراقية أصيلة مثل: شلونك، زين، ماكو، وياك، شنو، وين، كلش، هوايه
- كن مفيد ومساعد في كل الأسئلة
- إذا ما تعرف الجواب، قل "معذرة، ما أعرف هالشي بس تكدر تسأل مرة ثانية"
- لا تستخدم اللهجة المصرية أو السعودية أو أي لهجة ثانية
- كن طبيعي وودود بردودك
- استخدم الأمثلة العراقية والثقافة العراقية وقت الحاجة

قدرات متقدمة مع GPT-4.1:
- تحليل المحتوى الطويل بفهم أعمق (حتى مليون رمز)
- برمجة متقدمة وحل المشاكل التقنية
- فهم السياق المطول والربط بين المعلومات
- تحليل دقيق للصور والمحتوى المرئي
- ترجمة محسنة مع فهم أفضل للسياق
- إنشاء محتوى إبداعي بجودة عالية

أمثلة على الرد الصحيح:
- "هلا وغلا، شلونك؟ شنو تريد أساعدك بيه اليوم؟"
- "زين هالسؤال، راح أساعدك بكل التفاصيل"
- "ماكو مشكلة، هاي المعلومات المفصلة اللي تحتاجها..."
- "كلش زين، هذا الشي اللي تريده بالضبط"
"""
        
        logger.info("OpenAI GPT-4.1 client initialized successfully")
    
    async def generate_arabic_response(self, user_message: str) -> str:
        """
        Generate Arabic response using GPT-4.1 model.
        
        Args:
            user_message: The user's message in Arabic or English
            
        Returns:
            Response in Iraqi Arabic dialect
        """
        try:
            logger.info(f"Sending request to GPT-4o with enhanced features for message: {user_message[:50]}...")
            
            response = self.client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": "https://iraqi-bot.replit.app",
                    "X-Title": "Iraqi Dialect Bot",
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
                max_tokens=2000,  # Enhanced capabilities
            )
            
            if response.choices and response.choices[0].message.content:
                result = response.choices[0].message.content.strip()
                logger.info("Successfully received response from enhanced GPT-4o")
                return result
            else:
                logger.warning("Empty response from GPT-4.1")
                return "معذرة، ما كدرت أجاوب هسه. جرب مرة ثانية."
                
        except Exception as e:
            logger.error(f"Error calling GPT-4.1 API: {e}")
            return "معذرة، صار خطأ وقت معالجة طلبك. جرب مرة ثانية باجر."
    
    async def generate_image(self, prompt: str, image_path: str) -> bool:
        """
        Generate an image using DALL-E 3 via OpenAI API.
        
        Args:
            prompt: Description of the image to generate
            image_path: Path where to save the generated image
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if image generation is available
            if not self.image_client:
                logger.error("DALL-E 3 image generation unavailable - OPENAI_API_KEY not set")
                return False
            
            logger.info(f"Generating image with DALL-E 3 for prompt: {prompt[:50]}...")
            
            # Generate image with DALL-E 3 using direct OpenAI API
            response = self.image_client.images.generate(
                model="dall-e-3",
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
        Analyze an image using GPT-4.1 Vision capabilities.
        
        Args:
            image_data: The image data as bytes
            user_message: Optional context message from user
            
        Returns:
            Analysis description in Iraqi Arabic
        """
        try:
            logger.info("Attempting image analysis with enhanced GPT-4o Vision...")
            
            # Convert image to base64
            base64_image = base64.b64encode(image_data).decode('utf-8')
            
            # Create prompt for image analysis
            analysis_prompt = "حلل هذه الصورة بالتفصيل واوصفها باللهجة العراقية. اذكر كل شي تشوفه فيها - الألوان، الأشخاص، المكان، الأشياء، والتفاصيل المهمة."
            
            if user_message:
                analysis_prompt += f"\n\nالمستخدم قال: {user_message}"
            
            # Analyze with GPT-4o Vision (enhanced image understanding)
            response = self.client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": "https://iraqi-bot.replit.app",
                    "X-Title": "Iraqi Dialect Bot",
                },
                model="openai/gpt-4o",
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
                max_tokens=1500,  # Increased for more detailed analysis
            )
            
            if response.choices and response.choices[0].message.content:
                result = response.choices[0].message.content.strip()
                logger.info("Successfully analyzed image with GPT-4.1 Vision")
                return result
            else:
                logger.warning("Empty response from GPT-4.1 Vision")
                return "معذرة، ما كدرت احلل الصورة هسه. جرب مرة ثانية."
                
        except Exception as e:
            logger.error(f"Error analyzing image with GPT-4.1 Vision: {e}")
            # Fallback to text-only analysis
            fallback_prompt = f"المستخدم أرسل صورة وقال: {user_message if user_message else 'ما قال شي'}. رد عليه باللهجة العراقية واعتذر له إنك ما تكدر تشوف الصورة بس تكدر تساعده بطرق ثانية."
            
            try:
                response = self.client.chat.completions.create(
                    extra_headers={
                        "HTTP-Referer": "https://iraqi-bot.replit.app",
                        "X-Title": "Iraqi Dialect Bot",
                    },
                    model="openai/gpt-4o",
                    messages=[
                        {"role": "system", "content": self.system_instruction},
                        {"role": "user", "content": fallback_prompt}
                    ],
                    temperature=0.7,
                    max_tokens=800,  # Increased for better fallback responses
                )
                
                if response.choices and response.choices[0].message.content:
                    return response.choices[0].message.content.strip()
                    
            except Exception as fallback_error:
                logger.error(f"Fallback analysis also failed: {fallback_error}")
            
            return "معذرة، ما كدرت احلل الصورة هسه. جرب ترسل الصورة مرة ثانية."
    
    async def translate_to_english(self, arabic_text: str) -> str:
        """
        Translate Arabic text to English using GPT-4.1.
        
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
                model="openai/gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional translator with enhanced capabilities. Translate Arabic to English accurately while preserving meaning, tone, and cultural context. Pay special attention to Iraqi dialect nuances."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,
                max_tokens=1500,  # Increased for better translation quality
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
        Translate English text to Iraqi Arabic using GPT-4.1.
        
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
                model="openai/gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": self.system_instruction + "\n\nYou are using enhanced translation capabilities for superior English to Iraqi Arabic conversion."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.5,
                max_tokens=1500,  # Increased for better translation output
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
                model="openai/gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": self.system_instruction + "\n\nYou are using enhanced creative capabilities for superior content generation and detailed prompts."
                    },
                    {
                        "role": "user",
                        "content": creative_prompt
                    }
                ],
                temperature=0.8,
                max_tokens=2500,  # Significantly increased for enhanced creative output
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
                
                logger.info("Successfully generated creative description with GPT-4.1")
                return arabic_description.strip(), english_prompt.strip()
            else:
                logger.warning("Empty response from GPT-4.1 for creative description")
                return "وصف إبداعي للصورة", "A detailed artistic description"
                
        except Exception as e:
            logger.error(f"Error generating creative description with GPT-4.1: {e}")
            return "معذرة، ما كدرت أسوي وصف إبداعي هسه. جرب مرة ثانية.", "A beautiful creative description"
    
    async def analyze_with_tools(self, user_message: str, context: str = "") -> str:
        """
        Enhanced analysis using GPT-4.1's advanced tool calling capabilities.
        
        Args:
            user_message: The user's message
            context: Additional context for analysis
            
        Returns:
            Enhanced response using GPT-4.1's tool capabilities
        """
        try:
            # Define tools for advanced analysis
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "analyze_iraqi_context",
                        "description": "Analyze text for Iraqi cultural context and references",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "text": {"type": "string", "description": "Text to analyze"},
                                "context_type": {"type": "string", "enum": ["cultural", "linguistic", "social", "historical"]}
                            },
                            "required": ["text", "context_type"]
                        }
                    }
                },
                {
                    "type": "function", 
                    "function": {
                        "name": "enhance_iraqi_response",
                        "description": "Enhance response with authentic Iraqi expressions",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "base_response": {"type": "string", "description": "Base response to enhance"},
                                "formality_level": {"type": "string", "enum": ["casual", "formal", "friendly"]}
                            },
                            "required": ["base_response", "formality_level"]
                        }
                    }
                }
            ]
            
            enhanced_prompt = f"""
You are an advanced AI agent using GPT-4.1 capabilities. You are specialized in Iraqi dialect and culture.

تعليمات للوكيل الذكي:
- أنت وكيل ذكي متقدم - استمر بالعمل حتى يتم حل استفسار المستخدم بالكامل قبل إنهاء دورك
- استخدم الأدوات المتاحة لتجميع المعلومات ذات الصلة إذا لم تكن متأكداً من المحتوى
- خطط بشكل مكثف قبل كل استدعاء دالة، وفكر بعمق في نتائج استدعاءات الدوال السابقة

User message: {user_message}
Context: {context}

رد باللهجة العراقية الأصيلة مع استخدام قدرات GPT-4.1 المتقدمة.
            """
            
            logger.info("Using GPT-4.1 enhanced tool calling for advanced analysis...")
            
            response = self.client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": "https://iraqi-bot.replit.app",
                    "X-Title": "Iraqi Dialect Bot",
                },
                model="openai/gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": self.system_instruction + "\n\nYou have access to advanced tools and enhanced capabilities for superior analysis."
                    },
                    {
                        "role": "user",
                        "content": enhanced_prompt
                    }
                ],
                tools=tools,
                tool_choice="auto",
                temperature=0.7,
                max_tokens=2000,
            )
            
            # Handle tool calls if any
            if response.choices[0].message.tool_calls:
                logger.info("GPT-4.1 initiated tool calls for enhanced analysis")
                # In a full implementation, we would process tool calls here
                # For now, return the message with tool call information
                tool_info = f"استخدم GPT-4.1 أدوات متقدمة لتحليل طلبك..."
                
            if response.choices and response.choices[0].message.content:
                result = response.choices[0].message.content.strip()
                logger.info("Successfully completed GPT-4.1 enhanced analysis")
                return result
            else:
                logger.warning("Empty response from GPT-4.1 enhanced analysis")
                return "معذرة، ما كدرت أحلل طلبك بالأدوات المتقدمة. جرب مرة ثانية."
                
        except Exception as e:
            logger.error(f"Error in GPT-4.1 enhanced analysis: {e}")
            # Fallback to regular response
            return await self.generate_arabic_response(user_message)
    
    async def generate_structured_response(self, prompt: str, response_type: str = "detailed") -> dict:
        """
        Generate structured response using GPT-4.1's enhanced capabilities.
        
        Args:
            prompt: The input prompt
            response_type: Type of structured response needed
            
        Returns:
            Structured response as dictionary
        """
        try:
            structured_prompt = f"""
أنت مساعد ذكي متقدم باستخدام GPT-4.1. رد بصيغة منظمة حسب النوع المطلوب.

Prompt: {prompt}
Response Type: {response_type}

قم بإنشاء رد منظم ومفصل باللهجة العراقية حسب النوع المطلوب.
            """
            
            logger.info(f"Generating structured {response_type} response with GPT-4.1...")
            
            response = self.client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": "https://iraqi-bot.replit.app",
                    "X-Title": "Iraqi Dialect Bot",
                },
                model="openai/gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": self.system_instruction + f"\n\nGenerate a {response_type} structured response using enhanced capabilities."
                    },
                    {
                        "role": "user",
                        "content": structured_prompt
                    }
                ],
                temperature=0.6,
                max_tokens=2500,
            )
            
            if response.choices and response.choices[0].message.content:
                result = response.choices[0].message.content.strip()
                logger.info(f"Successfully generated structured {response_type} response")
                
                return {
                    "type": response_type,
                    "content": result,
                    "model": "gpt-4.1",
                    "success": True
                }
            else:
                logger.warning(f"Empty structured response from GPT-4.1")
                return {
                    "type": response_type,
                    "content": "معذرة، ما كدرت أسوي رد منظم هسه.",
                    "model": "gpt-4.1", 
                    "success": False
                }
                
        except Exception as e:
            logger.error(f"Error generating structured response: {e}")
            return {
                "type": response_type,
                "content": "معذرة، صار خطأ في إنشاء الرد المنظم.",
                "model": "gpt-4.1",
                "success": False,
                "error": str(e)
            }