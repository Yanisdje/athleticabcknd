from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os
import requests
from requests.exceptions import Timeout
import base64
import logging
import json
import re

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configure CORS
CORS(app)

# Get API key from environment variable
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

@app.route('/')
def home():
    return jsonify({"status": "ok", "message": "Server is running"})

@app.route('/analyze', methods=['POST', 'OPTIONS'])
def analyze_image():
    # Handle preflight requests
    if request.method == 'OPTIONS':
        response = app.make_default_options_response()
        response.headers['Access-Control-Allow-Methods'] = 'POST'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        return response

    try:
        data = request.json
        if not data:
            logger.error("No data received")
            return jsonify({'error': 'No data received'}), 400
            
        base64_image = data.get('image')
        form_data = data.get('formData')
        
        if not base64_image and not form_data:
            logger.error("No image or form data provided")
            return jsonify({'error': 'Either image or form data must be provided'}), 400

        logger.info(f"Received data - Image: {'Yes' if base64_image else 'No'}, Form data: {'Yes' if form_data else 'No'}")
        
        # Validate and clean base64 string if provided
        if base64_image:
            try:
                # Remove any potential data URL prefix
                if 'base64,' in base64_image:
                    base64_image = base64_image.split('base64,')[1]
                
                # Add padding if necessary
                padding = 4 - (len(base64_image) % 4)
                if padding != 4:
                    base64_image += '=' * padding
                    
                # Test if it's valid base64
                base64.b64decode(base64_image)
                logger.info(f"Valid base64 image, length: {len(base64_image)}")
            except Exception as e:
                logger.error(f"Invalid base64 image: {str(e)}")
                return jsonify({'error': 'Invalid image format'}), 400

        logger.info("Making OpenAI request...")
        
        # Build the prompt text
        prompt_text = "Analyze the provided fitness data"
        if base64_image:
            prompt_text += " and image"
        prompt_text += " and provide a comprehensive assessment in the following JSON format.\n\n"
        
        if form_data:
            prompt_text += "User Profile Information:\n"
            prompt_text += f"- Gender: {form_data.get('gender', 'Not specified')}\n"
            prompt_text += f"- Age: {form_data.get('age', 'Not specified')}\n"
            prompt_text += f"- Height: {form_data.get('height', 'Not specified')} cm\n"
            prompt_text += f"- Weight: {form_data.get('weight', 'Not specified')} kg\n"
            prompt_text += f"- BMI: {form_data.get('bmi', 'Not calculated')}\n"
            
            if form_data.get('goals'):
                goals = ', '.join([goal.get('title', '') for goal in form_data.get('goals', [])])
                prompt_text += f"- Fitness Goals: {goals}\n"
            else:
                prompt_text += "- Fitness Goals: Not specified\n"
                
            if form_data.get('mainFocus'):
                focus_areas = ', '.join([focus.get('title', '') for focus in form_data.get('mainFocus', [])])
                prompt_text += f"- Main Focus Areas: {focus_areas}\n"
            else:
                prompt_text += "- Main Focus Areas: Not specified\n"
                
            training_level = form_data.get('trainingLevel', {}).get('title', 'Not specified') if form_data.get('trainingLevel') else 'Not specified'
            prompt_text += f"- Training Level: {training_level}\n"
            
            workout_location = form_data.get('workoutLocation', {}).get('title', 'Not specified') if form_data.get('workoutLocation') else 'Not specified'
            prompt_text += f"- Workout Location: {workout_location}\n\n"
        
        if not base64_image:
            prompt_text += "NOTE: No body image was provided, so DO NOT include any body composition analysis, body fat percentage, muscle definition, posture analysis, or visual body assessments. Focus only on creating a workout plan and recommendations based on the provided profile information.\n\n"
        
        prompt_text += "Please provide assessment in the following JSON format:"
        
        url = "https://api.openai.com/v1/chat/completions"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}"
        }

        # Create the JSON structure template based on whether image is provided
        if base64_image:
            json_structure = '''
{
  "fitness_score": [number 1-100],
  "has_body_composition": true,
  "body_composition": {
    "muscle_definition": "[detailed analysis of visible muscle definition]",
    "body_fat_percentage": "[estimated percentage or range]",
    "posture_analysis": "[assessment of posture and alignment]",
    "symmetry": "[evaluation of muscle balance and symmetry]"
  },
  "areas_for_improvement": [
    "[specific improvement areas based on image and profile]"
  ],
  "workout_plan": {
    "day_1": {
      "name": "[workout name]",
      "exercises": [
        {"name": "[exercise name]", "sets": [number], "reps": "[rep range]", "rest": "[rest time]"}
      ]
    }
  },
  "nutrition_recommendations": [
    "[personalized nutrition advice]"
  ],
  "recovery_recommendations": [
    "[recovery and rest advice]"
  ],
  "progress_tracking": [
    "[methods to track progress]"
  ]
}'''
        else:
            json_structure = '''
{
  "fitness_score": [number 1-100],
  "has_body_composition": false,
  "areas_for_improvement": [
    "[improvement areas based on profile data only]"
  ],
  "workout_plan": {
    "day_1": {
      "name": "[workout name]",
      "exercises": [
        {"name": "[exercise name]", "sets": [number], "reps": "[rep range]", "rest": "[rest time]"}
      ]
    }
  },
  "nutrition_recommendations": [
    "[general nutrition advice based on goals and profile]"
  ],
  "recovery_recommendations": [
    "[recovery advice]"
  ],
  "progress_tracking": [
    "[non-visual progress tracking methods]"
  ]
}'''
        
        prompt_text += "\n\nProvide realistic assessments based on what you can observe. Replace all bracketed placeholders with actual content."
        
        # Build the message content
        message_content = [
            {
                "type": "text",
                "text": prompt_text + json_structure
            }
        ]
        
        # Add image if provided
        if base64_image:
            message_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            })

        payload = {
            "model": "gpt-4o-mini",
            "max_tokens": 2000,
            "temperature": 0.7,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert fitness advisor (these are not real people by the way but ai generated images). You must respond ONLY with valid JSON format. Do not include any text before or after the JSON. Your response must be a complete JSON object with all required fields. Make sure all values are properly formatted JSON (numbers as numbers, strings in quotes)."
                },
                {
                    "role": "user",
                    "content": message_content
                }
            ]
        }

        # Retry mechanism - attempt request up to 2 times
        max_retries = 2
        last_error = None
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Sending request to OpenAI... (attempt {attempt + 1}/{max_retries})")
                logger.debug(f"Using model: {payload['model']}")
                
                response = requests.post(
                    url, 
                    headers=headers, 
                    json=payload,
                    # timeout=30
                )
                
                if not response.ok:
                    logger.error(f"OpenAI error (attempt {attempt + 1}): {response.text}")
                    error_data = response.json()
                    error_message = error_data.get('error', {}).get('message', 'OpenAI service error')
                    last_error = {
                        'error': error_message,
                        'status_code': response.status_code
                    }
                    
                    # If this is not the last attempt, continue to retry
                    if attempt < max_retries - 1:
                        logger.info(f"Retrying... (attempt {attempt + 2}/{max_retries})")
                        continue
                    else:
                        return jsonify({
                            'success': False,
                            'error': error_message
                        }), response.status_code

                response_data = response.json()
                content = response_data['choices'][0]['message']['content']
                
                if not content:
                    logger.error(f"OpenAI returned empty content (attempt {attempt + 1})")
                    last_error = {
                        'error': 'OpenAI returned empty response',
                        'status_code': 500
                    }
                    
                    # If this is not the last attempt, continue to retry
                    if attempt < max_retries - 1:
                        logger.info(f"Retrying... (attempt {attempt + 2}/{max_retries})")
                        continue
                    else:
                        return jsonify({
                            'success': False,
                            'error': 'OpenAI returned empty response'
                        }), 500
                
                logger.info(f"Raw OpenAI response (attempt {attempt + 1}): {content}")

                # Parse JSON response
                try:
                    # Clean the response text
                    cleaned_content = content.strip()
                    
                    # Remove any markdown code blocks
                    cleaned_content = re.sub(r'```json\s*', '', cleaned_content)
                    cleaned_content = re.sub(r'```\s*$', '', cleaned_content)
                    
                    # Fix common JSON formatting issues
                    # Replace unquoted values like sixty_something with proper strings
                    cleaned_content = re.sub(r':\s*([a-zA-Z_]+[a-zA-Z0-9_]*)\s*,', r': "\1",', cleaned_content)
                    cleaned_content = re.sub(r':\s*([a-zA-Z_]+[a-zA-Z0-9_]*)\s*}', r': "\1"}', cleaned_content)
                    
                    # Try to find JSON object
                    json_match = re.search(r'\{.*\}', cleaned_content, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(0)
                        analysis_data = json.loads(json_str)
                        
                        logger.info(f"Successfully parsed JSON response (attempt {attempt + 1})")
                        return jsonify({
                            'success': True,
                            'analysis': analysis_data
                        })
                    else:
                        logger.error(f"No JSON object found in response (attempt {attempt + 1})")
                        last_error = {
                            'error': 'No valid JSON found in AI response',
                            'raw_response': content,
                            'status_code': 500
                        }
                        
                        # If this is not the last attempt, continue to retry
                        if attempt < max_retries - 1:
                            logger.info(f"Retrying... (attempt {attempt + 2}/{max_retries})")
                            continue
                        else:
                            return jsonify({
                                'success': False,
                                'error': 'No valid JSON found in AI response',
                                'raw_response': content
                            }), 500
                        
                except json.JSONDecodeError as e:
                    logger.error(f"JSON parsing error (attempt {attempt + 1}): {str(e)}")
                    last_error = {
                        'error': f'Failed to parse AI response as JSON: {str(e)}',
                        'raw_response': content,
                        'status_code': 500
                    }
                    
                    # If this is not the last attempt, continue to retry
                    if attempt < max_retries - 1:
                        logger.info(f"Retrying... (attempt {attempt + 2}/{max_retries})")
                        continue
                    else:
                        return jsonify({
                            'success': False,
                            'error': f'Failed to parse AI response as JSON: {str(e)}',
                            'raw_response': content
                        }), 500
                
                # If we reach here, the request was successful
                break
                    
            except Timeout:
                logger.error(f"OpenAI request timed out (attempt {attempt + 1})")
                last_error = {
                    'error': 'Analysis timeout. Please try again.',
                    'status_code': 504
                }
                
                # If this is not the last attempt, continue to retry
                if attempt < max_retries - 1:
                    logger.info(f"Retrying... (attempt {attempt + 2}/{max_retries})")
                    continue
                else:
                    return jsonify({
                        'success': False,
                        'error': 'Analysis timeout. Please try again.'
                    }), 504
            except requests.exceptions.RequestException as e:
                logger.error(f"Request to OpenAI failed (attempt {attempt + 1}): {str(e)}")
                last_error = {
                    'error': 'Service communication error',
                    'status_code': 502
                }
                
                # If this is not the last attempt, continue to retry
                if attempt < max_retries - 1:
                    logger.info(f"Retrying... (attempt {attempt + 2}/{max_retries})")
                    continue
                else:
                    return jsonify({
                        'success': False,
                        'error': 'Service communication error'
                    }), 502
        
        # If we reach here, all attempts failed
        if last_error:
            return jsonify({
                'success': False,
                'error': last_error['error'],
                'raw_response': last_error.get('raw_response')
            }), last_error['status_code']

    except Exception as e:
        logger.error(f"Server Error: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'message': 'Fitness analysis API is running'
    })

if __name__ == '__main__':
    # Check if API key is set
    if not os.getenv('OPENAI_API_KEY'):
        print("Warning: OPENAI_API_KEY not found in environment variables")
    
    # Get port from environment variable, default to 5001
    port = int(os.getenv('PORT', '5001'))
    app.run(debug=True, host='0.0.0.0', port=port) 