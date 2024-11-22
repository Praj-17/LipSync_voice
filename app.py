# main.py

from fastapi import FastAPI, UploadFile, File, Form
from typing import List
from celery.result import AsyncResult
from main import AiVideoGenerator
from celery_app import celery_app
import uuid
import boto3
import json
import os

app = FastAPI()
ai_generator = AiVideoGenerator()


# Celery task to generate AI video
@celery_app.task(bind=True)
def generate_ai_video_task(self, user_id, face_video_bytes, audio_url_list):
    try:
        
        outfile = ai_generator.generate_template_video(user_id, face_video_bytes, audio_url_list)
        print("Output ready at:", outfile)

        # Upload outfile to S3
        s3 = boto3.client('s3')
        bucket_name = 'your-s3-bucket-name'  # Replace with your S3 bucket name
        s3_file_key = f"ai_videos/{uuid.uuid4()}.mp4"

        s3.upload_file(outfile, bucket_name, s3_file_key)
        s3_url = f"https://{bucket_name}.s3.amazonaws.com/{s3_file_key}"

        # Clean up the local file
        os.remove(outfile)

        # Return the S3 URL
        return {'status': 'Task completed!', 'result': s3_url}

    except Exception as exc:
        self.update_state(state='FAILURE', meta={'status': str(exc)})
        raise

# Endpoint to start the AI video generation task
@app.post("/generate_ai_video")
async def generate_ai_video_endpoint(
    user_id: str = Form(...),
    face_video: UploadFile = File(...),
    audio_url_list: str = Form(...)
):
    # Parse the JSON string of audio URLs
    audio_url_list = json.loads(audio_url_list)

    # Read the face video file contents
    face_video_bytes = await face_video.read()

    # Start the Celery task
    task = generate_ai_video_task.delay(user_id, face_video_bytes, audio_url_list)

    # Return the task ID
    return {"task_id": task.id}

# Endpoint to check the status of the task
@app.get("/tasks/{task_id}")
def get_task_status(task_id: str):
    task_result = AsyncResult(task_id, app=celery_app)
    
    if task_result.state == 'PENDING':
        # Task is pending
        response = {
            'state': task_result.state,
            'status': 'Pending...'
        }
    elif task_result.state == 'SUCCESS':
        # Task completed successfully
        response = {
            'state': task_result.state,
            'status': task_result.info.get('status', ''),
            'result': task_result.info.get('result', '')
        }
    elif task_result.state == 'FAILURE':
        # Task failed
        response = {
            'state': task_result.state,
            'status': task_result.info.get('status', 'An error occurred')
        }
    else:
        # Task is in progress
        response = {
            'state': task_result.state,
            'status': task_result.info.get('status', '')
        }
    return response
