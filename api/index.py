from vercel_wsgi import make_lambda_handler
from app import app

handler = make_lambda_handler(app) 