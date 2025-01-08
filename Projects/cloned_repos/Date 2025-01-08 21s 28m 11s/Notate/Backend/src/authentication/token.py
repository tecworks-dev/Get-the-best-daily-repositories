from fastapi import Depends, Request
from fastapi.security import OAuth2PasswordBearer
from typing import Optional
import os
import jwt
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Get secret from environment variable
SECRET_KEY = os.environ.get("JWT_SECRET")
if not SECRET_KEY:
    raise RuntimeError("JWT_SECRET environment variable is not set")


async def get_optional_token(token: Optional[str] = Depends(oauth2_scheme)):
    return token


async def verify_token(token: Optional[str] = Depends(get_optional_token)):
    if token is None:
        return None
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        print(f"Payload: {payload}")
        user_id: str = payload.get("userId")
        logger.info(f"User ID: {user_id}")
        if user_id is None:
            return None
        return user_id

    except jwt.exceptions.InvalidTokenError:
        logger.error("Invalid token")
        return None


async def optional_auth(request: Request):
    if "Authorization" in request.headers:
        token = request.headers["Authorization"].split("Bearer ")[1]
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
            return payload.get("userId")
        except jwt.exceptions.InvalidTokenError:
            return None
    return None
