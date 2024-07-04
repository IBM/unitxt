"""This module handles authorization tokens for a service.

To generate a master token key, run "openssl rand -hex 32".
Then, save the value in the environment variable UNITXT_METRICS_MASTER_KEY_TOKEN.
To create tokens that have access for the master key, use create_token(..), as shown in main().
"""

from datetime import datetime, timedelta

from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from starlette import status

from ...logging_utils import get_logger
from ...settings_utils import get_settings

settings = get_settings()
logger = get_logger()


ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_DAYS = 360


class InvalidTokenError(Exception):
    pass


def create_token(name: str):
    assert settings.metrics_master_key_token is not None

    # create the token data
    now = datetime.utcnow()
    expires_delta = timedelta(days=ACCESS_TOKEN_EXPIRE_DAYS)
    payload = {
        "iss": "Unitxt Metrics",
        "sub": name,
        "iat": now,
        "exp": now + expires_delta,
    }

    # generate the jwt token and return it
    return jwt.encode(payload, settings.metrics_master_key_token, algorithm=ALGORITHM)


def verify_jwt_token(jwt_token):
    try:
        if settings.metrics_master_key_token:
            payload = jwt.decode(
                jwt_token, settings.metrics_master_key_token, algorithms=[ALGORITHM]
            )
            if payload["sub"] is None:
                raise InvalidTokenError("Token subject claim is empty")
            return payload
        return {"sub": "Anonymous"}
    except JWTError as e:
        raise InvalidTokenError from e


# This object makes sure that the incoming HTTP request has a header with
# an authorization token (e.g. passed with 'curl -H "Authorization: Bearer {token}"').
# It does NOT check that the token has a valid value (that is done by verify_token(..)).
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


async def verify_token(token: str = Depends(oauth2_scheme)):
    try:
        return verify_jwt_token(token)
    except InvalidTokenError as e:
        logger.exception(e)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        ) from e


def main():
    name = "unitxt-metrics-service-tester"
    logger.info(f"{name}: {create_token(name)}")


if __name__ == "__main__":
    main()
