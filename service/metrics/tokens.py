import logging
import os
from datetime import datetime, timedelta

from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from starlette import status

# to generate a master key run:
# openssl rand -hex 32
if "UNITXT_METRICS_MASTER_KEY" in os.environ:
    MASTER_KEY = os.environ["UNITXT_METRICS_MASTER_KEY"]
else:
    MASTER_KEY = None

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_DAYS = 360

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


log = logging.getLogger("tokens")


class InvalidTokenError(Exception):
    pass


def create_token(name: str):
    assert MASTER_KEY is not None

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
    return jwt.encode(payload, MASTER_KEY, algorithm=ALGORITHM)


def verify_jwt_token(jwt_token):
    try:
        if MASTER_KEY:
            payload = jwt.decode(jwt_token, MASTER_KEY, algorithms=[ALGORITHM])
            if payload["sub"] is None:
                raise InvalidTokenError("Token subject claim is empty")
            return payload
        return {"sub": "Anonymous"}
    except JWTError as e:
        raise InvalidTokenError from e


async def verify_token(token: str = Depends(oauth2_scheme)):
    try:
        return verify_jwt_token(token)
    except InvalidTokenError as e:
        log.exception(e)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        ) from e


def main():
    # for name in ["unitxt-metrics-service-tester", "rag-eval-unitest"]:
    #     print(name, create_token(name))
    pass


if __name__ == "__main__":
    main()
