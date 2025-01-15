from fastapi import FastAPI, Request
from fastapi.responses import Response
from typing import Callable
from . import BaitRoute

def register_with_fastapi(app: FastAPI, baitroute: BaitRoute) -> None:
    """Register bait endpoints with a FastAPI application.
    
    Args:
        app: FastAPI application instance
        baitroute: BaitRoute instance containing the rules
    """
    @app.middleware("http")
    async def bait_middleware(request: Request, call_next: Callable):
        rule = baitroute.get_matching_rule(request.url.path, request.method)
        if rule:
            # Send alert if handler is configured
            if baitroute.alert_handler is not None:
                body = None
                if request.headers.get("content-type") == "application/json":
                    body = await request.json()
                elif request.headers.get("content-type") == "application/x-www-form-urlencoded":
                    body = await request.form()
                else:
                    body = await request.body()
                    if body:
                        body = body.decode()

                alert = baitroute.create_alert(
                    path=str(request.url.path),
                    method=request.method,
                    remote_addr=request.client.host,
                    headers=dict(request.headers),
                    body=body
                )
                baitroute.alert_handler(alert)

            # Return response according to rule
            headers = {'Content-Type': rule.get('content-type', 'text/plain')}
            if rule.get('headers'):
                headers.update(rule['headers'])

            return Response(
                content=rule.get('body', ''),
                status_code=rule.get('status', 200),
                headers=headers
            )

        return await call_next(request)