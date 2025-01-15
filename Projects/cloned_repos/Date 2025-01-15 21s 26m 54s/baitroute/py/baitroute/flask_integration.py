from flask import Flask, request
from . import BaitRoute

def register_with_flask(app: Flask, baitroute: BaitRoute) -> None:
    """Register bait endpoints with a Flask application.
    
    Args:
        app: Flask application instance
        baitroute: BaitRoute instance containing the rules
    """
    @app.before_request
    def handle_bait_request():
        rule = baitroute.get_matching_rule(request.path, request.method)
        if rule:
            # Send alert if handler is configured
            if baitroute.alert_handler is not None:
                alert = baitroute.create_alert(
                    path=request.path,
                    method=request.method,
                    remote_addr=request.remote_addr,
                    headers=dict(request.headers),
                    body=request.get_data(as_text=True) if request.is_json else None
                )
                baitroute.alert_handler(alert)

            # Return response according to rule
            response = app.make_response((
                rule.get('body', ''),
                rule.get('status', 200),
                {'Content-Type': rule.get('content-type', 'text/plain')}
            ))

            # Add custom headers if specified
            if rule.get('headers'):
                for key, value in rule['headers'].items():
                    response.headers[key] = value

            return response