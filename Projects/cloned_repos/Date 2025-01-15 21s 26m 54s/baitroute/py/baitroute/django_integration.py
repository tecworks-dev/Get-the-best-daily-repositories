from django.conf import settings
from django.http import HttpResponse
from . import BaitRoute, AlertHandler

class BaitRouteMiddleware:
    default_alert_handler: AlertHandler = None

    def __init__(self, get_response):
        self.get_response = get_response

        # Get rules directory from settings
        rules_dir = getattr(settings, 'BAITROUTE_RULES_DIR', None)
        if not rules_dir:
            raise ValueError("BAITROUTE_RULES_DIR must be set in Django settings")

        # Get optional selected rules
        selected_rules = getattr(settings, 'BAITROUTE_SELECTED_RULES', None)

        # Create baitroute instance
        self.baitroute = BaitRoute(rules_dir, selected_rules=selected_rules)

        # Set default alert handler if provided
        if BaitRouteMiddleware.default_alert_handler:
            self.baitroute.on_bait_hit(BaitRouteMiddleware.default_alert_handler)

    def __call__(self, request):
        # Check if this is a bait endpoint
        rule = self.baitroute.get_matching_rule(request.path, request.method)
        if rule:
            # Send alert if handler is configured
            if self.baitroute.alert_handler is not None:
                alert = self.baitroute.create_alert(
                    path=request.path,
                    method=request.method,
                    remote_addr=request.META.get('REMOTE_ADDR', ''),
                    headers=dict(request.headers),
                    body=request.body.decode() if request.body else None
                )
                self.baitroute.alert_handler(alert)

            # Return response according to rule
            response = HttpResponse(
                content=rule.get('body', ''),
                status=rule.get('status', 200),
                content_type=rule.get('content-type', 'text/plain')
            )

            # Add custom headers if specified
            if rule.get('headers'):
                for key, value in rule['headers'].items():
                    response[key] = value

            return response

        return self.get_response(request)