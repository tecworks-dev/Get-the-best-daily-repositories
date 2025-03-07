package handler

import (
	"encoding/json"
	"net/http"
	"time"

	"github.com/angelicmuklu/saf/internal/domain/model/event"
	"github.com/angelicmuklu/saf/internal/infra/cmq"
	"github.com/angelicmuklu/saf/internal/infra/http/request"
	"github.com/gofiber/fiber/v2"
	"go.opentelemetry.io/otel/trace"
	"go.uber.org/zap"
)

type Event struct {
	CMQ    *cmq.CMQ
	Logger *zap.Logger
	Tracer trace.Tracer
}

// Receive receives event from api and publish them to jetstream.
// nolint: wrapcheck
func (h Event) Receive(c *fiber.Ctx) error {
	ctx, span := h.Tracer.Start(c.Context(), "handler.event",
		trace.WithSpanKind(trace.SpanKindServer),
	)
	defer span.End()

	var rq request.Event

	if err := c.BodyParser(&rq); err != nil {
		span.RecordError(err)

		return fiber.NewError(http.StatusBadRequest, err.Error())
	}

	if err := rq.Validate(); err != nil {
		span.RecordError(err)

		return fiber.NewError(http.StatusBadRequest, err.Error())
	}

	ev := event.Event{
		Subject:   rq.Subject,
		CreatedAt: time.Now(),
		Payload:   rq.Payload,
	}

	data, err := json.Marshal(ev)
	if err != nil {
		span.RecordError(err)

		return fiber.NewError(http.StatusInternalServerError, err.Error())
	}

	{
		ctx, span := h.Tracer.Start(
			ctx,
			"handler.event.publish",
			trace.WithSpanKind(trace.SpanKindProducer),
		)
		defer span.End()

		if err := h.CMQ.Publish(ctx, rq.ID, data); err != nil {
			span.RecordError(err)

			return fiber.NewError(http.StatusServiceUnavailable, err.Error())
		}
	}

	return c.Status(http.StatusOK).Send(nil)
}

// Register registers the routes of event handler on given fiber group.
func (h Event) Register(g fiber.Router) {
	g.Post("/event", h.Receive)
}
